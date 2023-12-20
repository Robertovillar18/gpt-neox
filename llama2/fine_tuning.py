# %%
import os
os.system('huggingface-cli login --token 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')
os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = 'true'

# %%
import torch, logging
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, HfArgumentParser, TrainingArguments, pipeline
from peft import LoraConfig, PeftModel
from trl import SFTTrainer
from peft import AutoPeftModelForCausalLM

# %%
import pandas as pd
import json
from datasets import Dataset
from random import randrange
from functools import partial
#from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# %%
# Model and tokenizer names
base_model_name = "meta-llama/Llama-2-7b-chat-hf"
refined_model = "llama-2-7b-enhanced"

def load_model(model_name_):
  device = f'cuda:{torch.cuda.current_device()}' if torch.cuda.is_available() else 'cpu'
  print('La ejecución se realizará en:',device)

  # Tokenizer
  llama_tokenizer = AutoTokenizer.from_pretrained(model_name_, trust_remote_code=True)
  llama_tokenizer.pad_token = llama_tokenizer.eos_token
  llama_tokenizer.padding_side = "right"  # Fix for fp16
  llama_tokenizer.device_map = device

  # Quantization Config
  quant_config = BitsAndBytesConfig(
      load_in_4bit=True,
      bnb_4bit_quant_type="nf4",
      bnb_4bit_compute_dtype=torch.float16,
      bnb_4bit_use_double_quant=False
  )

  # Model
  base_model = AutoModelForCausalLM.from_pretrained(
      model_name_,
      quantization_config=quant_config,
      device_map=device
  )

  return base_model, llama_tokenizer

# %%
base_model, llama_tokenizer = load_model(base_model_name)

# %%
def test_model(query_, model_, tokenizer_): 
  device = f'cuda:{torch.cuda.current_device()}' if torch.cuda.is_available() else 'cpu'
  print('La ejecución se realizará en:', device)

  model = model_

  sys = '<<SYS>>Eres un asistente de tramites del estado uruguayo. Das respuestas lo mas resumidas posible.'+\
  'Eres un asistente que responde en español.'+\
  'Eres un asistente servicial, respetuoso y honesto. Responde siempre de forma resumida de la manera más útil posible y siendo seguro.'+\
  'Tus respuestas no deben incluir ningún contenido dañino, poco ético, racista, sexista, tóxico, peligroso o ilegal.'+\
  'Asegúrese de que sus respuestas sean socialmente imparciales y de naturaleza positiva.'+\
  'Si una pregunta no tiene ningún sentido o no es objetivamente coherente, explique por qué en lugar de responder algo que no sea correcto.'+\
  'Si no sabe la respuesta a una pregunta, no comparta información falsa.'+\
  'Solo respondes en español con información relacionada al pais Uruguay.'

  query = query_
  inputs = tokenizer_(f'<s>[INST] {sys}" input:" {query} [/INST] response:', return_tensors="pt").to(device)
  outputs = model.generate(input_ids=inputs["input_ids"].to("cuda"),
                            attention_mask=inputs["attention_mask"],
                            max_new_tokens=500,
                            pad_token_id=tokenizer_.eos_token_id,
                            temperature=0.1,
                            do_sample=True,
                            top_k=20,
                            num_return_sequences=1,
                            repetition_penalty=1.1
                          )
  response = tokenizer_.decode(outputs[0], skip_special_tokens=True)
  return str(response.split('response:')[-1])

qry = "Donde me puedo sacar la cedula online en uruguay?"
print('query test:', qry)
print('-------------------------------------------------------------')
print(test_model(qry, base_model, llama_tokenizer))

# %%
base_model.config.use_cache = False
base_model.config.pretraining_tp = 1

# %%
data_name = "mlabonne/guanaco-llama2-1k"
data = load_dataset(data_name)['train']

# %%
dataset_name = 'tramite_cedula.json'
f = open(dataset_name,  encoding='UTF-8')
json_data = json.load(f)
df_train = pd.DataFrame(json_data['train'])

dataset = Dataset.from_pandas(df_train)

print(f'Number of prompts: {len(dataset)}')
print(f'Column names are: {dataset.column_names}')
dataset[randrange(len(dataset))]

# %%
def create_prompt_formats(sample):

  # Initialize static strings for the prompt template
  S_KEY_INIT = '<s>'
  INSTRUCTION_KEY_INIT = '[INST] '
  INSTRUCTION_KEY_END = ' [/INST] '
  S_KEY_END = '</s>'

  input_text = f'{S_KEY_INIT}{INSTRUCTION_KEY_INIT}'
  input_text += f'{sample["instruction"]}{INSTRUCTION_KEY_END}'
  lst_data = [sample["output"]]

  text = "\n".join(lst_data)
  # print(response)
  formatted_prompt = f'{input_text}{text} {S_KEY_END}'

  global data
  data = data.add_item({'text':formatted_prompt})

  # Store the formatted prompt template in a new key “text”
  sample['text'] = formatted_prompt

  return sample


# %%
dataset.map(create_prompt_formats)

# %%
print(data)

# %%
def get_max_length(model):
  # Pull model configuration
  conf = model.config
  # Initialize a “max_length” variable to store maximum sequence length as null
  max_length = None
  # Find maximum sequence length in the model configuration and save it in 'max_length' if found
  for length_setting in ['n_positions', 'max_position_embeddings', 'seq_length']:
    max_length = getattr(model.config, length_setting, None)
    if max_length:
      print(f'Found max lenth: {max_length}')
      break
  # Set “max_length” to 1024 (default value) if maximum sequence length is not found in the model configuration
  if not max_length:
    max_length = 1024
    print(f'Using default max length: {max_length}')
  return max_length

get_max_length(base_model)

# %%
## Question and Answer Task
def compute_metrics(pred):
    squad_labels = pred.label_ids
    squad_preds = pred.predictions.argmax(-1)

    # Calculate Exact Match (EM)
    em = sum([1 if any(p == l) else 0 for p, l in zip(squad_preds, squad_labels)]) / len(squad_labels)

    # Calculate F1-score
    f1 = f1_score(squad_labels[0], squad_preds[0], average='macro')

    return {
        'exact_match': em,
        'f1': f1
    }

# %%
splited_data = data.train_test_split(test_size=0.2)
training_data = splited_data['train']
evaluation_data = splited_data['test']
print('Total:', len(data),'Train:',len(training_data), 'Eval:',len(evaluation_data))

# %%
# LoRA Config
peft_parameters = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.1,
    r=8,
    bias="none",
    task_type="CAUSAL_LM"
)
# Training Params
train_params = TrainingArguments(
    output_dir="./results_modified",
    num_train_epochs=1,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=1,
    optim="paged_adamw_32bit",
    save_steps=25,
    logging_steps=25,
    learning_rate=2e-4,
    weight_decay=0.001,
    fp16=False,
    bf16=False,
    max_grad_norm=0.3,
    max_steps=-1,
    warmup_ratio=0.03,
    group_by_length=True,
    lr_scheduler_type="constant"
)
# Trainer
fine_tuning = SFTTrainer(
    model=base_model,
    train_dataset=training_data,
    peft_config=peft_parameters,
    dataset_text_field="text",
    tokenizer=llama_tokenizer,
    args=train_params,
    compute_metrics=compute_metrics,
    eval_dataset=evaluation_data,
    use_reentrant=True
)
# Training
fine_tuning.train()
# Save Model
fine_tuning.model.save_pretrained(refined_model)


