# %%
import sys, getopt

# %%
import transformers
import torch

from torch import cuda, bfloat16
from transformers import AutoTokenizer, AutoConfig, LlamaForCausalLM, LlamaTokenizer
from transformers StoppingCriteria, StoppingCriteriaList

# %%
model_id = "meta-llama/Llama-2-7b-chat-hf"

device = f'cuda:{cuda.current_device()}' if cuda.is_available() else 'cpu'
# device = 'auto'
print('La ejecución se realizará en:',device)

# %%
tokenizer = AutoTokenizer.from_pretrained(model_id)

# %%
stop_list = ['\ninput:', '\n```\n']

stop_token_ids = [tokenizer(x)['input_ids'] for x in stop_list]
stop_token_ids

stop_token_ids = [torch.LongTensor(x).to(device) for x in stop_token_ids]
stop_token_ids

# define custom stopping criteria object
class StopOnTokens(StoppingCriteria):
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        for stop_ids in stop_token_ids:
            if torch.eq(input_ids[0][-len(stop_ids):], stop_ids).all():
                return True
        return False

stopping_criteria = StoppingCriteriaList([StopOnTokens()])

# %%
pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    torch_dtype=torch.float16,
    # stopping_criteria=stopping_criteria,
    temperature=0.1,  # 'randomness' of outputs, 0.0 is the min and 1.0 the max
    repetition_penalty=1.1,  # without this output begins repeating
    # return_full_text=True,  # langchain expects the full text
    device_map=device,
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

# %%

default_prompt = '<s>[INST]'+\
'<<SYS>>'+\
'Eres un asistente de tramites del estado uruguayo. Das respuestas lo mas resumidas posible.'+\
'Eres un asistente que responde en español.'+\
'Eres un asistente servicial, respetuoso y honesto. Responde siempre de forma resumida de la manera más útil posible y siendo seguro.'+\
'Tus respuestas no deben incluir ningún contenido dañino, poco ético, racista, sexista, tóxico, peligroso o ilegal.'+\
'Asegúrese de que sus respuestas sean socialmente imparciales y de naturaleza positiva.'+\
'Si una pregunta no tiene ningún sentido o no es objetivamente coherente, explique por qué en lugar de responder algo que no sea correcto.'+\
'Si no sabe la respuesta a una pregunta, no comparta información falsa.'+\
'Solo hablas y escribes en español.'+\
'Dirección de trámites del estado de uruguay: https://www.gub.uy/tramites/.'+\
'<</SYS>>\n'

user_input = '\ninput:{input}'
last_part_promp = '</s>'

conversation = ''
def dialog(text):

    global conversation
    global default_prompt

    user_text = str(user_input).replace('{input}', text)
    conversation += user_text

    prompt = default_prompt + conversation + '[/INST] \nresponse: </s>'

    sequences = pipeline(
        prompt,
        do_sample=True,
        top_k=10,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id,
        max_length=1024
    )
    get_clean_answer = ''
    for seq in sequences:
        row_answer = str(seq['generated_text'])
        get_clean_answer =  row_answer.split('[/INST]')[-1].strip()
        get_clean_answer =  get_clean_answer.split('</s>')[-1].strip()
        conversation += '\nresponse:' + get_clean_answer
        return get_clean_answer
    return get_clean_answer



# %%
def main(argv, device):
    input = ''
    output = ''

    try:
        opts, args = getopt.getopt(argv,"hi:o:",["input=","output="])
    except getopt.GetoptError:
        print ('generate_llama2.py -d')
        sys.exit(2)

    for opt, arg in opts:
        if opt in ("-i", "--input"):
            input = arg
        elif opt in ("-o", "--output"):
            output = arg
        elif opt in ("-d"):
            text = ''
            while text != 'chau':
                text = input()
                result = dialog(text)
                print(result)
                
if __name__ == "__main__":
   main(sys.argv[1:], device)


