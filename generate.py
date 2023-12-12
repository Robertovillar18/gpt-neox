# %%
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import sys, getopt

# %%
model_id = "EleutherAI/gpt-neox-20b"
device = f'cuda:{torch.cuda.current_device()}' if torch.cuda.is_available() else 'cpu'
print('La ejecución se realizará en:',device)

# %%
def load_model(device):

    model_id = "EleutherAI/gpt-neox-20b"
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config, device_map=device)

    return model, tokenizer

def generate(input, device):

    model, tokenizer = load_model(device)
    inputs = tokenizer(input, return_tensors="pt").to(device)
    outputs = model.generate(**inputs, max_new_tokens=20)
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return result

# %%
def main(argv):
    input = ''
    output = ''

    try:
        opts, args = getopt.getopt(argv,"hi:o:",["input=","output="])
    except getopt.GetoptError:
        print ('test.py -i <input> -o <output>')
        sys.exit(2)

    for opt, arg in opts:
        if opt in ("-i", "--input"):
            input = arg
        elif opt in ("-o", "--output"):
            output = arg

    if input != '':
        generated = generate(input)
        print(generated)

if __name__ == "__main__":
   main(sys.argv[1:])


