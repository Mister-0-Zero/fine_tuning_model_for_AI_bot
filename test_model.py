from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_DIR = "./tinyllama-fp16-merged"
BASE_DIR  = "./tinyllama-base-fp16"     # где лежит tokenizer.model

tok = AutoTokenizer.from_pretrained(
    BASE_DIR,          # ← берем токенизатор из базы
    trust_remote_code=True,
    use_fast=False      # явно запрещаем конвертацию
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_DIR,
    torch_dtype="auto",
    device_map="auto",
)

out = model.generate(**tok("Привет!", return_tensors="pt"), max_new_tokens=40)
print(tok.decode(out[0], skip_special_tokens=True))