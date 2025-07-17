from transformers import AutoModelForCausalLM
AutoModelForCausalLM.from_pretrained(
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    torch_dtype="auto"
).save_pretrained("./tinyllama-base-fp16")
print("✅ базовая модель сохранена в ./tinyllama-base-fp16")