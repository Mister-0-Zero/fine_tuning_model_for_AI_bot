from pathlib import Path
import shutil, json, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

                
LORA_DIRS = [                                       
    "./checkpoints/step_dialogue/checkpoint-8200",
    "./checkpoints/step_asdiv",
    "./checkpoints/step_math",
    "./checkpoints/step_math",
    "./checkpoints/step_mawps",
    "./checkpoints/step_openbookqa",
]

BASE_FP16 = Path("./tinyllama-base-fp16")
OUT_DIR   = Path("./tinyllama-fp16-merged")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# загружаем базу FP16
model = AutoModelForCausalLM.from_pretrained(
    BASE_FP16, torch_dtype=torch.float16, device_map="cpu"
)

# применяем все адаптеры по очереди
for lora in LORA_DIRS:
    model = PeftModel.from_pretrained(model, lora, torch_dtype=torch.float16)
    model = model.merge_and_unload()                # слили и сняли PEFT-обёртку

# сохраняем итог
model.save_pretrained(OUT_DIR, safe_serialization=True)

# копируем файлы токенизатора из базы
for fname in [
    "tokenizer.model",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "chat_template.jinja",
]:
    src = BASE_FP16 / fname
    if src.exists():
        shutil.copy(src, OUT_DIR / fname)

# удаляем quantization_config на всякий случай
cfg_p = OUT_DIR / "config.json"
cfg   = json.loads(cfg_p.read_text())
cfg.pop("quantization_config", None)
cfg_p.write_text(json.dumps(cfg, indent=2, ensure_ascii=False))

print("✅ FP16-модель с учётом всех LoRA сохранена в", OUT_DIR)