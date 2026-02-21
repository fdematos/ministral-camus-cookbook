import sys
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoProcessor, Mistral3ForConditionalGeneration

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BASE_PATH = PROJECT_ROOT / "ministral-8b"
ARTIFACTS_DIR = PROJECT_ROOT / "ministral-8b-albert-camus"
STYLE_LORA_PATH = ARTIFACTS_DIR / "lora-style"
CHAT_LORA_PATH = ARTIFACTS_DIR / "lora-chat"
FINAL_OUTPUT = ARTIFACTS_DIR / "model-final"


def main():
    print("=== Merge unique : Base + LoRA Style + LoRA Chat -> camus-final ===\n")

    for path, name in [
        (BASE_PATH, "base model"),
        (STYLE_LORA_PATH, "LoRA Phase 1 (style)"),
        (CHAT_LORA_PATH, "LoRA Phase 2 (chat)"),
    ]:
        if not path.exists():
            print(f"ERREUR: {name} non trouvé: {path}", file=sys.stderr)
            sys.exit(1)

    if FINAL_OUTPUT.exists():
        print(f"ATTENTION: {FINAL_OUTPUT} existe déjà, sera écrasé\n")

    print(f"Chargement du modèle base en bf16 ({BASE_PATH.name})...")
    model = Mistral3ForConditionalGeneration.from_pretrained(
        str(BASE_PATH),
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    print("  OK\n")

    print(f"Application du LoRA Phase 1 ({STYLE_LORA_PATH.name})...")
    model = PeftModel.from_pretrained(model, str(STYLE_LORA_PATH))
    print("  OK\n")

    print("Merge Phase 1 en mémoire...")
    model = model.merge_and_unload()
    print("  OK\n")

    print(f"Application du LoRA Phase 2 ({CHAT_LORA_PATH.name})...")
    model = PeftModel.from_pretrained(model, str(CHAT_LORA_PATH))
    print("  OK\n")

    print("Merge Phase 2 en mémoire...")
    model = model.merge_and_unload()
    print("  OK\n")

    print(f"Sauvegarde du modèle final vers {FINAL_OUTPUT}...")
    print("  (peut prendre quelques minutes)\n")
    model.save_pretrained(str(FINAL_OUTPUT))

    print("Sauvegarde du processor/tokenizer...")
    processor = AutoProcessor.from_pretrained(str(BASE_PATH))
    processor.save_pretrained(str(FINAL_OUTPUT))

    print(f"\nMerge terminé: {FINAL_OUTPUT}")
    print("Done.")


if __name__ == "__main__":
    main()
