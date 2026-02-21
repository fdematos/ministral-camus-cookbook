import sys
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoProcessor, Mistral3ForConditionalGeneration

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_ID = str(PROJECT_ROOT / "ministral-8b")
ARTIFACTS_DIR = PROJECT_ROOT / "ministral-8b-albert-camus"
LORA_PATH = ARTIFACTS_DIR / "lora-style"
MERGED_OUTPUT = ARTIFACTS_DIR / "model-style-merged"


def main():
    print("=== Merge Phase 1 : Base + LoRA Style -> bf16 ===\n")

    if not LORA_PATH.exists():
        print(f"ERREUR: LoRA non trouvé: {LORA_PATH}", file=sys.stderr)
        sys.exit(1)

    if not (LORA_PATH / "adapter_config.json").exists():
        print(
            f"ERREUR: adapter_config.json introuvable dans {LORA_PATH}",
            file=sys.stderr,
        )
        sys.exit(1)

    if MERGED_OUTPUT.exists():
        print(f"ATTENTION: {MERGED_OUTPUT} existe déjà, sera écrasé\n")

    print(f"Chargement du modèle base en bf16 ({MODEL_ID})...")
    base_model = Mistral3ForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    print("  OK\n")

    print(f"Chargement du LoRA Phase 1 ({LORA_PATH.name})...")
    model = PeftModel.from_pretrained(base_model, str(LORA_PATH))
    print("  OK\n")

    print("Merge des poids LoRA dans le modèle de base...")
    merged_model = model.merge_and_unload()
    print("  OK\n")

    print(f"Sauvegarde du modèle mergé vers {MERGED_OUTPUT}...")
    print("  (peut prendre quelques minutes)\n")
    merged_model.save_pretrained(str(MERGED_OUTPUT))

    print("Sauvegarde du processor/tokenizer...")
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    processor.save_pretrained(str(MERGED_OUTPUT))

    print(f"\nMerge terminé: {MERGED_OUTPUT}")
    print("Done.")


if __name__ == "__main__":
    main()
