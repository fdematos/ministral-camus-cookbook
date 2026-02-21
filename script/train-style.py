import argparse
import sys
from pathlib import Path

from datasets import load_dataset
from unsloth import FastVisionModel
from trl import SFTTrainer, SFTConfig

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_DEFAULT_PATH = (
    PROJECT_ROOT
    / "albert-camus-chat-style-chat-dataset"
    / "phase1"
    / "style-train.jsonl"
)
ARTIFACTS_DIR = PROJECT_ROOT / "ministral-8b-albert-camus"
OUTPUT_DIR = ARTIFACTS_DIR / "lora-style"

MODEL_ID = str(PROJECT_ROOT / "ministral-8b")
MAX_SEQ_LENGTH = 2048


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-path",
        type=str,
        default=str(DATASET_DEFAULT_PATH),
        help="Chemin vers le dataset Phase 1 JSONL",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    dataset_path = Path(args.dataset_path)

    print("=== Phase 1 : Style Pretraining LoRA ===\n")

    if not dataset_path.exists():
        print(f"ERREUR: Dataset non trouvé: {dataset_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Chargement du modèle ({MODEL_ID})...")
    model, tokenizer = FastVisionModel.from_pretrained(
        model_name=MODEL_ID,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=None,
        load_in_4bit=True,
    )
    print("  OK\n")

    print("Configuration LoRA...")
    model = FastVisionModel.get_peft_model(
        model,
        r=32,
        lora_alpha=64,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        finetune_vision_layers=False,
        finetune_language_layers=True,
        random_state=42,
    )

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(
        f"  Params: {trainable:,} trainable / {total:,} total ({100 * trainable / total:.2f}%)"
    )
    print("  OK\n")

    print(f"Chargement du dataset ({dataset_path})...")
    dataset = load_dataset("json", data_files=str(dataset_path), split="train")
    print(f"  {len(dataset)} exemples")

    if "text" not in dataset.column_names:
        print("ERREUR: le dataset ne contient pas la colonne 'text'", file=sys.stderr)
        sys.exit(1)

    sample = dataset[0]["text"]
    print(f"  Exemple (début): {repr(sample[:100])}...")
    print(f"  Exemple (fin):   ...{repr(sample[-40:])}")
    print()

    print("Lancement de l'entraînement...\n")

    training_args = SFTConfig(
        output_dir=str(OUTPUT_DIR),
        num_train_epochs=2,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        learning_rate=1e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        optim="adamw_8bit",
        fp16=False,
        bf16=True,
        logging_steps=1,
        save_strategy="epoch",
        save_total_limit=2,
        seed=42,
        max_seq_length=MAX_SEQ_LENGTH,
        dataset_text_field="text",
        packing=False,
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        args=training_args,
    )

    trainer.train()

    print(f"\nSauvegarde du LoRA dans {OUTPUT_DIR}...")
    model.save_pretrained(str(OUTPUT_DIR))
    tokenizer.save_pretrained(str(OUTPUT_DIR))
    print("Done.")


if __name__ == "__main__":
    main()
