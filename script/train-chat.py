import argparse
import sys
from pathlib import Path

from datasets import concatenate_datasets, load_dataset
from unsloth import FastVisionModel
from trl import SFTTrainer, SFTConfig

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = PROJECT_ROOT / "ministral-8b-albert-camus"
DATASET_DIR = PROJECT_ROOT / "albert-camus-chat-style-chat-dataset" / "phase2"

MERGED_MODEL_PATH = ARTIFACTS_DIR / "model-style-merged"
OUTPUT_DIR = ARTIFACTS_DIR / "lora-chat"
DATASET_CORPUS_DEFAULT_PATH = DATASET_DIR / "chat-pairs-corpus-final-clean.jsonl"
DATASET_LIGHT_DEFAULT_PATH = DATASET_DIR / "chat-pairs-light-boost-clean.jsonl"

MAX_SEQ_LENGTH = 2048


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-corpus-path",
        type=str,
        default=str(DATASET_CORPUS_DEFAULT_PATH),
        help="Chemin vers le dataset corpus clean (Phase 2)",
    )
    parser.add_argument(
        "--dataset-light-path",
        type=str,
        default=str(DATASET_LIGHT_DEFAULT_PATH),
        help="Chemin vers le dataset light clean (Phase 2)",
    )
    return parser.parse_args()


def validate_inputs(dataset_files):
    if not MERGED_MODEL_PATH.exists():
        print(
            f"ERREUR: Modèle mergé non trouvé: {MERGED_MODEL_PATH}\n"
            f"  Lancer d'abord: python script/merge-phase1.py",
            file=sys.stderr,
        )
        sys.exit(1)

    for path in dataset_files:
        if not path.exists():
            print(f"ERREUR: Dataset non trouvé: {path}", file=sys.stderr)
            sys.exit(1)


def load_chat_datasets(dataset_files):
    datasets = []
    for path in dataset_files:
        ds = load_dataset("json", data_files=str(path), split="train")
        print(f"  {path.name}: {len(ds)} exemples")
        datasets.append(ds)

    combined = concatenate_datasets(datasets)
    combined = combined.shuffle(seed=42)
    print(f"  Total: {len(combined)} exemples (shuffled)")

    if "messages" not in combined.column_names:
        print(
            "ERREUR: le dataset ne contient pas la colonne 'messages'",
            file=sys.stderr,
        )
        sys.exit(1)

    return combined


def make_formatting_func(tokenizer):
    def formatting_func(examples):
        texts = []
        for messages in examples["messages"]:
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )
            texts.append(text)
        return {"text": texts}

    return formatting_func


def main():
    args = parse_args()
    dataset_files = [
        Path(args.dataset_corpus_path),
        Path(args.dataset_light_path),
    ]

    print("=== Phase 2 : Chat Tuning LoRA ===\n")

    validate_inputs(dataset_files)

    print(f"Chargement du modèle mergé ({MERGED_MODEL_PATH.name})...")
    model, tokenizer = FastVisionModel.from_pretrained(
        model_name=str(MERGED_MODEL_PATH),
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=None,
        load_in_4bit=True,
    )
    print("  OK\n")

    print("Configuration LoRA Phase 2...")
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
        f"  Params: {trainable:,} trainable / {total:,} total "
        f"({100 * trainable / total:.2f}%)"
    )
    print("  OK\n")

    print("Chargement des datasets...")
    dataset = load_chat_datasets(dataset_files)

    sample = dataset[0]["messages"]
    print(f"  Rôles: {[m['role'] for m in sample]}")
    print(f"  System: {sample[0]['content'][:80]}...")
    print(f"  User: {sample[1]['content'][:80]}...")
    print()

    print("Application du chat template...")
    formatting_func = make_formatting_func(tokenizer)
    dataset = dataset.map(formatting_func, batched=True, remove_columns=["messages"])
    print(f"  Exemple formaté (début): {repr(dataset[0]['text'][:200])}...")
    print()

    print("Lancement de l'entraînement...\n")

    training_args = SFTConfig(
        output_dir=str(OUTPUT_DIR),
        num_train_epochs=3,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=2e-5,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
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

    print(f"\nSauvegarde du LoRA Phase 2 dans {OUTPUT_DIR}...")
    model.save_pretrained(str(OUTPUT_DIR))
    tokenizer.save_pretrained(str(OUTPUT_DIR))
    print("Done.")


if __name__ == "__main__":
    main()
