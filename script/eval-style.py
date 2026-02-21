import argparse
import sys
from pathlib import Path

import torch
from unsloth import FastVisionModel
from transformers import TextStreamer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = PROJECT_ROOT / "ministral-8b-albert-camus"
LORA_PATH = ARTIFACTS_DIR / "lora-style"
MERGED_PATH = ARTIFACTS_DIR / "model-style-merged"
MODEL_ID = str(PROJECT_ROOT / "ministral-8b")

MAX_NEW_TOKENS = 150
TEMPERATURE = 0.7
TOP_P = 0.9

STYLE_PROMPTS = [
    "L'absurde naît quand",
    "Il faut imaginer Sisyphe",
    "Dans ce monde sans hiérarchie",
    "La révolte est",
    "L'homme est condamné à",
    "Le soleil couchant inondait",
]

MEMORIZATION_TESTS = [
    ("L'Étranger", "Aujourd'hui, maman est morte"),
    ("Sisyphe", "Il n'y a qu'un problème philosophique vraiment sérieux"),
    ("La Peste", "On n'en sort jamais"),
    ("L'Homme révolté", "Toute révolte est un acte de foi"),
]

MODERN_TOPICS = [
    "L'intelligence artificielle remplace-t-elle l'homme ?",
    "Que penses-tu de la société de consommation moderne ?",
    "L'écologie est-elle une nouvelle forme de religion ?",
    "Le travail à distance change-t-il notre rapport au monde ?",
]


def load_model(use_merged=False):
    if use_merged:
        print("=== Chargement du modèle mergé ===\n")
        print(f"Modèle: {MERGED_PATH}")

        model, tokenizer = FastVisionModel.from_pretrained(
            model_name=str(MERGED_PATH),
            max_seq_length=2048,
            dtype=None,
            load_in_4bit=True,
        )

        FastVisionModel.for_inference(model)
        print("  OK\n")
        return model, tokenizer

    print("=== Chargement du modèle + LoRA ===\n")

    print(f"Modèle base: {MODEL_ID}")
    print(f"LoRA: {LORA_PATH.name}")

    model, tokenizer = FastVisionModel.from_pretrained(
        model_name=MODEL_ID,
        max_seq_length=2048,
        dtype=None,
        load_in_4bit=True,
    )

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

    model.load_adapter(str(LORA_PATH), adapter_name="camus_style")
    model.set_adapter("camus_style")

    FastVisionModel.for_inference(model)

    print("  OK\n")
    return model, tokenizer


def generate(model, tokenizer, prompt, system_prompt=None):
    if system_prompt:
        messages = [
            {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
            {"role": "user", "content": [{"type": "text", "text": prompt}]},
        ]
    else:
        messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]

    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        return_tensors="pt",
        add_generation_prompt=True,
    )

    if isinstance(inputs, torch.Tensor):
        input_ids = inputs.to("cuda")
        attention_mask = torch.ones_like(input_ids)
    else:
        inputs = inputs.to("cuda")
        input_ids = inputs["input_ids"]
        attention_mask = inputs.get("attention_mask", torch.ones_like(input_ids))

    streamer = TextStreamer(tokenizer, skip_prompt=True)

    outputs = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=MAX_NEW_TOKENS,
        temperature=TEMPERATURE,
        top_p=TOP_P,
        do_sample=True,
        streamer=streamer,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    generated_text = tokenizer.decode(
        outputs[0][input_ids.shape[1] :], skip_special_tokens=True
    )
    generated_text = generated_text.replace("</s>", "").strip()
    return generated_text


def test_style_completion(model, tokenizer):
    print("\n" + "=" * 70)
    print("TEST 1: Complétion de style Camus")
    print("=" * 70)

    for i, prompt in enumerate(STYLE_PROMPTS, 1):
        print(f"\n--- Prompt {i}/{len(STYLE_PROMPTS)} ---")
        print(f'Début: "{prompt}"')
        print("\nGénération:")
        generate(model, tokenizer, prompt)
        print()


def test_memorization(model, tokenizer):
    print("\n" + "=" * 70)
    print("TEST 2: Anti-mémorisation (passages célèbres)")
    print("=" * 70)
    print("\n⚠️  Si le modèle récite mot pour mot, c'est de l'overfitting\n")

    for book, start_text in MEMORIZATION_TESTS:
        print(f"\n--- {book} ---")
        print(f'Début: "{start_text}"')
        print("\nGénération:")
        generate(model, tokenizer, start_text)
        print()


def test_modern_topics(model, tokenizer):
    print("\n" + "=" * 70)
    print("TEST 3: Style sur sujets modernes")
    print("=" * 70)
    print("\n✓ Le style doit persister même sur des sujets inconnus de Camus\n")

    system_prompt = "Tu es un philosophe existentialiste. Réponds de manière concise et introspective."

    for i, topic in enumerate(MODERN_TOPICS, 1):
        print(f"\n--- Sujet {i}/{len(MODERN_TOPICS)} ---")
        print(f'Question: "{topic}"')
        print("\nRéponse:")
        generate(model, tokenizer, topic, system_prompt)
        print()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--merged",
        action="store_true",
        help="Charger le modèle mergé (ministral-8b-albert-camus/model-style-merged/) au lieu de base + LoRA",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.merged:
        if not MERGED_PATH.exists():
            print(f"ERREUR: Modèle mergé non trouvé: {MERGED_PATH}", file=sys.stderr)
            sys.exit(1)
    else:
        if not LORA_PATH.exists():
            print(f"ERREUR: LoRA non trouvé: {LORA_PATH}", file=sys.stderr)
            sys.exit(1)

    model, tokenizer = load_model(use_merged=args.merged)

    test_style_completion(model, tokenizer)
    test_memorization(model, tokenizer)
    test_modern_topics(model, tokenizer)

    print("\n" + "=" * 70)
    print("Évaluation terminée")
    print("=" * 70)
    print("\nAnalyse manuelle nécessaire:")
    print("1. Style: Vocabulaire (absurde, révolte, lucidité), rythme, ton")
    print("2. Anti-mémorisation: Pas de récitation verbatim des passages célèbres")
    print("3. Généralisation: Style cohérent sur sujets modernes")


if __name__ == "__main__":
    main()
