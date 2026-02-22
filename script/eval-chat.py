import sys
from pathlib import Path

import torch
from transformers import AutoProcessor, Mistral3ForConditionalGeneration

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = PROJECT_ROOT / "ministral-8b-albert-camus"
FINAL_PATH = ARTIFACTS_DIR / "model-final"

MAX_NEW_TOKENS = 300
TEMPERATURE = 0.7
TOP_P = 0.9

PHILOSOPHY_PROMPTS = [
    "Qu'est-ce que la liberté ?",
    "La mort donne-t-elle un sens à la vie ?",
    "Comment vivre dans un monde absurde ?",
    "La souffrance a-t-elle une valeur ?",
]

MODERN_PROMPTS = [
    "Que penses-tu de l'intelligence artificielle ?",
    "Les réseaux sociaux rapprochent-ils les hommes ?",
    "La démocratie est-elle en danger ?",
    "Le changement climatique peut-il être une révolte ?",
]

PERSONAL_PROMPTS = [
    "Comment aimer sans posséder ?",
    "Qu'est-ce que la solitude ?",
    "As-tu peur de la mort ?",
    "Que fais-tu quand tu doutes de tout ?",
]

ANTI_ASSISTANT_PROMPTS = [
    "Donne-moi 5 conseils pour être heureux.",
    "Explique-moi la philosophie de l'absurde.",
    "Fais une liste des thèmes de ton œuvre.",
    "Résume ta pensée en 3 points.",
]

CONCRETE_PROMPTS = [
    "Ton vol est annulé, que penses-tu ?",
    "Tu scrolles ton téléphone à 3h du matin, que ressens-tu ?",
    "Un ami te demande pourquoi tu n'es pas sur Instagram.",
    "Tu regardes les infos et tout semble s'effondrer.",
]


def load_model():
    print("=== Chargement du modèle final (camus-final) ===\n")

    print(f"  Modèle: {FINAL_PATH}")

    model = Mistral3ForConditionalGeneration.from_pretrained(
        str(FINAL_PATH),
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(str(FINAL_PATH))

    print("  OK\n")
    return model, processor.tokenizer


def generate(model, tokenizer, prompt):
    messages = [
        {"role": "user", "content": [{"type": "text", "text": prompt}]},
    ]

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

    with torch.inference_mode():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    text = tokenizer.decode(outputs[0][input_ids.shape[1] :], skip_special_tokens=True)
    return text.replace("</s>", "").strip()


def run_test(model, tokenizer, title, prompts):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)

    for i, prompt in enumerate(prompts, 1):
        print(f"\n--- {i}/{len(prompts)} ---")
        print(f'Q: "{prompt}"\n')
        response = generate(model, tokenizer, prompt)
        print(f"R: {response}")
        print()


def main():
    if not FINAL_PATH.exists():
        print(
            f"ERREUR: Modèle final non trouvé: {FINAL_PATH}\n"
            f"  Lancer d'abord: python script/merge-phase2.py",
            file=sys.stderr,
        )
        sys.exit(1)

    model, tokenizer = load_model()

    run_test(model, tokenizer, "TEST 1: Philosophie existentielle", PHILOSOPHY_PROMPTS)
    run_test(model, tokenizer, "TEST 2: Sujets modernes", MODERN_PROMPTS)
    run_test(model, tokenizer, "TEST 3: Personnel / intime", PERSONAL_PROMPTS)
    run_test(
        model,
        tokenizer,
        "TEST 4: Anti-mode-assistant (prompts pièges)",
        ANTI_ASSISTANT_PROMPTS,
    )
    run_test(
        model, tokenizer, "TEST 5: Situations concrètes / modernes", CONCRETE_PROMPTS
    )

    print("\n" + "=" * 70)
    print("Évaluation terminée")
    print("=" * 70)
    print("\nCritères d'évaluation manuelle:")
    print("1. Prose directe (pas de listes, markdown, titres)")
    print('2. Pas de citations explicites ("Dans Le Mythe de Sisyphe...")')
    print('3. Pas de ton assistant ("Voici", "Certainement", "Je vais vous expliquer")')
    print("4. Images concrètes (soleil, mer, pierre, lumière)")
    print("5. Ton introspectif, personnel — Camus parle, il n'explique pas Camus")


if __name__ == "__main__":
    main()
