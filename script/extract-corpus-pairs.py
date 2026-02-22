import argparse
import json
import os
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch
from dotenv import load_dotenv
from openai import OpenAI
from unsloth import FastVisionModel

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CORPUS_DIR = PROJECT_ROOT / "data" / "corpus"
OUTPUT_DIR = PROJECT_ROOT / "data" / "dataset"
DEFAULT_OUTPUT_FILE = OUTPUT_DIR / "chat-pairs-corpus.jsonl"
ARTIFACTS_DIR = PROJECT_ROOT / "ministral-8b-albert-camus"

MODEL_ID = str(PROJECT_ROOT / "ministral-8b")
LORA_PATH = ARTIFACTS_DIR / "lora-style"
MAX_SEQ_LENGTH = 2048

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_MODEL = "anthropic/claude-opus-4.6"

QUESTION_SYSTEM_PROMPT = (
    "Tu génères une question unique, précise et naturelle en français. "
    "Pas de commentaire, pas d'explication."
)

QUESTION_GENERATION_PROMPT = (
    "Lis ce passage.\n\n"
    "Génère UNE question en français qu'un lecteur pourrait poser "
    "dans une conversation sérieuse, et à laquelle ce passage "
    "constitue une réponse pertinente et autonome.\n\n"
    "Règles :\n"
    "- Vouvoiement ou formulation neutre (pas de tutoiement)\n"
    "- 8 à 20 mots\n"
    "- Varie les formulations d'ouverture; n'utilise pas toujours 'Comment'\n"
    "- Privilégie en alternance: 'Pourquoi', 'En quoi', 'Dans quelle mesure', 'Que révèle', 'Faut-il', 'Peut-on'\n"
    "- N'utilise 'Comment' que si c'est clairement la meilleure formulation\n"
    "- La question doit coller à l'idée centrale du passage\n"
    "- Pas de références au passage, pas de noms propres, pas de dates\n"
    "- Pas de logistique (horaires, lieux, rendez-vous)\n"
    "- Pas de relation amoureuse explicite\n"
    "- Préférer des questions abstraites mais simples "
    "(vie, justice, culpabilité, liberté, bonheur, solitude, mort)\n\n"
    "Règle qualité : la question doit être difficile à recycler "
    "sur un autre passage. Si le passage n'est pas assez autonome "
    "ou généralisable, réponds EXACTEMENT : SKIP\n\n"
    "Réponds UNIQUEMENT avec la question ou SKIP.\n\n"
    "Passage :\n{passage}"
)

ANSWER_GENERATION_PROMPT = (
    "Tu lis un passage de référence (style et idées) et une question.\n\n"
    "Écris une réponse ORIGINALE en français à la question, "
    "cohérente avec les idées du passage, mais sans reprendre de phrases "
    "ni de segments reconnaissables.\n\n"
    "Règles :\n"
    "- 1 à 2 paragraphes COMPLETS avec une conclusion claire\n"
    "- Termine toujours par une ponctuation finale (. ! ?)\n"
    "- Pas de listes, pas de titres, pas de markdown\n"
    "- Pas de citations, pas de guillemets\n"
    "- Ne parle pas du passage ou du fait que tu réponds à partir d'un texte\n\n"
    "Question :\n{question}\n\n"
    "Passage de référence :\n{passage}"
)

ANSWER_NARRATIVE_PROMPT = (
    "Tu lis un passage tiré d'une œuvre narrative (réflexion ou description). "
    "Une question en a été déduite.\n\n"
    "Écris une réponse ORIGINALE à cette question, qui CAPTE L'IDÉE PHILOSOPHIQUE "
    "du passage mais SANS reprendre son contexte narratif.\n\n"
    "RÈGLES :\n"
    "- 1 à 2 paragraphes COMPLETS avec une conclusion claire\n"
    "- Termine toujours par une ponctuation finale (. ! ?)\n"
    "- Aucun nom propre (personnages, lieux: Meursault, Oran, Céleste...)\n"
    "- Aucun repère temporel/spatial spécifique ('ce jour-là', 'dans cette rue')\n"
    "- Pas de mention d'événements de l'intrigue\n\n"
    "Question : {question}\n\n"
    "Passage (pour l'idée seulement) : {passage}"
)

CORE_SOURCE_FILES = {
    "l-homme-revolte": "l-homme-revolte.txt",
    "le-mythe-de-sisyphe": "le-mythe-de-sisyphe.txt",
    "discours-de-suede": "discours-de-suede.txt",
    "noces-et-l-ete": "noces-et-l-ete.txt",
    "l-envers-et-l-endroit": "l-envers-et-l-endroit.txt",
    "la-chute": "la-chute.txt",
    "l-etranger": "l-etranger.txt",
    "la-peste": "la-peste.txt",
    "jonas-et-la-pierre-qui-pousse": "jonas-et-la-pierre-qui-pousse.txt",
}

REFERENCE_SOURCES = {
    "l-homme-revolte",
    "le-mythe-de-sisyphe",
    "discours-de-suede",
    "noces-et-l-ete",
    "l-envers-et-l-endroit",
}

NARRATIVE_SOURCES = {
    "la-chute",
    "l-etranger",
    "la-peste",
    "jonas-et-la-pierre-qui-pousse",
}

MIN_WORDS = 50
MAX_WORDS = 180
MIN_REFLECTIVE_SCORE = 2

MAX_NEW_TOKENS = 600
TEMPERATURE = 0.6
TOP_P = 0.9
REPETITION_PENALTY = 1.08

QUESTION_REJECT_PATTERNS = [
    r"tu penses à moi",
    r"tu m'aimes",
    r"tu me manques",
    r"on se voit",
    r"on se retrouve",
    r"tu reviens quand",
]

ANSWER_REJECT_PATTERNS = [
    r"\b(cam(us)?|sartre|l[’']etranger|la peste|la chute|sisyphe)\b",
    r"^[-*]\s+",
    r"^#+\s+",
    r"\b(premièrement|deuxièmement|troisièmement)\b",
]

CONTEXT_STRONG_PATTERNS = [
    r"\bnotre hôte\b",
    r"\bcomme je (vous )?disais\b",
    r"\bje vous ai (dit|parlé)\b",
    r"\bje vous l'ai\b",
    r"\bje vous racontais\b",
    r"\bje vous le disais\b",
    r"\bla femme dont\b",
]

CONTEXT_WEAK_PATTERNS = [
    r"\bcet homme\b",
    r"\bce monsieur\b",
    r"\bvous (vous )?souvenez\b",
    r"\bce soir-là\b",
    r"\bl'autre jour\b",
    r"\bnotre ami\b",
    r"\bvous savez bien\b",
    r"\bcomme vous le voyez\b",
    r"\bn'est-ce pas\s*\?",
    r"\bvoyez-vous\b",
]

NARRATIVE_REJECT_PATTERNS = [
    r"\bMeursault\b",
    r"\bRieux\b",
    r"\bTarrou\b",
    r"\bCottard\b",
    r"\bGrand\b",
    r"\bRambert\b",
    r"\bmaman\b",
    r"\bCéleste\b",
    r"\bEmmanuel\b",
    r"\bSalamano\b",
    r"\bRaymond\b",
    r"\bMarie\b",
    r"\bJean-Baptiste\b",
    r"\bClamence\b",
    r"\bJonas\b",
    r"\bRateau\b",
    r"\bLouise\b",
    r"\bcher ami\b",
    r"\bmon cher\b",
    r"\bje vous en prie\b",
    r"\bpermettez-moi\b",
    r"\bMarengo\b",
    r"\bAlger\b",
    r"\bOran\b",
]

LOGISTIC_PATTERNS = [
    r"rendez-vous",
    r"\b\d+\s*h(eures?)?\b",
    r"\bNRF\b",
    r"\bGallimard\b",
    r"téléphon",
    r"coup de fil",
    r"\bgare\b",
    r"\btrain\b",
    r"\bavion\b",
    r"coin de la rue",
    r"\bboulevard\b",
    r"\brue du\b",
    r"\brue de\b",
    r"\bhôtel\b",
    r"\badresse\b",
    r"\btaxi\b",
    r"\bbillet\b",
    r"\bvalise\b",
    r"\bpasseport\b",
]

REFLECTIVE_WORDS = [
    "angoisse",
    "souffrir",
    "souffrance",
    "douleur",
    "joie",
    "bonheur",
    "peur",
    "espoir",
    "désespoir",
    "solitude",
    "silence",
    "vide",
    "tristesse",
    "larmes",
    "cœur",
    "âme",
    "tourment",
    "fatigue",
    "amertume",
    "détresse",
    "honte",
    "fierté",
    "absurde",
    "révolte",
    "liberté",
    "justice",
    "vérité",
    "sens",
    "mort",
    "vie",
    "destin",
    "condition",
    "lucidité",
    "courage",
    "innocence",
    "culpabilité",
    "jugement",
    "exil",
    "penser",
    "comprendre",
    "savoir",
    "croire",
    "sentir",
    "aimer",
    "vivre",
    "mourir",
    "écrire",
    "créer",
    "lutter",
    "choisir",
    "rêver",
    "douter",
    "espérer",
    "renoncer",
    "soleil",
    "mer",
    "lumière",
    "nuit",
    "ciel",
    "terre",
    "vent",
    "montagne",
    "étoiles",
    "chaleur",
    "ombre",
    "horizon",
]

NATURE_WORDS = [
    "soleil",
    "mer",
    "lumière",
    "nuit",
    "ciel",
    "terre",
    "vent",
    "montagne",
    "étoiles",
    "chaleur",
    "ombre",
    "horizon",
    "pluie",
    "neige",
    "eau",
    "rivage",
    "sable",
    "aube",
    "crépuscule",
]

EMOTION_WORDS = [
    "angoisse",
    "souffrir",
    "souffrance",
    "douleur",
    "joie",
    "bonheur",
    "peur",
    "espoir",
    "désespoir",
    "solitude",
    "silence",
    "vide",
    "tristesse",
    "larmes",
    "cœur",
    "âme",
    "tourment",
    "amour",
    "haine",
    "tendresse",
    "mélancolie",
    "nostalgie",
]

REASONING_MODELS = ("openai/gpt-5.2", "openai/o3", "openai/o4-mini")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--sleep", type=float, default=0.25)
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT_FILE))
    return parser.parse_args()


def create_client():
    if not OPENROUTER_API_KEY:
        print("ERREUR: OPENROUTER_API_KEY manquante dans .env", file=sys.stderr)
        sys.exit(1)
    return OpenAI(base_url=OPENROUTER_BASE_URL, api_key=OPENROUTER_API_KEY)


def load_lora_model():
    if not LORA_PATH.exists():
        print(f"ERREUR: LoRA non trouvé: {LORA_PATH}", file=sys.stderr)
        sys.exit(1)

    config_path = LORA_PATH / "adapter_config.json"
    if not config_path.exists():
        print("ERREUR: adapter_config.json introuvable", file=sys.stderr)
        sys.exit(1)

    with open(config_path) as f:
        saved_config = json.load(f)

    expected_r = 32
    expected_alpha = 64
    expected_modules = {
        "down_proj",
        "up_proj",
        "q_proj",
        "o_proj",
        "k_proj",
        "v_proj",
        "gate_proj",
    }

    saved_r = saved_config.get("r")
    saved_alpha = saved_config.get("lora_alpha")
    saved_modules = set(saved_config.get("target_modules", []))

    if saved_r != expected_r or saved_alpha != expected_alpha:
        print(
            f"ERREUR: Mismatch config LoRA (r={saved_r}, alpha={saved_alpha}) "
            f"vs attendu (r={expected_r}, alpha={expected_alpha})",
            file=sys.stderr,
        )
        sys.exit(1)

    if saved_modules != expected_modules:
        print(
            f"ERREUR: Mismatch target_modules\n"
            f"  Sauvegardé: {sorted(saved_modules)}\n"
            f"  Attendu:    {sorted(expected_modules)}",
            file=sys.stderr,
        )
        sys.exit(1)

    print("Chargement modèle local + LoRA...")
    model, tokenizer = FastVisionModel.from_pretrained(
        model_name=MODEL_ID,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=None,
        load_in_4bit=True,
    )

    model = FastVisionModel.get_peft_model(
        model,
        r=expected_r,
        lora_alpha=expected_alpha,
        target_modules=list(expected_modules),
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
    print("  OK (config validée)")
    return model, tokenizer


def normalize_passage_text(text):
    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def split_long_passage(text):
    text = normalize_passage_text(text)
    words = text.split()
    if len(words) <= MAX_WORDS:
        return [text]

    sentences = re.split(r"(?<=[.!?])\s+(?=[A-ZÀ-Ü])", text)
    if len(sentences) <= 1:
        return [text] if len(words) >= MIN_WORDS else []

    chunks = []
    current = []
    current_count = 0

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        sentence_words = len(sentence.split())

        if current_count + sentence_words > MAX_WORDS and current_count >= MIN_WORDS:
            chunks.append(" ".join(current).strip())
            current = [sentence]
            current_count = sentence_words
            continue

        current.append(sentence)
        current_count += sentence_words

    if current_count >= MIN_WORDS:
        chunks.append(" ".join(current).strip())

    return chunks


def is_logistic(text):
    lower = text.lower()
    matches = sum(1 for p in LOGISTIC_PATTERNS if re.search(p, lower))
    return matches >= 2


def is_context_dependent(text):
    lower = text.lower()
    strong_matches = sum(1 for p in CONTEXT_STRONG_PATTERNS if re.search(p, lower))
    weak_matches = sum(1 for p in CONTEXT_WEAK_PATTERNS if re.search(p, lower))
    total_matches = strong_matches + weak_matches
    return strong_matches >= 1 or (total_matches >= 2)


def reflective_score(text):
    lower = text.lower()
    return sum(1 for w in REFLECTIVE_WORDS if w in lower)


def has_lyric_combo(text):
    lower = text.lower()
    has_nature = any(w in lower for w in NATURE_WORDS)
    has_emotion = any(w in lower for w in EMOTION_WORDS)
    return has_nature and has_emotion


def classify_passage(passage, source):
    if is_logistic(passage):
        return "logistic"

    if is_context_dependent(passage):
        return "context_dependent"

    if source in NARRATIVE_SOURCES:
        lower = passage.lower()
        if any(re.search(p, lower) for p in NARRATIVE_REJECT_PATTERNS):
            return "narrative_proper_noun"

    score = reflective_score(passage)
    if score < MIN_REFLECTIVE_SCORE and not has_lyric_combo(passage):
        return "low_reflective"

    return None


def is_question_valid(question):
    if not question:
        return False
    q = question.strip()
    if q == "SKIP":
        return False
    if "?" not in q:
        return False
    word_count = len(q.split())
    if word_count < 8 or word_count > 20:
        return False
    lower = q.lower()
    if any(re.search(p, lower) for p in QUESTION_REJECT_PATTERNS):
        return False
    return True


def tokenize_words(text):
    return re.findall(r"[a-zà-ÿ]+", text.lower())


def has_excessive_overlap(source, answer):
    src_tokens = tokenize_words(source)
    ans_tokens = tokenize_words(answer)
    if len(src_tokens) < 6 or len(ans_tokens) < 6:
        return False

    src_trigrams = {tuple(src_tokens[i : i + 3]) for i in range(len(src_tokens) - 2)}
    ans_trigrams = {tuple(ans_tokens[i : i + 3]) for i in range(len(ans_tokens) - 2)}
    if not ans_trigrams:
        return False

    overlap_ratio = len(src_trigrams & ans_trigrams) / len(ans_trigrams)
    if overlap_ratio > 0.30:
        return True

    normalized_source = normalize_passage_text(source).lower()
    normalized_answer = normalize_passage_text(answer).lower()
    source_segments = [
        segment.strip() for segment in re.split(r"(?<=[.!?])\s+", normalized_source)
    ]
    return any(
        len(segment) >= 90 and segment in normalized_answer
        for segment in source_segments
    )


def validate_answer(answer):
    if not answer:
        return False, "vide"

    a = answer.strip()
    words = a.split()
    if len(words) < 50:
        return False, f"trop courte ({len(words)} mots)"

    lower = a.lower()
    for pattern in ANSWER_REJECT_PATTERNS:
        if re.search(pattern, lower, flags=re.MULTILINE):
            return False, f"pattern interdit ({pattern})"

    if a[-1] not in ".!?…":
        return False, "tronquée (pas de ponctuation finale)"

    return True, "ok"


def generate_question(client, passage, model_name):
    prompt = QUESTION_GENERATION_PROMPT.format(passage=passage)
    is_reasoning = any(model_name.startswith(prefix) for prefix in REASONING_MODELS)

    params = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": QUESTION_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
    }
    if is_reasoning:
        params["max_completion_tokens"] = 420
    else:
        params["max_tokens"] = 80
        params["temperature"] = 0.3

    try:
        response = client.chat.completions.create(**params)
    except Exception as e:
        error_type = type(e).__name__
        print(f"\n  API EXCEPTION question [{error_type}]: {e}", file=sys.stderr)

        status_code = getattr(e, "status_code", None)
        if status_code is not None:
            print(f"  HTTP status: {status_code}", file=sys.stderr)

        response = getattr(e, "response", None)
        if response is not None:
            try:
                print(f"  Response body: {response.text[:500]}", file=sys.stderr)
            except Exception:
                print(f"  Response repr: {repr(response)[:500]}", file=sys.stderr)

        error_body = getattr(e, "body", None)
        if error_body is not None:
            print(f"  Error body: {error_body}", file=sys.stderr)
        sys.stderr.flush()
        return None

    if not response or not response.choices:
        print(
            f"\n  API EMPTY RESPONSE question: {repr(response)[:500]}", file=sys.stderr
        )
        sys.stderr.flush()
        return None

    content = response.choices[0].message.content
    if not content:
        finish = response.choices[0].finish_reason
        print(
            f"\n  API NULL CONTENT question (finish_reason={finish}): {repr(response.choices[0])[:300]}",
            file=sys.stderr,
        )
        sys.stderr.flush()
        return None

    return content.strip().strip('"').strip("'").strip()


def generate_answer_lora(model, tokenizer, question, passage, source):
    if source in NARRATIVE_SOURCES:
        prompt = ANSWER_NARRATIVE_PROMPT.format(question=question, passage=passage)
    else:
        prompt = ANSWER_GENERATION_PROMPT.format(question=question, passage=passage)
    messages = [
        {"role": "user", "content": [{"type": "text", "text": prompt}]},
    ]

    try:
        inputs = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            return_tensors="pt",
            add_generation_prompt=True,
        ).to("cuda")

        if isinstance(inputs, dict):
            input_ids = inputs["input_ids"]
            attention_mask = inputs.get("attention_mask", torch.ones_like(input_ids))
        else:
            input_ids = inputs
            attention_mask = torch.ones_like(input_ids)

        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            repetition_penalty=REPETITION_PENALTY,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

        generated_text = tokenizer.decode(
            outputs[0][input_ids.shape[1] :], skip_special_tokens=True
        )
        generated_text = generated_text.replace("</s>", "").strip()
        generated_text = generated_text.strip('"').strip("'").strip()
        return generated_text
    except Exception as e:
        print(f"\n  LOCAL EXCEPTION answer [{type(e).__name__}]: {e}", file=sys.stderr)
        sys.stderr.flush()
        return None


def build_pair(question, answer):
    return {
        "messages": [
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ]
    }


def segment_generic(source, filepath):
    text = filepath.read_text(encoding="utf-8")
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

    passages = []
    stats = defaultdict(int)

    for paragraph in paragraphs:
        stats["total"] += 1
        paragraph = normalize_passage_text(paragraph)
        if not paragraph:
            stats["empty"] += 1
            continue

        for candidate in split_long_passage(paragraph):
            word_count = len(candidate.split())
            if word_count < MIN_WORDS:
                stats["too_short"] += 1
                continue
            if word_count > MAX_WORDS:
                stats["too_long"] += 1
                continue

            reason = classify_passage(candidate, source)
            if reason:
                stats[reason] += 1
                continue

            passages.append((source, candidate))
            stats["kept"] += 1

    return passages, dict(stats)


def collect_passages():
    all_passages = []
    all_stats = {}

    for source, filename in CORE_SOURCE_FILES.items():
        path = CORPUS_DIR / filename
        if not path.exists():
            print(f"  SKIP: {path} introuvable")
            continue

        passages, stats = segment_generic(source, path)
        all_passages.extend(passages)
        all_stats[source] = stats

    return all_passages, all_stats


def main():
    args = parse_args()

    print("=== Extraction de paires chat depuis le corpus ===\n")
    print(f"  Modèle questions (API): {args.model}")
    print(f"  Modèle réponses (local): {LORA_PATH.name}")
    print(f"  Dry run: {args.dry_run}")
    print()

    passages, stats = collect_passages()

    import random

    random.seed(42)
    random.shuffle(passages)

    print("=== Stats segmentation ===\n")
    total_kept = 0
    for source, source_stats in stats.items():
        kept = source_stats.get("kept", 0)
        total_kept += kept
        print(f"  {source}: {source_stats}")
    print(f"\n  Total passages retenus: {total_kept}")

    output_file = Path(args.output)
    progress_file = output_file.with_suffix(".progress")

    start_index = 0
    if progress_file.exists():
        start_index = int(progress_file.read_text().strip())
        if start_index > 0:
            print(f"  Reprise détectée: passage {start_index}/{total_kept}")
            passages = passages[start_index:]

    if args.limit:
        passages = passages[: args.limit]

    print(f"  Passages à traiter: {len(passages)}\n")

    if args.dry_run:
        print("=== Échantillon (dry-run) ===\n")
        for idx, (source, passage) in enumerate(passages[:15]):
            print(f"  [{idx + 1}] {source} ({len(passage.split())} mots)")
            print(f"      {passage[:220]}...")
            print()
        print("Done (dry-run).")
        return

    client = create_client()
    model, tokenizer = load_lora_model()

    output_file.parent.mkdir(parents=True, exist_ok=True)

    skipped_by_model = 0
    invalid_questions = 0
    invalid_answers = 0
    copy_rejects = 0
    api_errors = 0
    lora_errors = 0
    pairs_count = 0

    output_handle = open(output_file, "a", encoding="utf-8")

    for i, (source, passage) in enumerate(passages):
        print(f"  [{i + 1}/{len(passages)}] {source}...", end=" ", flush=True)

        question = generate_question(client, passage, args.model)
        if not question:
            api_errors += 1
            print("ERREUR API (question)")
            continue

        if question.strip() == "SKIP":
            skipped_by_model += 1
            print("SKIP (modèle question)")
            continue

        if not is_question_valid(question):
            invalid_questions += 1
            print(f"SKIP (question invalide: {question})")
            continue

        answer = generate_answer_lora(model, tokenizer, question, passage, source)
        if not answer:
            lora_errors += 1
            print("ERREUR LOCAL (answer)")
            continue

        if has_excessive_overlap(passage, answer):
            copy_rejects += 1
            print("SKIP (copie trop proche du passage)")
            continue

        is_valid_answer, invalid_reason = validate_answer(answer)
        if not is_valid_answer:
            invalid_answers += 1
            print(f"SKIP (réponse invalide: {invalid_reason})")
            print(f"    A(rejetée): {answer[:170]}...")
            continue

        pair = build_pair(question, answer)
        json.dump(pair, output_handle, ensure_ascii=False)
        output_handle.write("\n")
        output_handle.flush()
        pairs_count += 1

        print("OK")
        print(f"    Q: {question}")
        print(f"    A: {answer[:170]}...")

        if (i + 1) % 50 == 0:
            print(
                f"\n  --- Progression: {i + 1}/{len(passages)} "
                f"(paires: {pairs_count}, skip_modele: {skipped_by_model}, "
                f"q_invalides: {invalid_questions}, a_invalides: {invalid_answers}, "
                f"copies: {copy_rejects}) ---\n"
            )

        # Sauvegarder l'index de progression
        progress_file.write_text(str(start_index + i + 1))

        time.sleep(args.sleep)

    output_handle.close()

    # Supprimer le fichier de progression à la fin
    if progress_file.exists():
        progress_file.unlink()

    total_passages = len(passages) + start_index
    print("\n=== Résumé ===\n")
    print(f"  Passages traités (total): {total_passages}")
    print(f"  Paires générées (cette session): {pairs_count}")
    print(f"  SKIP par modèle question: {skipped_by_model}")
    print(f"  Questions invalides: {invalid_questions}")
    print(f"  Réponses invalides: {invalid_answers}")
    print(f"  Rejets copie passage: {copy_rejects}")
    print(f"  Erreurs API question: {api_errors}")
    print(f"  Erreurs locales réponse: {lora_errors}")
    print(
        f"  Taux de rétention (cette session): {100 * pairs_count / max(len(passages), 1):.0f}%"
    )
    print(f"  Fichier: {output_file}")

    if pairs_count > 0:
        print("\n=== Échantillon (dernières paires générées) ===\n")
        print(f"  Session terminée avec {pairs_count} nouvelles paires.")

    print("Done.")


if __name__ == "__main__":
    main()
