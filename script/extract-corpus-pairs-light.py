import argparse
import json
import os
import random
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
ARTIFACTS_DIR = PROJECT_ROOT / "ministral-8b-albert-camus"

DEFAULT_OUTPUT_FILE = OUTPUT_DIR / "chat-pairs-light-boost.jsonl"
DEFAULT_META_FILE = OUTPUT_DIR / "chat-pairs-light-boost.meta.jsonl"

MODEL_ID = str(PROJECT_ROOT / "ministral-8b")
LORA_PATH = ARTIFACTS_DIR / "lora-style"
MAX_SEQ_LENGTH = 2048

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_MODEL = "google/gemini-3-flash-preview"

QUESTION_SYSTEM_PROMPT = (
    "Tu écris UNE seule question de conversation en français, basée uniquement sur le passage fourni. "
    "La question doit être naturelle, claire, non académique, et fidèle à l'idée centrale du passage. "
    "Si la question ne peut pas être formulée à partir du passage seul, réponds SKIP. "
    "Réponds uniquement avec la question ou SKIP."
)

SYSTEM_PROMPT = (
    "Écris en français, en prose sobre et lucide. "
    "Pas de listes, pas de titres, pas de markdown. "
    "Pas de citations. Ne mentionne aucun auteur ni aucune œuvre. "
    "1-2 paragraphes courts."
)

QUESTION_GENERATION_PROMPT = (
    "Lis ce passage.\n\n"
    "Tâche: écris UNE question qu'un lecteur réel poserait, "
    "et à laquelle ce passage répond directement.\n\n"
    "Règles:\n"
    "- 8 à 16 mots\n"
    "- formulation naturelle (vouvoiement ou neutre)\n"
    "- une seule idée centrale\n"
    "- pas de noms propres, dates, lieux précis\n"
    "- évite le ton scolaire et le jargon\n"
    "- varie les ouvertures (pas toujours Comment)\n"
    "- privilégie des angles vivables: joie simple, présence, amitié, lumière, création, repos\n"
    "- évite les axes sombres: meurtre, crime, condamnation, exécution, mort, nihilisme\n"
    "- si le passage dépend trop d'un contexte narratif, réponds EXACTEMENT: SKIP\n\n"
    "Sortie: uniquement la question ou SKIP.\n\n"
    "Passage:\n{passage}"
)

ANSWER_GENERATION_PROMPT = (
    "Tu lis un passage de référence (style et idées) et une question.\n\n"
    "Écris une réponse ORIGINALE en français à la question, "
    "cohérente avec les idées du passage, sans reprendre de phrases reconnaissables.\n\n"
    "Règles:\n"
    "- 1 à 2 paragraphes COMPLETS\n"
    "- termine toujours par une ponctuation finale (. ! ?)\n"
    "- pas de listes, pas de titres, pas de markdown\n"
    "- pas de citations\n"
    "- ne parle pas du passage ni de la tâche\n\n"
    "Question:\n{question}\n\n"
    "Passage de référence:\n{passage}"
)

ANSWER_NARRATIVE_PROMPT = (
    "Tu lis un passage narratif et une question.\n\n"
    "Écris une réponse ORIGINALE qui capte l'idée du passage sans reprendre son contexte narratif.\n\n"
    "Règles:\n"
    "- 1 à 2 paragraphes COMPLETS\n"
    "- termine toujours par une ponctuation finale (. ! ?)\n\n"
    "Question:\n{question}\n\n"
    "Passage:\n{passage}"
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
    "correspondance-camus": "correspondance-camus.txt",
}

CORRESPONDANCE_SOURCE = "correspondance-camus"

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

LIGHT_PRIORITY_SOURCES = {
    "l-etranger",
    "noces-et-l-ete",
    "l-envers-et-l-endroit",
    "jonas-et-la-pierre-qui-pousse",
    "la-peste",
    CORRESPONDANCE_SOURCE,
}

MIN_WORDS = 50
MAX_WORDS = 180
MIN_REFLECTIVE_SCORE = 2

MAX_NEW_TOKENS = 600
TEMPERATURE = 0.6
TOP_P = 0.9
REPETITION_PENALTY = 1.08

QUESTION_REJECT_PATTERNS = [
    r"\bmétaphysique\b",
    r"\btranscendance\b",
    r"\bdialectique\b",
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
]

CONTEXT_WEAK_PATTERNS = [
    r"\bcet homme\b",
    r"\bce monsieur\b",
    r"\bvous (vous )?souvenez\b",
    r"\bce soir-là\b",
    r"\bl'autre jour\b",
    r"\bnotre ami\b",
    r"\bn'est-ce pas\s*\?",
]

NARRATIVE_REJECT_PATTERNS = [
    r"\bmeursault\b",
    r"\brieux\b",
    r"\btarrou\b",
    r"\bcéleste\b",
    r"\bjean-baptiste\b",
    r"\bclamence\b",
    r"\bmarengo\b",
    r"\boran\b",
    r"\balger\b",
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
]

REFLECTIVE_WORDS = [
    "joie",
    "bonheur",
    "solitude",
    "silence",
    "cœur",
    "âme",
    "absurde",
    "révolte",
    "liberté",
    "justice",
    "vérité",
    "sens",
    "mort",
    "vie",
    "lucidité",
    "courage",
    "innocence",
    "culpabilité",
    "exil",
    "soleil",
    "mer",
    "lumière",
    "nuit",
    "terre",
    "vent",
    "étoiles",
    "chaleur",
]

LIGHT_KEYWORDS = {
    "joie",
    "bonheur",
    "lumière",
    "soleil",
    "mer",
    "été",
    "amitié",
    "tendresse",
    "promenade",
    "respirer",
    "gratitude",
    "paix",
    "corps",
    "sel",
    "nuit",
    "étoiles",
}

SPIRITUAL_LIGHT_KEYWORDS = {
    "dieu",
    "aumônier",
    "prêtre",
    "péché",
    "foi",
    "âme",
    "indifférence du monde",
    "tendre indifférence",
}

DARK_KEYWORDS = {
    "meurtre",
    "crime",
    "condamnation",
    "exécution",
    "nihilisme",
    "torture",
    "massacre",
}

CORRESPONDENCE_HEADER_PATTERNS = [
    r"^\[.*\]$",
    r"^(lundi|mardi|mercredi|jeudi|vendredi|samedi|dimanche)\b",
    r"^\d+\s*heures?\b",
]

CORRESPONDENCE_LOGISTIC_PATTERNS = [
    r"rendez-vous",
    r"téléphon",
    r"\b\d+\s*h(eures?)?\b",
    r"\bgare\b",
    r"\btrain\b",
    r"\brue\b",
    r"\bboulevard\b",
    r"\badresse\b",
    r"\bNRF\b",
    r"\bGallimard\b",
]

CORRESPONDENCE_INTIMATE_PATTERNS = [
    r"\bmon ch[ée]ri\b",
    r"\bma petite\b",
    r"\bmon amour\b",
    r"\bje t[’']",
    r"\bje t[’']aime\b",
    r"\bje t[’']embrasse\b",
    r"\bnotre amour\b",
    r"\bton visage\b",
    r"\bta voix\b",
]

CORRESPONDENCE_DARK_PATTERNS = [
    r"\bmort\b",
    r"\bmourir\b",
    r"\bmeurtre\b",
    r"\bcrime\b",
    r"\bcondamn",
    r"\bdésespoir\b",
    r"\bangoisse\b",
    r"\bmalheur\b",
]

CORRESPONDENCE_LIGHT_SIGNALS = {
    "soleil",
    "lumière",
    "vent",
    "mer",
    "nuit",
    "étoiles",
    "jardin",
    "arbres",
    "pluie",
    "silence",
    "joie",
    "bonheur",
    "travail",
    "roman",
    "écrire",
    "création",
    "campagne",
    "paix",
}

CORRESPONDENCE_MIN_WORDS = 55
CORRESPONDENCE_MAX_WORDS = 120

REASONING_MODELS = ("openai/gpt-5.2", "openai/o3", "openai/o4-mini")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--sleep", type=float, default=0.25)
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT_FILE))
    parser.add_argument("--meta-output", type=str, default=str(DEFAULT_META_FILE))
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

    with open(config_path, encoding="utf-8") as config_handle:
        saved_config = json.load(config_handle)

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

    if (
        saved_config.get("r") != expected_r
        or saved_config.get("lora_alpha") != expected_alpha
    ):
        print("ERREUR: Mismatch config LoRA", file=sys.stderr)
        sys.exit(1)

    if set(saved_config.get("target_modules", [])) != expected_modules:
        print("ERREUR: Mismatch target_modules LoRA", file=sys.stderr)
        sys.exit(1)

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
        target_modules=sorted(expected_modules),
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
    return model, tokenizer


def normalize_text(text):
    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def split_long_passage(text):
    text = normalize_text(text)
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
        count = len(sentence.split())
        if current_count + count > MAX_WORDS and current_count >= MIN_WORDS:
            chunks.append(" ".join(current).strip())
            current = [sentence]
            current_count = count
            continue
        current.append(sentence)
        current_count += count
    if current_count >= MIN_WORDS:
        chunks.append(" ".join(current).strip())
    return chunks


def reflective_score(text):
    lower = text.lower()
    return sum(1 for word in REFLECTIVE_WORDS if word in lower)


def is_logistic(text):
    lower = text.lower()
    return sum(1 for pattern in LOGISTIC_PATTERNS if re.search(pattern, lower)) >= 2


def is_context_dependent(text):
    lower = text.lower()
    strong = sum(1 for pattern in CONTEXT_STRONG_PATTERNS if re.search(pattern, lower))
    weak = sum(1 for pattern in CONTEXT_WEAK_PATTERNS if re.search(pattern, lower))
    return strong >= 1 or (strong + weak) >= 2


def theme_tag(source, passage):
    lower = passage.lower()
    light_hits = sum(1 for word in LIGHT_KEYWORDS if word in lower)
    spiritual_hits = sum(1 for word in SPIRITUAL_LIGHT_KEYWORDS if word in lower)
    dark_hits = sum(1 for word in DARK_KEYWORDS if word in lower)

    if source == "l-etranger" and spiritual_hits >= 1:
        return "spiritual_luminous"
    if light_hits >= 2 and dark_hits == 0:
        return "light_luminous"
    if source in LIGHT_PRIORITY_SOURCES and light_hits >= 1 and dark_hits <= 1:
        return "light_priority"
    return None


def classify_passage(source, passage):
    lower = passage.lower()
    if is_logistic(passage):
        return "logistic", None
    if is_context_dependent(passage):
        return "context_dependent", None
    if source in NARRATIVE_SOURCES and any(
        re.search(pattern, lower) for pattern in NARRATIVE_REJECT_PATTERNS
    ):
        return "narrative_proper_noun", None
    if reflective_score(passage) < MIN_REFLECTIVE_SCORE:
        return "low_reflective", None
    tag = theme_tag(source, passage)
    if not tag:
        return "not_light_theme", None
    return None, tag


def contains_any_pattern(text, patterns):
    lower = text.lower()
    return any(re.search(pattern, lower, flags=re.IGNORECASE) for pattern in patterns)


def correspondance_light_signal_count(text):
    lower = text.lower()
    return sum(1 for word in CORRESPONDENCE_LIGHT_SIGNALS if word in lower)


def extract_correspondance_strict_chunks(text, stats):
    chunks = []
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

    for paragraph in paragraphs:
        stats["total"] += 1
        paragraph = normalize_text(paragraph)
        if not paragraph:
            stats["empty"] += 1
            continue

        if contains_any_pattern(paragraph, CORRESPONDENCE_HEADER_PATTERNS):
            stats["header"] += 1
            continue

        sentences = [
            normalize_text(s)
            for s in re.split(r"(?<=[.!?])\s+", paragraph)
            if normalize_text(s)
        ]

        current = []
        current_count = 0
        for sentence in sentences:
            sentence_count = len(sentence.split())
            if (
                current_count + sentence_count > CORRESPONDENCE_MAX_WORDS
                and current_count >= CORRESPONDENCE_MIN_WORDS
            ):
                candidate = " ".join(current).strip()
                reason = classify_correspondance_candidate(candidate)
                if reason:
                    stats[reason] += 1
                else:
                    chunks.append(
                        (CORRESPONDANCE_SOURCE, "correspondance_strict", candidate)
                    )
                    stats["kept"] += 1
                current = [sentence]
                current_count = sentence_count
                continue

            current.append(sentence)
            current_count += sentence_count

        if current_count >= CORRESPONDENCE_MIN_WORDS:
            candidate = " ".join(current).strip()
            reason = classify_correspondance_candidate(candidate)
            if reason:
                stats[reason] += 1
            else:
                chunks.append(
                    (CORRESPONDANCE_SOURCE, "correspondance_strict", candidate)
                )
                stats["kept"] += 1
        else:
            stats["too_short"] += 1

    return chunks


def classify_correspondance_candidate(candidate):
    lower = candidate.lower()

    if contains_any_pattern(lower, CORRESPONDENCE_LOGISTIC_PATTERNS):
        return "reject_logistic"

    if contains_any_pattern(lower, CORRESPONDENCE_INTIMATE_PATTERNS):
        return "reject_intimate"

    if contains_any_pattern(lower, CORRESPONDENCE_DARK_PATTERNS):
        return "reject_dark"

    second_person_density = (
        lower.count(" tu ")
        + lower.count(" toi ")
        + lower.count(" te ")
        + lower.count(" ton ")
        + lower.count(" ta ")
        + lower.count(" tes ")
        + lower.count(" t'")
    )
    if second_person_density > 2:
        return "reject_second_person"

    if correspondance_light_signal_count(lower) < 2:
        return "reject_not_light"

    return None


def collect_passages():
    all_passages = []
    all_stats = {}
    for source, filename in CORE_SOURCE_FILES.items():
        path = CORPUS_DIR / filename
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8")

        if source == CORRESPONDANCE_SOURCE:
            stats = defaultdict(int)
            chunks = extract_correspondance_strict_chunks(text, stats)
            all_passages.extend(chunks)
            all_stats[source] = dict(stats)
            continue

        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        stats = defaultdict(int)
        for paragraph in paragraphs:
            stats["total"] += 1
            paragraph = normalize_text(paragraph)
            if not paragraph:
                stats["empty"] += 1
                continue
            for candidate in split_long_passage(paragraph):
                count = len(candidate.split())
                if count < MIN_WORDS:
                    stats["too_short"] += 1
                    continue
                if count > MAX_WORDS:
                    stats["too_long"] += 1
                    continue
                reason, tag = classify_passage(source, candidate)
                if reason:
                    stats[reason] += 1
                    continue
                all_passages.append((source, tag, candidate))
                stats["kept"] += 1
        all_stats[source] = dict(stats)
    return all_passages, all_stats


def is_question_valid(question):
    if not question:
        return False
    question = question.strip()
    if question == "SKIP":
        return False
    if "?" not in question:
        return False
    count = len(question.split())
    if count < 8 or count > 20:
        return False
    lower = question.lower()
    if any(re.search(pattern, lower) for pattern in QUESTION_REJECT_PATTERNS):
        return False
    return True


def validate_answer(answer):
    if not answer:
        return False, "vide"
    answer = answer.strip()
    if len(answer.split()) < 50:
        return False, "trop courte"
    lower = answer.lower()
    for pattern in ANSWER_REJECT_PATTERNS:
        if re.search(pattern, lower, flags=re.MULTILINE):
            return False, f"pattern interdit ({pattern})"
    if answer[-1] not in ".!?…":
        return False, "tronquée (pas de ponctuation finale)"
    return True, "ok"


def generate_question(client, passage, model_name):
    prompt = QUESTION_GENERATION_PROMPT.format(passage=passage)
    params = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": QUESTION_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
    }
    if any(model_name.startswith(prefix) for prefix in REASONING_MODELS):
        params["max_completion_tokens"] = 420
    else:
        params["max_tokens"] = 80
        params["temperature"] = 0.3
    try:
        response = client.chat.completions.create(**params)
    except Exception as exc:
        print(
            f"\n  API EXCEPTION question [{type(exc).__name__}]: {exc}", file=sys.stderr
        )
        return None
    if not response or not response.choices:
        return None
    content = response.choices[0].message.content
    if not content:
        return None
    return content.strip().strip('"').strip("'").strip()


def generate_answer_lora(model, tokenizer, question, passage, source):
    if source in NARRATIVE_SOURCES:
        prompt = ANSWER_NARRATIVE_PROMPT.format(question=question, passage=passage)
    else:
        prompt = ANSWER_GENERATION_PROMPT.format(question=question, passage=passage)

    messages = [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
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

        text = tokenizer.decode(
            outputs[0][input_ids.shape[1] :], skip_special_tokens=True
        )
        text = text.replace("</s>", "").strip().strip('"').strip("'").strip()
        return text
    except Exception as exc:
        print(
            f"\n  LOCAL EXCEPTION answer [{type(exc).__name__}]: {exc}", file=sys.stderr
        )
        return None


def build_pair(question, answer):
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ]
    }


def main():
    args = parse_args()
    print("=== Extraction de paires chat LIGHT ===\n")
    print(f"  Modèle questions (API): {args.model}")
    print(f"  Modèle réponses (local): {LORA_PATH.name}")
    print(f"  Dry run: {args.dry_run}\n")

    passages, stats = collect_passages()
    random.seed(42)
    random.shuffle(passages)

    print("=== Stats segmentation ===\n")
    total_kept = 0
    for source, source_stats in stats.items():
        kept = source_stats.get("kept", 0)
        total_kept += kept
        print(f"  {source}: {source_stats}")
    print(f"\n  Total passages retenus (light): {total_kept}")

    output_file = Path(args.output)
    meta_file = Path(args.meta_output)
    progress_file = output_file.with_suffix(".progress")

    start_index = 0
    if progress_file.exists():
        start_index = int(progress_file.read_text().strip())
        if start_index > 0:
            print(f"  Reprise détectée: passage {start_index}/{len(passages)}")
            passages = passages[start_index:]

    if args.limit:
        passages = passages[: args.limit]

    print(f"  Passages à traiter: {len(passages)}\n")

    if args.dry_run:
        for idx, (source, tag, passage) in enumerate(passages[:15], 1):
            print(f"  [{idx}] {source} [{tag}] ({len(passage.split())} mots)")
            print(f"      {passage[:220]}...")
        print("\nDone (dry-run).")
        return

    client = create_client()
    model, tokenizer = load_lora_model()

    output_file.parent.mkdir(parents=True, exist_ok=True)
    meta_file.parent.mkdir(parents=True, exist_ok=True)

    output_handle = open(output_file, "a", encoding="utf-8")
    meta_handle = open(meta_file, "a", encoding="utf-8")

    pairs_count = 0
    invalid_questions = 0
    invalid_answers = 0
    skipped_by_model = 0
    api_errors = 0
    lora_errors = 0

    for i, (source, tag, passage) in enumerate(passages):
        global_index = start_index + i
        question = None
        answer = None
        print(f"  [{i + 1}/{len(passages)}] {source} [{tag}]...", end=" ", flush=True)

        question = generate_question(client, passage, args.model)
        if not question:
            api_errors += 1
            status = "api_error_question"
            print("ERREUR API (question)")
        elif question == "SKIP":
            skipped_by_model += 1
            status = "skip_question_model"
            print("SKIP (modèle question)")
        elif not is_question_valid(question):
            invalid_questions += 1
            status = "invalid_question"
            print(f"SKIP (question invalide: {question})")
        else:
            answer = generate_answer_lora(model, tokenizer, question, passage, source)
            if not answer:
                lora_errors += 1
                status = "local_error_answer"
                print("ERREUR LOCAL (answer)")
            else:
                valid_answer, reason = validate_answer(answer)
                if not valid_answer:
                    invalid_answers += 1
                    status = f"invalid_answer:{reason}"
                    print(f"SKIP (réponse invalide: {reason})")
                else:
                    pair = build_pair(question, answer)
                    json.dump(pair, output_handle, ensure_ascii=False)
                    output_handle.write("\n")
                    output_handle.flush()
                    pairs_count += 1
                    status = "ok"
                    print("OK")
                    print(f"    Q: {question}")
                    print(f"    A: {answer[:170]}...")

        meta = {
            "index": global_index,
            "source": source,
            "theme_tag": tag,
            "status": status,
            "question_model": args.model,
            "answer_model": "local_lora",
            "question": question,
            "answer_word_count": len(answer.split()) if answer else 0,
        }
        json.dump(meta, meta_handle, ensure_ascii=False)
        meta_handle.write("\n")
        meta_handle.flush()

        progress_file.write_text(str(global_index + 1))
        time.sleep(args.sleep)

    output_handle.close()
    meta_handle.close()
    if progress_file.exists():
        progress_file.unlink()

    print("\n=== Résumé ===\n")
    print(f"  Paires générées: {pairs_count}")
    print(f"  SKIP modèle question: {skipped_by_model}")
    print(f"  Questions invalides: {invalid_questions}")
    print(f"  Réponses invalides: {invalid_answers}")
    print(f"  Erreurs API question: {api_errors}")
    print(f"  Erreurs locales réponse: {lora_errors}")
    print(f"  Fichier dataset: {output_file}")
    print(f"  Fichier meta: {meta_file}")
    print("Done.")


if __name__ == "__main__":
    main()
