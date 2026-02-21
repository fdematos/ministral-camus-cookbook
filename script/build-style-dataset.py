import json
import random
import statistics
from pathlib import Path

from tokenizers import Tokenizer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CORPUS_DIR = PROJECT_ROOT / "data" / "corpus"
DATASET_DIR = PROJECT_ROOT / "data" / "dataset"
OUTPUT_FILE = DATASET_DIR / "style-train.jsonl"

TOKENIZER_PATH = PROJECT_ROOT / "ministral-8b" / "tokenizer.json"
TARGET_CHUNK_TOKENS = 2048
OVERLAP_TOKENS = 128
MIN_CHUNK_TOKENS = 256
SEED = 42


def load_tokenizer():
    return Tokenizer.from_file(str(TOKENIZER_PATH))


def chunk_by_tokens(text, tokenizer):
    encoded = tokenizer.encode(text)
    token_ids = encoded.ids
    total_tokens = len(token_ids)

    if total_tokens <= TARGET_CHUNK_TOKENS:
        return [text], [total_tokens]

    chunks = []
    chunk_sizes = []
    stride = TARGET_CHUNK_TOKENS - OVERLAP_TOKENS
    start = 0

    while start < total_tokens:
        end = min(start + TARGET_CHUNK_TOKENS, total_tokens)
        chunk_ids = token_ids[start:end]
        chunk_text = tokenizer.decode(chunk_ids)
        chunks.append(chunk_text)
        chunk_sizes.append(len(chunk_ids))

        if end >= total_tokens:
            break
        start += stride

    return chunks, chunk_sizes


def process_file(filepath, tokenizer):
    with open(filepath, "r", encoding="utf-8") as f:
        text = f.read()

    print("    tokenization...", end=" ")
    chunks, sizes = chunk_by_tokens(text, tokenizer)
    total = len(tokenizer.encode(text).ids)
    print(f"OK ({total:,} tokens)")

    return chunks, sizes


def main():
    print("=== Construction du dataset style ===\n")

    print(f"Chargement du tokenizer ({TOKENIZER_PATH.parent.name})...")
    tokenizer = load_tokenizer()
    print("  OK\n")

    all_chunks = []
    all_sizes = []

    for filepath in sorted(CORPUS_DIR.glob("*.txt")):
        print(f"  {filepath.name}")
        chunks, sizes = process_file(filepath, tokenizer)
        all_chunks.extend(chunks)
        all_sizes.extend(sizes)
        print(f"    → {len(chunks)} chunks")

    print(f"\n  Total avant filtre: {len(all_chunks)} chunks")

    combined = list(zip(all_chunks, all_sizes))
    random.seed(SEED)
    random.shuffle(combined)

    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    kept_sizes = []
    filtered_count = 0

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for chunk, size in combined:
            if size < MIN_CHUNK_TOKENS:
                filtered_count += 1
                continue
            chunk = chunk.rstrip()
            json.dump({"text": chunk}, f, ensure_ascii=False)
            f.write("\n")
            kept_sizes.append(size)

    kept_sizes.sort()
    total_tokens = sum(kept_sizes)
    p50 = kept_sizes[len(kept_sizes) // 2]
    p95 = kept_sizes[int(len(kept_sizes) * 0.95)]

    print(f"\n=== Résumé ===\n")
    print(f"  Fichier: {OUTPUT_FILE}")
    print(
        f"  Chunks conservés: {len(kept_sizes)} (filtrés: {filtered_count} < {MIN_CHUNK_TOKENS} tokens)"
    )
    print(f"  Tokens total: {total_tokens:,}")
    print(
        f"  Tokens/chunk: min={kept_sizes[0]}, p50={p50}, p95={p95}, max={kept_sizes[-1]}"
    )
    print(f"  EOS: géré par le tokenizer lors du training")

    print(f"\n=== Contrôle qualité (3 chunks aléatoires) ===\n")
    with open(OUTPUT_FILE) as f:
        lines = f.readlines()
    sample_indices = [0, len(lines) // 2, len(lines) - 1]
    for idx in sample_indices:
        data = json.loads(lines[idx])
        text = data["text"]
        print(f"  Chunk {idx}:")
        print(f"    Début: {repr(text[:80])}...")
        print(f"    Fin:   ...{repr(text[-80:])}")
        print()

    print("Done.")


if __name__ == "__main__":
    main()
