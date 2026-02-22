import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_ROOT / "albert-camus-chat-style-chat-dataset" / "phase2"

TARGET_FILES = [
    "chat-pairs-corpus-final-clean.jsonl",
    "chat-pairs-light-boost-clean.jsonl",
]


def strip_system_messages(input_path, output_path):
    count = 0
    stripped = 0

    with (
        open(input_path, encoding="utf-8") as f_in,
        open(output_path, "w", encoding="utf-8") as f_out,
    ):
        for line in f_in:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            messages = entry["messages"]
            original_len = len(messages)
            messages = [m for m in messages if m["role"] != "system"]

            if len(messages) < original_len:
                stripped += 1

            entry["messages"] = messages
            json.dump(entry, f_out, ensure_ascii=False)
            f_out.write("\n")
            count += 1

    return count, stripped


def main():
    print("=== Strip system prompt from Phase 2 datasets ===\n")

    for filename in TARGET_FILES:
        input_path = DATASET_DIR / filename
        if not input_path.exists():
            print(f"  SKIP: {input_path} not found", file=sys.stderr)
            continue

        backup_path = input_path.with_suffix(".jsonl.bak")
        input_path.rename(backup_path)

        count, stripped = strip_system_messages(backup_path, input_path)
        print(f"  {filename}: {count} entries, {stripped} system messages removed")
        print(f"    Backup: {backup_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
