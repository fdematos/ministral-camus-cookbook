## AGENTS.md

Guidelines for AI agents working on the Ministral-Camus fine-tuning project.

## Project Overview

Fine-tuning `Ministral-3-8B-Instruct-2512` to adopt Albert Camus' literary style using QLoRA via Unsloth. Python 3.12+ project with data preparation and training scripts.

## Build & Run Commands

### Environment Setup
```bash
# Create and activate virtualenv
python3.12 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
# For training (Phase 1):
pip install unsloth torch transformers peft trl bitsandbytes accelerate datasets pymupdf
```

### Data Preparation
```bash
# Prepare corpus from EPUBs/PDFs (cleans metadata, extracts text)
python script/prepare-corpus.py

# Build dataset (chunks text into 2048 tokens)
python script/build-style-dataset.py
```

### Training
```bash
# Single GPU
torchrun --nproc_per_node=1 script/train-style.py

# Multi-GPU (2 GPUs)
torchrun --nproc_per_node=2 --master_port=29500 script/train-style.py
```

### Evaluation
```bash
# Evaluate style, anti-memorization, modern topics
python script/eval-style.py
```

### Linting (Recommended)
```bash
# Format with ruff
ruff format script/

# Check with ruff
ruff check script/
```

## Code Style Guidelines

### Language & Naming
- **English only** for code, variables, functions, and commit messages
- Use French only for Camus-related content strings
- `snake_case` for functions, variables, modules
- `UPPER_CASE` for constants at module level
- `PascalCase` for classes (if any)

### Imports
```python
# Standard library first
import sys
import re
from pathlib import Path

# Third-party packages
import pymupdf
from ebooklib import epub
from unsloth import FastVisionModel

# Local imports (none in this project)
```

### Project Structure
```python
# Constants at module level (UPPER_CASE)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_PATH = PROJECT_ROOT / "data" / "dataset" / "style-train.jsonl"
MAX_SEQ_LENGTH = 2048

# Main entry point pattern
def main():
    print("=== Phase 1 : Style Pretraining LoRA ===\n")
    # Implementation

if __name__ == "__main__":
    main()
```

### Path Handling
- Always use `pathlib.Path`, never string concatenation
- Define `PROJECT_ROOT` at module level
- Build paths with `/` operator: `PROJECT_ROOT / "data" / "corpus"`

### Functions
- Keep functions focused and small (< 50 lines ideally)
- Use early returns to reduce nesting
- Fail fast: validate inputs at function entry
- No type hints required (project doesn't use them)

### Error Handling
```python
# Fail fast with clear messages
if not DATASET_PATH.exists():
    print(f"ERREUR: {DATASET_PATH} introuvable", file=sys.stderr)
    sys.exit(1)

# Propagate errors, don't swallow them
try:
    result = risky_operation()
except SpecificException as e:
    print(f"Failed: {e}", file=sys.stderr)
    raise  # or sys.exit(1)
```

### Comments
- **Avoid comments** - code should be self-documenting
- Only exception: document non-obvious intent or constraints
- Use descriptive variable names instead

### Side Effects
- Isolate side effects (file I/O, printing)
- Favor pure functions for data transformation
- Print progress messages for long operations

## Philosophy (from existing AGENTS.md)

- **Simple and Direct** - Less code is better. No unnecessary abstractions.
- **Readable & Explicit** - Logic must be obvious. Self-documenting names.
- **Reuse First** - Duplicate code forbidden. Extend existing patterns.
- **Robustness** - Handle errors explicitly. Never swallow errors.
- **Fail Fast** - Detect and report errors immediately.

## Before Making Changes

Always verify:
1. Does this logic already exist in the codebase?
2. Can I use an existing pattern instead of inventing a new one?
3. Is this the right directory for this code? (`script/` for executables)
4. How will I test this change? (Run the script, check output)
5. Will this break existing functionality?

## Definition of Done

1. **Verify it works** - Run the code, check output
2. **Check for regressions** - Ensure existing scripts still work
3. **Clean up** - Remove dead code, unused imports, debug prints
4. **Re-read your changes** - Would a human reviewer approve?
5. **If you cannot verify** - Say so explicitly. Do not assume success.

## Testing

This project currently has **no automated tests**. Verification is manual:
- Run `python script/prepare-corpus.py` → Check `data/corpus/` files
- Run `python script/build-style-dataset.py` → Check chunks quality
- Run training → Monitor loss convergence
- Run evaluation → Check style quality

When adding new scripts, follow the pattern: `main()` function with clear output messages.

## Dataset Generation Resumption

The `extract-corpus-pairs.py` script supports resuming interrupted generation:

- Uses **deterministic shuffle with seed 42** for reproducibility
- Writes pairs incrementally to disk (append mode with flush)
- To resume after interruption:
  ```python
  # Count already generated pairs
  if output_file.exists():
      already_done = sum(1 for _ in open(output_file))
      passages = passages[already_done:]  # Skip processed ones
  ```
- This ensures no duplicates and efficient continuation
- Always verify API key validity before resuming long runs
