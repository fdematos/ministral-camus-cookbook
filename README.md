# Ministral Camus Fine-Tuning Cookbook

This project fine-tunes `Ministral-3-8B-Instruct-2512` so the model answers in a Camus-like French prose style while still handling modern prompts.

The pipeline has two phases:
- **Phase 1 (style):** teach tone, rhythm, and writing style from corpus text.
- **Phase 2 (chat):** teach conversational question/answer behavior using curated chat pairs.

Main deployment artifact:
- `ministral-8b-albert-camus/model-final/`

## Quick start (using published datasets)

If you want the fastest path to a working final model, run:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

hf download unsloth/Ministral-3-8B-Instruct-2512 --local-dir ministral-8b
hf download freddm/albert-camus-chat-style-chat-dataset --repo-type dataset --local-dir albert-camus-chat-style-chat-dataset

torchrun --nproc_per_node=2 --master_port=29500 script/train-style.py
python script/merge-phase1.py
torchrun --nproc_per_node=2 --master_port=29500 script/train-chat.py
python script/merge-phase2.py
python script/verify-merge.py --phase2
python script/eval-chat.py
```

This produces a deployable final model in:
- `ministral-8b-albert-camus/model-final/`

---

## Overview

This repository gives you a practical end-to-end workflow:
1. Prepare and clean source texts.
2. Build a style training dataset.
3. Train a style LoRA (Phase 1).
4. Generate chat pairs from corpus passages.
5. Train a chat LoRA on top of Phase 1 (Phase 2).
6. Merge adapters into a final standalone model.
7. Run manual quality checks to confirm style and behavior.

---

## Base model and dataset links

## Base model

- https://huggingface.co/unsloth/Ministral-3-8B-Instruct-2512

```bash
hf download unsloth/Ministral-3-8B-Instruct-2512 --local-dir ministral-8b
```

## Dataset package

- https://huggingface.co/datasets/freddm/albert-camus-chat-style-chat-dataset

```bash
hf download freddm/albert-camus-chat-style-chat-dataset --repo-type dataset --local-dir albert-camus-chat-style-chat-dataset
```

Expected dataset files:
- `albert-camus-chat-style-chat-dataset/phase1/style-train.jsonl`
- `albert-camus-chat-style-chat-dataset/phase2/chat-pairs-corpus-final-clean.jsonl`
- `albert-camus-chat-style-chat-dataset/phase2/chat-pairs-light-boost-clean.jsonl`

---

## Project structure

```text
.
├── albert-camus-chat-style-chat-dataset/
│   ├── phase1/
│   │   └── style-train.jsonl
│   └── phase2/
│       ├── chat-pairs-corpus-final-clean.jsonl
│       └── chat-pairs-light-boost-clean.jsonl
├── books/                                  # source ebooks/texts (input for corpus prep)
├── data/
│   ├── corpus/                             # cleaned corpus text files
│   └── dataset/                            # local generation outputs (intermediate)
├── ministral-8b/                           # downloaded base model
├── ministral-8b-albert-camus/
│   ├── lora-style/                         # Phase 1 adapter
│   ├── lora-chat/                          # Phase 2 adapter
│   └── model-final/                        # final merged model for deployment
└── script/
    ├── prepare-corpus.py
    ├── build-style-dataset.py
    ├── train-style.py
    ├── eval-style.py
    ├── merge-phase1.py
    ├── extract-corpus-pairs.py
    ├── extract-corpus-pairs-light.py
    ├── train-chat.py
    ├── eval-chat.py
    ├── merge-phase2.py
    └── verify-merge.py
```

Note: `model-style-merged/` is an intermediate artifact created on demand by `script/merge-phase1.py`.

---

## Setup

### 1) Create environment and install dependencies

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**What this step does:** creates an isolated Python environment and installs all libraries used by data prep, training, merge, and evaluation scripts.

**Why it matters:** prevents missing-package errors and keeps your system Python clean.

### 2) Add API key for Phase 2 data generation

Create a `.env` file with:

```bash
OPENROUTER_API_KEY=your_key_here
```

**What this step does:** provides credentials for the API model that generates one question per selected passage.

**Why it matters:** Phase 2 pair generation scripts fail fast without this key.

---

## Phase 1 (step-by-step)

### Step 1 - Prepare cleaned corpus text

```bash
python script/prepare-corpus.py
```

**What this step does:**
- Converts source EPUB/PDF/TXT files into clean plain-text files.
- Removes front matter, metadata, editorial noise, and structural artifacts.
- Extracts Camus-only correspondence content.

**Why it matters:** cleaner input text gives cleaner style learning.

**Output:** `data/corpus/*.txt`

### Step 2 - Build style dataset chunks

```bash
python script/build-style-dataset.py
```

**What this step does:**
- Tokenizes corpus text with the base model tokenizer.
- Splits text into overlapping chunks (target around 2048 tokens).
- Filters very short chunks and shuffles the final dataset.

**Why it matters:** training needs stable chunked examples, not full book files.

**Output:** `data/dataset/style-train.jsonl`

### Step 3 - Train Phase 1 style LoRA

```bash
torchrun --nproc_per_node=2 --master_port=29500 script/train-style.py
```

**What this step does:** trains the first adapter (`lora-style`) on style-only text.

**Why it matters:** this adapter becomes the style foundation for all later steps.

Default Phase 1 dataset path in script:
- `albert-camus-chat-style-chat-dataset/phase1/style-train.jsonl`

Optional custom dataset path:

```bash
python script/train-style.py --dataset-path data/dataset/style-train.jsonl
```

### Step 4 - Evaluate style behavior

```bash
python script/eval-style.py
```

**What this step does:** runs qualitative checks for style completion, memorization risk, and modern prompt generalization.

**Why it matters:** confirms style transfer quality before chat tuning.

### Step 5 - Merge base + style adapter

```bash
python script/merge-phase1.py
```

**What this step does:** merges base model + `lora-style` into an intermediate merged checkpoint.

**Why it matters:** Phase 2 training uses this merged checkpoint as starting model.

---

## Phase 2 (step-by-step)

### Step 1 - Generate corpus-based chat pairs

```bash
python script/extract-corpus-pairs.py
```

**What this step does:**
- Splits source text into filtered passages/chunks.
- For each kept chunk, an API LLM generates one user question.
- Local model (`ministral-8b + lora-style`) generates the assistant answer from:
  - generated question,
  - reference chunk.
- Applies quality filters (invalid questions/answers, anti-copy overlap, formatting rules).

**Why it matters:** builds the main high-quality chat dataset while preserving learned style.

### Step 2 - Generate light-boost chat pairs

```bash
python script/extract-corpus-pairs-light.py
```

**What this step does:** same core pipeline as Step 1, but tuned toward lighter/more concrete themes and with metadata logging.

**Why it matters:** improves coverage and balance of conversational behavior.

### Step 3 - Train Phase 2 chat LoRA

```bash
torchrun --nproc_per_node=2 --master_port=29500 script/train-chat.py
```

**What this step does:**
- Loads `model-style-merged` as base.
- Loads both clean Phase 2 datasets.
- Concatenates + shuffles them.
- Applies chat template formatting.
- Trains `lora-chat`.

**Why it matters:** teaches conversational behavior while keeping the Phase 1 style.

Default Phase 2 dataset paths in script:
- `albert-camus-chat-style-chat-dataset/phase2/chat-pairs-corpus-final-clean.jsonl`
- `albert-camus-chat-style-chat-dataset/phase2/chat-pairs-light-boost-clean.jsonl`

Optional custom dataset paths:

```bash
python script/train-chat.py \
  --dataset-corpus-path /path/to/chat-pairs-corpus-final-clean.jsonl \
  --dataset-light-path /path/to/chat-pairs-light-boost-clean.jsonl
```

### Step 4 - Evaluate chat behavior

```bash
python script/eval-chat.py
```

**What this step does:** runs prompt suites across philosophy, modern topics, personal prompts, anti-assistant traps, and concrete scenarios.

**Why it matters:** verifies that outputs stay in prose style and avoid generic assistant formatting.

### Step 5 - Merge final model

```bash
python script/merge-phase2.py
```

**What this step does:** merges base + style LoRA + chat LoRA into `model-final/`.

**Why it matters:** produces a standalone model for deployment/inference.

### Step 6 - Verify merge integrity

```bash
python script/verify-merge.py --phase2
```

**What this step does:** checks that target weights changed and that generation differs before/after merge.

**Why it matters:** catches ineffective merges before publishing or deployment.

---

## Script reference

- `script/prepare-corpus.py` - Converts raw EPUB/PDF/TXT sources into cleaned corpus text files.
- `script/build-style-dataset.py` - Builds chunked style JSONL from cleaned corpus text.
- `script/train-style.py` - Trains Phase 1 style LoRA.
- `script/eval-style.py` - Evaluates style quality and memorization risk.
- `script/merge-phase1.py` - Merges base + style LoRA into the intermediate style-merged model.
- `script/extract-corpus-pairs.py` - Builds core Phase 2 chat pairs via chunking, API question generation, local answer generation, and filtering.
- `script/extract-corpus-pairs-light.py` - Builds a light-boost Phase 2 pair set with additional theme constraints and metadata.
- `script/train-chat.py` - Trains Phase 2 chat LoRA on merged clean Phase 2 datasets.
- `script/eval-chat.py` - Evaluates final conversational behavior on multiple prompt families.
- `script/merge-phase2.py` - Merges base + both LoRAs into final deployable weights.
- `script/verify-merge.py` - Verifies that merge changed expected weights and generation behavior.

---

## Validation and quality checks

- Before training: verify model and dataset paths exist.
- After Phase 1: run `eval-style.py` and review style consistency.
- After Phase 2 data generation: review rejection causes and sample outputs.
- After final merge: run `verify-merge.py --phase2`.
- Final manual review: run `eval-chat.py` and check prose quality and anti-assistant behavior.

---

## Artifacts and deployment notes

Main artifacts:
- `ministral-8b-albert-camus/lora-style` - Phase 1 adapter.
- `ministral-8b-albert-camus/lora-chat` - Phase 2 adapter.
- `ministral-8b-albert-camus/model-final` - final merged model for inference/deployment.

Deployment notes:
- Keep LoRAs if you want adapter-based iteration.
- Use `model-final` if you want one standalone model directory.
- Do not publish secrets (`.env`) or local caches.

---

## Known limitations

- Evaluation is mostly qualitative/manual.
- Quality is sensitive to corpus cleaning and pair filtering strictness.
- Some prompts can still trigger generic assistant formatting.
- Phase 2 pair generation depends on API availability and cost.
- Source text licensing/copyright constraints may affect redistribution.
