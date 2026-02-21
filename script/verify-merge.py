import argparse
import json
import sys
from pathlib import Path

import torch
from safetensors import safe_open

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = PROJECT_ROOT / "ministral-8b-albert-camus"

BASE_PATH = PROJECT_ROOT / "ministral-8b"
MERGED_PATH = ARTIFACTS_DIR / "model-style-merged"
FINAL_PATH = ARTIFACTS_DIR / "model-final"

LORA_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]

LAYERS_TO_CHECK = [0, 5, 10, 15, 20, 25, 30, 33]

NON_TARGET_KEYS = [
    "language_model.model.layers.0.input_layernorm.weight",
    "language_model.model.norm.weight",
    "language_model.lm_head.weight",
    "language_model.model.embed_tokens.weight",
]

PROMPT = "L'absurde naît quand l'homme"
SYSTEM = "Tu es Albert Camus. Réponds en prose courte et lucide, sans liste."


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--phase2",
        action="store_true",
        help="Vérifier Phase 2: compare model-style-merged vs model-final",
    )
    return parser.parse_args()


def load_tensors(path, key):
    single_file = path / "model.safetensors"
    if single_file.exists():
        with safe_open(str(single_file), framework="pt") as f:
            return f.get_tensor(key)

    index_path = path / "model.safetensors.index.json"
    if index_path.exists():
        weight_map = json.loads(index_path.read_text())["weight_map"]
        shard = weight_map.get(key)
        if not shard:
            return None
        with safe_open(str(path / shard), framework="pt") as f:
            return f.get_tensor(key)

    return None


def check_weights(before_path, after_path, phase_label):
    print(f"=== Test 1 : Vérification des poids ({phase_label}) ===\n")
    print(f"  Avant: {before_path.name}")
    print(f"  Après: {after_path.name}\n")

    target_modified = 0
    target_identical = 0
    non_target_modified = 0
    non_target_identical = 0

    target_keys = []
    for layer_idx in LAYERS_TO_CHECK:
        for module in LORA_TARGET_MODULES:
            if module in ("q_proj", "k_proj", "v_proj", "o_proj"):
                target_keys.append(
                    f"language_model.model.layers.{layer_idx}.self_attn.{module}.weight"
                )
            else:
                target_keys.append(
                    f"language_model.model.layers.{layer_idx}.mlp.{module}.weight"
                )

    all_keys = target_keys + NON_TARGET_KEYS

    for key in all_keys:
        is_target = key in target_keys

        before_tensor = load_tensors(before_path, key)
        if before_tensor is None:
            print(f"  SKIP (pas dans avant): {key}")
            continue

        after_tensor = load_tensors(after_path, key)
        if after_tensor is None:
            print(f"  SKIP (pas dans après): {key}")
            continue

        is_equal = torch.equal(before_tensor, after_tensor)
        max_diff = float((after_tensor - before_tensor).abs().max())

        if is_target:
            if is_equal:
                target_identical += 1
                print(f"  FAIL {key}: IDENTIQUE (LoRA non appliqué)")
            else:
                target_modified += 1
                print(f"  OK   {key}: max_diff={max_diff:.6f}")
        else:
            if is_equal:
                non_target_identical += 1
                print(f"  OK   {key}: identique (attendu)")
            else:
                non_target_modified += 1
                print(f"  WARN {key}: modifié (inattendu) max_diff={max_diff:.6f}")

    print(
        f"\n  Modules LoRA modifiés: {target_modified}/{target_modified + target_identical}"
    )
    print(
        f"  Modules non-LoRA identiques: {non_target_identical}/{non_target_identical + non_target_modified}"
    )

    if target_identical > 0:
        print(f"\n  FAIL: {target_identical} module(s) LoRA identiques")
        return False

    if target_modified == 0:
        print("\n  FAIL: aucun module LoRA modifié")
        return False

    print("\n  PASS: poids LoRA correctement mergés")
    return True


def check_generation(before_path, after_path, phase_label):
    print(f"\n=== Test 2 : Comparaison génération ({phase_label}) ===\n")

    from transformers import AutoProcessor, Mistral3ForConditionalGeneration

    processor = AutoProcessor.from_pretrained(str(before_path))
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": PROMPT},
    ]

    inputs = processor.tokenizer.apply_chat_template(
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

    gen_kwargs = dict(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=60,
        do_sample=False,
        temperature=None,
        top_p=None,
        pad_token_id=processor.tokenizer.eos_token_id,
        eos_token_id=processor.tokenizer.eos_token_id,
    )

    def decode(outputs):
        return processor.tokenizer.decode(
            outputs[0][input_ids.shape[1] :], skip_special_tokens=True
        ).strip()

    print(f"  Chargement {before_path.name}...")
    before_model = Mistral3ForConditionalGeneration.from_pretrained(
        str(before_path), torch_dtype=torch.bfloat16, device_map="auto"
    )
    with torch.inference_mode():
        before_text = decode(before_model.generate(**gen_kwargs))
    del before_model
    torch.cuda.empty_cache()

    print(f"  Chargement {after_path.name}...")
    after_model = Mistral3ForConditionalGeneration.from_pretrained(
        str(after_path), torch_dtype=torch.bfloat16, device_map="auto"
    )
    with torch.inference_mode():
        after_text = decode(after_model.generate(**gen_kwargs))
    del after_model
    torch.cuda.empty_cache()

    print(f"\n  AVANT: {before_text[:200]}")
    print(f"  APRÈS: {after_text[:200]}")

    if before_text == after_text:
        print("\n  FAIL: génération identique (merge sans effet)")
        return False

    print("\n  PASS: générations différentes (merge a un effet)")
    return True


def main():
    args = parse_args()

    if args.phase2:
        phase_label = "Phase 2"
        before_path = MERGED_PATH
        after_path = FINAL_PATH
        required = [
            (MERGED_PATH, "merged (Phase 1)"),
            (FINAL_PATH, "final (Phase 1+2)"),
        ]
    else:
        phase_label = "Phase 1"
        before_path = BASE_PATH
        after_path = MERGED_PATH
        required = [
            (BASE_PATH, "base"),
            (MERGED_PATH, "merged (Phase 1)"),
        ]

    print(f"=== Vérification du merge {phase_label} ===\n")

    for path, name in required:
        if not path.exists():
            print(f"ERREUR: {name} non trouvé: {path}", file=sys.stderr)
            sys.exit(1)

    weights_ok = check_weights(before_path, after_path, phase_label)
    gen_ok = check_generation(before_path, after_path, phase_label)

    print("\n" + "=" * 50)
    if weights_ok and gen_ok:
        print(f"VERDICT: PASS — merge {phase_label} validé")
    else:
        print(f"VERDICT: FAIL — problème avec le merge {phase_label}")
        if not weights_ok:
            print("  - Poids LoRA non intégrés")
        if not gen_ok:
            print("  - Génération identique")
    print("=" * 50)


if __name__ == "__main__":
    main()
