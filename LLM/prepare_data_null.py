import os
import time
import torch
import pickle
import copy
import argparse
from collections import defaultdict
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from generation import WatermarkGenerate


def parse_args():
    """Parse command-line arguments for null (non-watermarked) data generation."""
    parser = argparse.ArgumentParser(description="Generate non-watermarked (null) data.")
    parser.add_argument('--model', default="facebook/opt-1.3b", type=str, help="Model name from HuggingFace")
    parser.add_argument('--seed', default=15485863, type=int, help="Random seed for reproducibility")
    parser.add_argument('--c', default=4, type=int, help="Window size for watermark seeding")
    parser.add_argument('--m', default=500, type=int, help="Number of tokens to generate")
    parser.add_argument('--T', default=2000, type=int, help="Number of total prompts")
    parser.add_argument('--seed_way', default="noncomm_prf", type=str, help="PRF seeding scheme")
    parser.add_argument('--prompt_tokens', default=50, type=int, help="Length of each prompt before generation")
    parser.add_argument('--buffer_tokens', default=20, type=int, help="Held-out tokens for context")
    parser.add_argument('--truncate_vocab', default=8, type=int, help="Number of tokens excluded from vocab")
    return parser.parse_args()


def main(method: str):
    """Generate null (human-written) data for given watermark method.

    Args:
        method: 'Gumbel' or 'Inverse'. Determines how watermark scores are computed.
    """
    args = parse_args()
    args.method = method
    print(f"Running null data generation for method: {method}")

    start_time = time.time()
    torch.manual_seed(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, device_map="auto", offload_folder="./offload_folder"
    )
    model.eval()
    vocab_size = model.get_output_embeddings().weight.shape[0]
    print(f"Model loaded ({time.time() - start_time:.2f}s). Using vocab size = {vocab_size}.")

    # =================== Prompt Preparation ===========================
    buffer_tokens, prompt_tokens, T = args.buffer_tokens, args.prompt_tokens, args.T
    dataset = load_dataset("allenai/c4", "realnewslike", split="train", streaming=True)
    ds_iterator = iter(dataset)

    prompts = []
    while len(prompts) < T:
        example = next(ds_iterator)
        text = example['text']
        tokens = tokenizer.encode(text, return_tensors='pt', truncation=True, max_length=2048 - buffer_tokens)[0]
        if len(tokens) < prompt_tokens + buffer_tokens:
            continue
        prompt = tokens[-(buffer_tokens + prompt_tokens):-buffer_tokens]
        prompts.append(prompt)

    prompts = torch.vstack(prompts)
    print(f"Collected {len(prompts)} prompts. Dataset ready.\n")

    # =================== Watermark Computation =========================
    WG = WatermarkGenerate(
        model=model,
        vocab_size=vocab_size,
        key=args.seed,
        text_length=args.m,
        watermark_type=method,
        temperature=0.5,
        text_window=args.c,
        seeding_scheme=args.seed_way,
    )

    results = defaultdict(dict)
    results['args'] = copy.deepcopy(args)
    results['prompts'] = copy.deepcopy(prompts)

    gen_start = time.time()
    if method == "Inverse":
        Null_Ys, Null_Ds = WG.compute_Ys(corrupted_data=prompts, prompts=None, is_null=True, output_xi=True)
        results['null_Y'], results['null_D'] = copy.deepcopy(Null_Ys), copy.deepcopy(Null_Ds)
    else:
        Null_Ys = WG.compute_Ys(corrupted_data=prompts, prompts=None, is_null=True, output_xi=False)
        results['null_Y'] = copy.deepcopy(Null_Ys)

    print(f"Samples generated ({time.time() - gen_start:.2f}s).")

    # =================== Save Results =========================
    model_name = "1p3B" if args.model == "facebook/opt-1.3b" else args.model.split("/")[-1]
    exp_name = f"text_data/{model_name}-{method}-c{args.c}-m{args.m}-T{args.T}-{args.seed_way}-{args.seed}-null.pkl"
    os.makedirs(os.path.dirname(exp_name), exist_ok=True)
    with open(exp_name, "wb") as f:
        pickle.dump(results, f)
    print(f"Saved results to {exp_name}")

if __name__ == "__main__":
    for method in ["Gumbel", "Inverse"]:
        main(method)
