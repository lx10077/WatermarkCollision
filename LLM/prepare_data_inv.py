import os
import time
import json
import torch
import pickle
import copy
import argparse
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from generation import WatermarkGenerate


def parse_args():
    """Parse command-line arguments for inverse-transform watermark data preparation."""
    parser = argparse.ArgumentParser(description="Prepare data using inverse-transform watermark generation.")

    # Core watermark configuration
    parser.add_argument('--method', default="Inverse", type=str, help="Watermarking method (default: Inverse).")
    parser.add_argument('--temperature', default=0.5, type=float, help="Softmax temperature for generation.")

    # Model & generation setup
    parser.add_argument('--model', default="facebook/opt-1.3b", type=str,
                        help="Model name from HuggingFace.")
    parser.add_argument('--seed', default=15485863, type=int, help="Random seed for reproducibility.")
    parser.add_argument('--c', default=4, type=int, help="Window size for watermark seeding.")
    parser.add_argument('--m', default=800, type=int, help="Number of tokens to generate per sample.")
    parser.add_argument('--seed_way', default="noncomm_prf", type=str, help="PRF seeding scheme.")
    parser.add_argument('--prompt_tokens', default=50, type=int, help="Prompt length before generation.")
    parser.add_argument('--buffer_tokens', default=20, type=int, help="Held-out tokens for context.")
    parser.add_argument('--truncate_vocab', default=8, type=int, help="Number of tokens excluded from vocab.")

    # Sampling controls
    parser.add_argument('--unique_threshold', default=300, type=int,
                        help='Minimum unique Y count within a sample to qualify as unique.')
    parser.add_argument('--unique_target', default=200, type=int,
                        help='Target number of unique samples to collect.')
    parser.add_argument('--ordered_first_n', default=200, type=int,
                        help='Keep the first N ordered samples (0 disables).')
    parser.add_argument('--max_prompts', default=6000, type=int,
                        help='Maximum total number of prompts processed (0 disables).')

    # Data source options
    parser.add_argument('--load_local_data', dest='load_local_data', action='store_true',
                        help='Use local c4.json file instead of online dataset.')
    parser.add_argument('--no-load_local_data', dest='load_local_data', action='store_false',
                        help='Use HuggingFace online C4 (streaming mode).')
    parser.add_argument('--local_path', default='c4/c4.json', type=str, help='Path to local c4.json file.')
    parser.set_defaults(load_local_data=True)

    return parser.parse_args()


def stream_dataset(tokenizer, prompt_tokens, buffer_tokens, load_local_data=False, local_path='c4/c4.json'):
    """Yield prompts (as tensors) satisfying the required length."""
    if load_local_data:
        with open(local_path, 'r') as f:
            for line in f:
                example = json.loads(line)
                text = example['text']
                tokens = tokenizer.encode(text, return_tensors='pt', truncation=True, max_length=2048 - buffer_tokens)[0]
                if len(tokens) < prompt_tokens + buffer_tokens:
                    continue
                yield tokens[-(buffer_tokens + prompt_tokens):-buffer_tokens]
    else:
        dataset = load_dataset("allenai/c4", "realnewslike", split="train", streaming=True)
        for example in dataset:
            text = example['text']
            tokens = tokenizer.encode(text, return_tensors='pt', truncation=True, max_length=2048 - buffer_tokens)[0]
            if len(tokens) < prompt_tokens + buffer_tokens:
                continue
            yield tokens[-(buffer_tokens + prompt_tokens):-buffer_tokens]


def main():
    """Generate unique samples using inverse-transform watermark."""
    args = parse_args()
    print(f"\nRunning: method={args.method}, temperature={args.temperature}")

    model_name = "1p3B" if args.model == "facebook/opt-1.3b" else args.model.split('/')[-1]
    exp_name = (f"text_data/{model_name}-{args.method}-c{args.c}-m{args.m}-{args.seed_way}-{args.seed}-"
                f"temp{args.temperature}-stream-unique.pkl")
    os.makedirs(os.path.dirname(exp_name), exist_ok=True)

    if os.path.exists(exp_name):
        print(f"[Skip] File already exists: {exp_name}")
        return

    torch.manual_seed(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using {torch.cuda.device_count()} GPU(s). Memory: {torch.cuda.memory_allocated() / 1e9:.2f} GB.")

    # Model setup
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, device_map="auto", offload_folder="./offload_folder")
    model.eval()
    vocab_size = model.get_output_embeddings().weight.shape[0]
    print(f"Model loaded ({time.time() - t0:.2f}s). Vocab size = {vocab_size}.")

    batch_size = 10 if args.model == "facebook/opt-1.3b" else 4
    prompt_stream = stream_dataset(tokenizer, args.prompt_tokens, args.buffer_tokens,
                                   args.load_local_data, args.local_path)

    WG = WatermarkGenerate(model=model, vocab_size=vocab_size, key=args.seed, text_length=args.m,
                           watermark_type=args.method, temperature=args.temperature,
                           text_window=args.c, seeding_scheme=args.seed_way)

    results = defaultdict(dict)
    results['args'] = copy.deepcopy(args)

    # --- Storage ---
    unique_tokens, unique_Ys_full, unique_probs_full = [], [], []
    unique_idx, unique_Ys_keys, unique_top_probs_keys, unique_Ys_counts_row = [], [], [], []

    ord_n, unique_target, unique_threshold = args.ordered_first_n, args.unique_target, args.unique_threshold
    ordered_tokens, ordered_Ys, ordered_probs, ordered_idx = [], [], [], []
    ordered_done = (ord_n == 0)

    pbar = tqdm(total=unique_target, desc="Collecting unique samples", leave=True)
    stop_early, total_seen = False, 0

    while not stop_early:
        if args.max_prompts > 0 and total_seen >= args.max_prompts:
            break

        batch_prompts = []
        try:
            for _ in range(4):
                p = next(prompt_stream)
                batch_prompts.append(p.unsqueeze(0))
        except StopIteration:
            if not batch_prompts:
                break

        batch_prompts = torch.vstack(batch_prompts)
        generated_tokens, Ys, top_probs = WG.full_generate(batch_prompts)

        for j_in_batch in range(batch_prompts.size(0)):
            Ys_row, probs_row = Ys[j_in_batch], top_probs[j_in_batch]
            gen_tokens_part = generated_tokens[j_in_batch, args.prompt_tokens:]

            if not ordered_done and len(ordered_tokens) < ord_n:
                ordered_tokens.append(gen_tokens_part.cpu())
                ordered_Ys.append(Ys_row.cpu())
                ordered_probs.append(probs_row.cpu())
                ordered_idx.append(total_seen + j_in_batch)
                if len(ordered_tokens) >= ord_n:
                    ordered_done = True

            ys_np = Ys_row.detach().cpu().numpy()
            uniq_vals_np, idx, inverse = np.unique(ys_np, return_index=True, return_inverse=True)
            order = np.argsort(idx)
            uniq_vals_np, inverse = uniq_vals_np[order], order[inverse]
            counts_row_np = np.bincount(inverse)
            first_idx_np = np.array([np.argmax(inverse == k) for k in range(len(uniq_vals_np))], dtype=np.int64)
            uniq_cnt = uniq_vals_np.shape[0]

            if uniq_cnt >= unique_threshold and len(unique_tokens) < unique_target:
                unique_tokens.append(gen_tokens_part.cpu())
                unique_Ys_full.append(Ys_row.cpu())
                unique_probs_full.append(probs_row.cpu())
                unique_idx.append(total_seen + j_in_batch)
                unique_Ys_keys.append(torch.from_numpy(uniq_vals_np).long())
                unique_top_probs_keys.append(probs_row[first_idx_np].detach().cpu())
                unique_Ys_counts_row.append(torch.from_numpy(counts_row_np).long())

                pbar.update(1)
                if len(unique_tokens) >= unique_target:
                    stop_early = True
                    break

        total_seen += batch_prompts.size(0)
        tqdm.write(f"[stats] unique={len(unique_tokens)}/{unique_target} | seen={total_seen}")

    pbar.close()

    results['unique_samples'] = {
        'tokens': torch.stack(unique_tokens, dim=0).cpu() if unique_tokens else torch.empty((0, args.m), dtype=torch.long),
        'Ys': torch.stack(unique_Ys_full, dim=0).cpu() if unique_Ys_full else torch.empty((0, args.m), dtype=torch.long),
        'top_probs': torch.stack(unique_probs_full, dim=0).cpu() if unique_probs_full else torch.empty((0, args.m), dtype=torch.float32),
        'idx': torch.tensor(unique_idx, dtype=torch.long).cpu(),
        'unique_Ys_keys': [t.cpu() for t in unique_Ys_keys],
        'unique_top_probs_keys': [t.cpu() for t in unique_top_probs_keys],
        'unique_Ys_counts_row': [t.cpu() for t in unique_Ys_counts_row],
    }

    with open(exp_name, 'wb') as f:
        pickle.dump(results, f)
    print(f"Saved results to {exp_name}")


if __name__ == "__main__":
    main()
