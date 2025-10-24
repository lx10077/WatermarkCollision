import os
import time
import torch
import pickle
import argparse
from generation import WatermarkGenerate
from transformers import AutoTokenizer, AutoModelForCausalLM


# ============================ Argument Parsing ==============================
parser = argparse.ArgumentParser(description="Experiment Settings")

parser.add_argument('--method', default="Gumbel", type=str, help="Watermark method to use.")
parser.add_argument('--model', default="facebook/opt-1.3b", type=str, help="Model name from HuggingFace.")
parser.add_argument('--temperature', default=0.3, type=float, help="Softmax temperature for generation.")
parser.add_argument('--seed', default=15485863, type=int, help="Random seed for reproducibility.")
parser.add_argument('--c', default=4, type=int, help="Window size for watermark seeding.")
parser.add_argument('--m', default=500, type=int, help="Number of new tokens to generate.")
parser.add_argument('--T', default=2000, type=int, help="Number of total prompts.")
parser.add_argument('--seed_way', default="noncomm_prf", type=str, help="PRF seeding scheme.")
parser.add_argument('--prompt_tokens', default=50, type=int, help="Prompt length before generation.")
parser.add_argument('--buffer_tokens', default=20, type=int, help="Held-out tokens for context.")
parser.add_argument('--max_seed', default=100000, type=int, help="(Unused).")
parser.add_argument('--norm', default=1, type=int, help="(Unused).")
parser.add_argument('--rt_translate', action='store_true', help="(Unused).")
parser.add_argument('--language', default="french", type=str, help="(Unused).")
parser.add_argument('--truncate_vocab', default=8, type=int, help="Number of tokens excluded from vocab.")

args = parser.parse_args()


# =============================== Main Routine ===============================
def main(model_name, method, temp, is_null=False):
    """Recalculate top probabilities for given tokens using a pre-trained LM."""
    print(model_name, method, temp, is_null)

    t0 = time.time()
    torch.manual_seed(args.seed)

    print(f"Using {torch.cuda.device_count()} GPUs - "
          f"{torch.cuda.memory_allocated() / 1e9:.2f} GB allocated per GPU.")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # ---------------------------- Load Model ----------------------------
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        offload_folder="./offload_folder"
    )
    model.eval()

    vocab_size = model.get_output_embeddings().weight.shape[0]
    print(f"Vocab size: {vocab_size}")
    print(f"Model loaded in {time.time() - t0:.2f} seconds.")

    # Initialize watermark generator
    WG = WatermarkGenerate(
        model=model,
        vocab_size=vocab_size,
        key=args.seed,
        text_length=args.m,
        watermark_type=method,
        temperature=temp,
        text_window=args.c,
        seeding_scheme=args.seed_way
    )

    # Short model name for saving
    model_short = "1p3B" if model_name == "facebook/opt-1.3b" else model_name.split("/")[-1]

    # ---------------------------- Load Data ----------------------------
    if is_null:
        last = "null"
        exp_name = f"text_data/{model_short}-{method}-c{args.c}-m{args.m}-T{args.T}-{args.seed_way}-{args.seed}-{last}.pkl"
        with open(exp_name, 'rb') as f:
            data = pickle.load(f)
        tokens = data['prompts']
        base_name = f"null_recoverPs/{model_short}-{method}-c{args.c}-m{args.m}-T{args.T}-{args.seed_way}-{args.seed}-temp{temp}-{last}-Ps-all.pkl"
        # saved_dir = "null_recoverPs"
        print(f"Loaded null data: {tokens.shape}")
    else:
        last = "stream-unique-raw"
        exp_name = f"text_data/{model_short}-{method}-c{args.c}-m800-{args.seed_way}-{args.seed}-temp{temp}-{last}.pkl"
        with open(exp_name, 'rb') as f:
            data = pickle.load(f)
        tokens = data['watermark']['tokens']
        base_name = f"recoverPs/{model_short}-{method}-c{args.c}-m{args.m}-T{args.T}-{args.seed_way}-{args.seed}-temp{temp}-{last}-Ps-all.pkl"

    print(f"Data keys: {data.keys()}")
    print(f"Token matrix size: {tokens.size()}")

    # --------------------- Recover Probabilities ---------------------
    recovered_Ps_dict = WG.recover_top_probs_no_prompt(
        tokens, delta=0.1, batch_size=5, temperature_list=[temp]
    )

    # --------------------------- Save Results ---------------------------
    os.makedirs(base_name, exist_ok=True)

    with open(base_name, 'wb') as f:
        pickle.dump(recovered_Ps_dict, f)

    print(f"Recovered probabilities saved to {base_name}")


# ================================ Execution =================================
if __name__ == "__main__":
    main(args.model, args.method, args.temperature, is_null=False)
    main(args.model, args.method, args.temperature, is_null=True)
