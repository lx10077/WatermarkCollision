import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import argparse
from detectors import (
    OptInvDetector,
    IdInvDetector,
    RepeatOptInvDetector,
    rowwise_unique_topk_multi,
    pad_with_uniform,
)


def parse_args():
    """Parse command-line arguments for Inverse watermark detection experiments."""
    parser = argparse.ArgumentParser(description="Evaluate detectors on Inverse watermarked data.")
    parser.add_argument('--method', default="Inverse", type=str, help="Watermark method name.")
    parser.add_argument('--model', default="facebook/opt-1.3b", type=str, help="Model name.")
    parser.add_argument('--temperature', default=0.5, type=float, help="Temperature for sampling.")
    parser.add_argument('--seed', default=15485863, type=int, help="Random seed for reproducibility.")
    parser.add_argument('--c', default=4, type=int, help="Window size for watermark seeding.")
    parser.add_argument('--m', default=500, type=int, help="Number of generated tokens.")
    parser.add_argument('--T', default=1000, type=int, help="Number of total prompts.")
    parser.add_argument('--seed_way', default="noncomm_prf", type=str, help="Seeding scheme.")
    parser.add_argument('--prompt_tokens', default=50, type=int, help="Prompt length.")
    parser.add_argument('--buffer_tokens', default=20, type=int, help="Buffer tokens for context.")
    parser.add_argument('--max_seed', default=100000, type=int)
    parser.add_argument('--norm', default=1, type=int)
    parser.add_argument('--rt_translate', action='store_true')
    parser.add_argument('--language', default="french", type=str)
    parser.add_argument('--truncate_vocab', default=8, type=int, help="Truncated vocabulary size.")

    # === Detection Parameters ===
    parser.add_argument('--alpha', default=0.01, type=float, help="Target Type I error rate.")
    parser.add_argument('--min_count', default=300, type=int, help="Number of unique elements required.")
    parser.add_argument('--min_null_count', default=300, type=int, help="Number of unique null elements required.")
    parser.add_argument('--long_m', default=800, type=int, help="Length of each generated sequence.")
    parser.add_argument('--voc', default=50272, type=int, help="Vocabulary size.")
    return parser.parse_args()


def main():
    """Run detection and evaluation for Inverse watermark data."""
    args = parse_args()

    alpha = args.alpha
    min_count = args.min_count
    min_null_count = args.min_null_count
    long_m = args.long_m
    temp = args.temperature
    method = args.method
    voc = args.voc

    model_name = "1p3B" if args.model == "facebook/opt-1.3b" else args.model.split("/")[-1]
    print(f"==> Running {method} detection with temp={temp}, alpha={alpha}, min_count={min_count}")

    # ------------------ Paths ------------------
    source_data = f"text_data/{model_name}-{method}-c{args.c}-m{long_m}-{args.seed_way}-{args.seed}-temp{temp}-stream-unique.pkl"
    exp_name = f"plot_data/{model_name}-{method}-c{args.c}-m{long_m}-{args.seed_way}-{args.seed}-temp{temp}-stream-unique.pkl"
    result_save_path = exp_name.replace(".pkl", f"-{min_count}-all_scores.pkl")

    os.makedirs(os.path.dirname(result_save_path), exist_ok=True)
    os.makedirs("plot", exist_ok=True)

    # ------------------ Load or Compute ------------------
    if os.path.exists(result_save_path):
        print(f"Loading cached detector results from {result_save_path}")
        with open(result_save_path, "rb") as f:
            detector_outputs = pickle.load(f)
    else:
        with open(source_data, "rb") as f:
            results = pickle.load(f)

        generated_Ys_all = results['watermark']['Ys'].numpy()
        generated_Ds_all = results['watermark']['Ds'].numpy()
        generated_top_probs = results['watermark']['top_probs'].numpy()

        # Load recovered probabilities
        probs_dir = f"recoverPs/{model_name}-{method}-c{args.c}-m{long_m}-{args.seed_way}-{args.seed}-temp{temp}-stream-unique-raw-Ps-all.pkl"
        with open(probs_dir, "rb") as f:
            computed_probs = pickle.load(f)[temp].numpy()

        # Extract unique top-k samples
        unique_Ys_all, alter_count = rowwise_unique_topk_multi(generated_Ys_all, k=min_count)

        print("Watermarking data summary:")
        print(f"  All Ys: {generated_Ys_all.shape}")
        print(f"  Unique Ys: {unique_Ys_all.shape}")
        print(f"  Avg repeat count: {np.mean(alter_count):.2f}")

        # ------------------ Load Null Data ------------------
        exp_name = f"text_data/{model_name}-{method}-c{args.c}-m500-T2000-{args.seed_way}-{args.seed}-null.pkl"
        with open(exp_name, "rb") as f:
            data = pickle.load(f)
            null_Ys, null_Ds = data["null_Y"], data["null_D"]
        null_Ys = pad_with_uniform(null_Ys, target_len=500)
        unique_null_Ys_all, null_count = rowwise_unique_topk_multi(null_Ys, k=min_null_count)

        # Load recovered null probabilities
        null_probs_dir = f"null_recoverPs/{model_name}-{method}-c{args.c}-m500-T2000-{args.seed_way}-{args.seed}-temp{temp}-null-Ps-all.pkl"
        with open(null_probs_dir, "rb") as f:
            null_probs = pickle.load(f)[temp].numpy()

        # ------------------ Detection ------------------
        print("Running detectors...")
        detector_outputs = {
            "type2_opt_rep": OptInvDetector(generated_Ys_all[:, :min_count], vocab_size=voc, delta0=0.05, alpha=alpha),
            "type2_opt_unq": OptInvDetector(unique_Ys_all, vocab_size=voc, delta0=0.05, alpha=alpha),
            "type2_new_exact_ora": RepeatOptInvDetector(generated_Ys_all, generated_Ds_all, generated_top_probs, voc, alpha=alpha, min_count=min_count),
            "type2_new_exact_sur": RepeatOptInvDetector(generated_Ys_all, generated_Ds_all, computed_probs, voc, alpha=alpha, min_count=min_count),
            "type2_id_rep": IdInvDetector(generated_Ys_all[:, :min_count], alpha=alpha),
            "type2_id_unq": IdInvDetector(unique_Ys_all, alpha=alpha),

            "type1_id_rep": IdInvDetector(null_Ys[:, :min_null_count], alpha=alpha),
            "type1_id_unq": IdInvDetector(unique_null_Ys_all, alpha=alpha),
            "type1_opt_rep": OptInvDetector(null_Ys[:, :min_null_count], vocab_size=voc, delta0=0.05, alpha=alpha),
            "type1_opt_unq": OptInvDetector(unique_null_Ys_all, vocab_size=voc, delta0=0.05, alpha=alpha),
            "type1_new_exact_sur": RepeatOptInvDetector(null_Ys, null_Ds, null_probs, voc, alpha=alpha, min_count=480),
        }

        with open(result_save_path, "wb") as f:
            pickle.dump(detector_outputs, f)
        print(f"Saved computed detector results to {result_save_path}")

    # ------------------ Visualization ------------------
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))

    def plot_curve(ax, arr, label, linestyle):
        x, y = arr
        ax.plot(x, y, label=label, linestyle=linestyle)

    def plot_curve1(ax, arr, label, linestyle):
        x, y = arr
        ax.plot(x, 1 - y, label=label, linestyle=linestyle)

    plot_curve(axs[0], detector_outputs["type1_id_unq"], "Id Unq", ':')
    plot_curve(axs[0], detector_outputs["type1_opt_unq"], "Opt Unq", ':')
    plot_curve(axs[0], detector_outputs["type1_id_rep"], "Id Rep", ':')
    plot_curve(axs[0], detector_outputs["type1_opt_rep"], "Opt Rep", ':')
    plot_curve(axs[0], detector_outputs["type1_new_exact_sur"], "Our Recover", '-')
    axs[0].set_xlabel("Text length")
    axs[0].set_ylabel("Type I error")

    plot_curve1(axs[1], detector_outputs["type2_id_unq"], "Id Unq", ':')
    plot_curve1(axs[1], detector_outputs["type2_opt_unq"], "Opt Unq", ':')
    plot_curve1(axs[1], detector_outputs["type2_id_rep"], "Id Rep", ':')
    plot_curve1(axs[1], detector_outputs["type2_opt_rep"], "Opt Rep", ':')
    plot_curve1(axs[1], detector_outputs["type2_new_exact_sur"], "Our Recover", '-')
    plot_curve1(axs[1], detector_outputs["type2_new_exact_ora"], "Our Oracle", '-')
    axs[1].set_xlabel("Text length")
    axs[1].set_ylabel("Type II error")

    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    save_path = f"plot/{model_name}-{method}-c{args.c}-m{long_m}-{args.seed_way}-{args.seed}-temp{temp}-{min_count}{temp}all_scores-final.pdf"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved plot to {save_path}")


if __name__ == "__main__":
    main()
