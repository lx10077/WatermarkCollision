import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import argparse
from detectors import (
    LogDetector, ArsDetector, OptGumDetector, TrGofDetector,
    RepeatWeightLogGumDetectorNew, SwitchOptGumDetector,
    rowwise_unique_topk_multi, pad_with_uniform
)


def parse_args():
    """Parse command-line arguments for Gumbel watermark detection experiments."""
    parser = argparse.ArgumentParser(description="Evaluate detectors on Gumbel watermarked data.")
    parser.add_argument('--method', default="Gumbel", type=str, help="Watermark method name.")
    parser.add_argument('--model', default="facebook/opt-1.3b", type=str, help="Model name.")
    parser.add_argument('--temperature', default=0.3, type=float, help="Temperature for sampling.")
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
    parser.add_argument('--long_m', default=800, type=int, help="Length of each generated sequence.")
    return parser.parse_args()


def main():
    """Run detection and evaluation for Gumbel watermark data."""
    args = parse_args()

    alpha = args.alpha
    min_count = args.min_count
    long_m = args.long_m
    temp = args.temp
    method = args.method

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
        generated_top_probs = results['watermark']['top_probs'].numpy()

        probs_dir = f"recoverPs/{model_name}-{method}-c{args.c}-m800-{args.seed_way}-{args.seed}-temp{temp}-stream-unique-raw-Ps-all.pkl"
        with open(probs_dir, "rb") as f:
            computed_probs = pickle.load(f)[temp].numpy()
        ordered_Ys = results['ordered_firstN']['Ys'].numpy()

        # Extract unique top-k samples
        unique_Ys, unique_top_probs, computed_probs, alter_count = rowwise_unique_topk_multi(
            generated_Ys_all, generated_top_probs, computed_probs, k=min_count
        )

        print("Watermarking data summary:")
        print(f"  All Ys: {generated_Ys_all.shape}, mean = {generated_Ys_all.mean():.3f}")
        print(f"  Unique Ys: {unique_Ys.shape}, mean = {unique_Ys.mean():.3f}")
        print(f"  Avg repeat count: {np.mean(alter_count):.2f}")

        # ------------------ Load Null Data ------------------
        exp_name = f"text_data/{model_name}-{method}-c{args.c}-m500-T2000-noncomm_prf-{args.seed}-null.pkl"
        with open(exp_name, "rb") as f:
            null_Ys = pickle.load(f)["null_Y"]
        null_Ys = pad_with_uniform(null_Ys)

        probs_dir = f"null_recoverPs/{model_name}-{method}-c{args.c}-m500-T2000-{args.seed_way}-{args.seed}-temp{temp}-null-Ps-all.pkl"
        with open(probs_dir, "rb") as f:
            null_probs = pickle.load(f)[temp].numpy()

        unique_null_Ys, null_probs, null_count = rowwise_unique_topk_multi(
            null_Ys, null_probs, k=250
        )

        print("Null data summary:")
        print(f"  All Ys: {null_Ys.shape}, mean = {null_Ys.mean():.3f}")
        print(f"  Unique Ys: {unique_null_Ys.shape}, mean = {unique_null_Ys.mean():.3f}")

        # ------------------ Detection ------------------
        detector_outputs = {
            "type2_trgof_unq": TrGofDetector(unique_Ys, alpha),
            "type2_new_switch_ora": SwitchOptGumDetector(unique_Ys, unique_top_probs, alter_count, alpha),
            "type2_new_switch_sur": SwitchOptGumDetector(unique_Ys, computed_probs, alter_count, alpha),
            "type2_new_exact_ora": RepeatWeightLogGumDetectorNew(unique_Ys, unique_top_probs, alpha),
            "type2_new_exact_sur": RepeatWeightLogGumDetectorNew(unique_Ys, computed_probs, alpha),
            "type2_trgof_rep": TrGofDetector(ordered_Ys, alpha),
            "type2_log_unq": LogDetector(unique_Ys, alpha),
            "type2_ars_unq": ArsDetector(unique_Ys, alpha),
            "type2_opt_unq": OptGumDetector(unique_Ys, delta0=0.1, alpha=alpha),
            "type2_opt_unq_05": OptGumDetector(unique_Ys, delta0=0.05, alpha=alpha),

            "type1_log_unq": LogDetector(unique_null_Ys, alpha),
            "type1_ars_unq": ArsDetector(unique_null_Ys, alpha),
            "type1_opt_unq": OptGumDetector(unique_null_Ys, delta0=0.1, alpha=alpha),
            "type1_opt_unq_05": OptGumDetector(unique_null_Ys, delta0=0.05, alpha=alpha),
            "type1_trgof_unq": TrGofDetector(unique_null_Ys, alpha),
            "type1_new_exact_sur": RepeatWeightLogGumDetectorNew(unique_null_Ys, null_probs, alpha),
            "type1_new_switch_sur": SwitchOptGumDetector(unique_null_Ys, null_probs, null_count, alpha),
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

    plot_curve(axs[0], detector_outputs["type1_log_unq"], "Log Unq", ':')
    plot_curve(axs[0], detector_outputs["type1_ars_unq"], "ARS Unq", ':')
    plot_curve(axs[0], detector_outputs["type1_opt_unq"], "Opt Unq", ':')
    plot_curve(axs[0], detector_outputs["type1_trgof_unq"], "GoF Unq", ':')
    plot_curve(axs[0], detector_outputs["type1_new_exact_sur"], "Our Recover", '-')
    plot_curve(axs[0], detector_outputs["type1_new_switch_sur"], "Our Switch", '--')
    axs[0].set_xlabel("Text length")
    axs[0].set_ylabel("Type I error")

    plot_curve1(axs[1], detector_outputs["type2_log_unq"], "Log Unq", ':')
    plot_curve1(axs[1], detector_outputs["type2_ars_unq"], "ARS Unq", ':')
    plot_curve1(axs[1], detector_outputs["type2_opt_unq"], "Opt Unq", ':')
    plot_curve1(axs[1], detector_outputs["type2_trgof_unq"], "GoF Unq", ':')
    plot_curve1(axs[1], detector_outputs["type2_new_exact_sur"], "Our Recover", '-')
    plot_curve1(axs[1], detector_outputs["type2_new_exact_ora"], "Our Oracle", '-')
    plot_curve1(axs[1], detector_outputs["type2_new_switch_sur"], "Our Switch Recover", '--')
    plot_curve1(axs[1], detector_outputs["type2_new_switch_ora"], "Our Switch Oracle", '--')
    axs[1].set_xlabel("Text length")
    axs[1].set_ylabel("Type II error")
    axs[1].set_yscale("log")

    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    save_path = f"plot/{model_name}-{method}-c{args.c}-m{long_m}-{args.seed_way}-{args.seed}-temp{temp}-{min_count}{temp}all_scores-final.pdf"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved plot to {save_path}")


if __name__ == "__main__":
    main()
