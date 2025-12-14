# visualize_ablation.py
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import os


def make_method_name(row):
    return f"{row['YCbCr']}-{row['IDCT']}-{row['Dequant']}"


def plot_metric(df, metric, out_dir):
    plt.figure(figsize=(14, 6))

    methods = df["method"]
    values = df[metric]

    plt.bar(methods, values)
    plt.xticks(rotation=45, ha="right")
    plt.ylabel(metric)
    plt.title(f"Ablation Study: {metric}")

    plt.tight_layout()
    out_path = os.path.join(out_dir, f"{metric}.png")
    plt.savefig(out_path, dpi=200)
    plt.close()

    print(f"[Saved] {out_path}")


def main(args):
    df = pd.read_csv(args.csv)

    # ------------------------------------------------------------
    # Ignore per-run columns (run_1 ~ run_10)
    # ------------------------------------------------------------
    df = df[[c for c in df.columns if not c.startswith("run_")]]

    # ------------------------------------------------------------
    # Optional: filter by image
    # ------------------------------------------------------------
    if args.image is not None:
        df = df[df["Image"] == args.image]

    # ------------------------------------------------------------
    # Create method label
    # ------------------------------------------------------------
    df["method"] = df.apply(make_method_name, axis=1)

    # Sort for nicer plots
    df = df.sort_values(by=["YCbCr", "IDCT", "Dequant"])

    os.makedirs(args.out_dir, exist_ok=True)

    # ------------------------------------------------------------
    # Plot metrics
    # ------------------------------------------------------------
    plot_metric(df, "time_mean", args.out_dir)
    plot_metric(df, "PSNR", args.out_dir)
    plot_metric(df, "SSIM", args.out_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="Path to results_all.csv")
    parser.add_argument(
        "--image",
        default=None,
        help="Optional: filter by image name (e.g. lena)"
    )
    parser.add_argument(
        "--out_dir",
        default="figures",
        help="Output directory for plots"
    )
    args = parser.parse_args()

    main(args)
