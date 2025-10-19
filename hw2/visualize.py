import re
import matplotlib.pyplot as plt
import numpy as np

# === Load and parse the log file ===
log_path = "dct_result.log"  # Change to your filename
with open(log_path, "r", encoding="utf-8") as f:
    log_text = f.read()

# === Parse data ===
data = {"Basic": {"DCT": {}, "IDCT": {}, "PSNR": {}},
        "Fast(1D)": {"DCT": {}, "IDCT": {}, "PSNR": {}}}

for m in re.finditer(r"\[(Basic|Fast\(1D\)) (DCT|IDCT) runtime, uv_size (\d+)\]: ([\d.]+)", log_text):
    model, metric, uv, value = m.groups()
    data[model][metric][int(uv)] = float(value)

for m in re.finditer(r"\[(Basic|Fast\(1D\)) PSNR, uv_size (\d+)\]: ([\d.]+)", log_text):
    model, uv, value = m.groups()
    data[model]["PSNR"][int(uv)] = float(value)

# === Prepare data arrays ===
uv_sizes = sorted(set(list(data["Basic"]["DCT"].keys()) +
                      list(data["Fast(1D)"]["DCT"].keys())))

def arr(model, metric):
    return np.array([data[model][metric].get(u, np.nan) for u in uv_sizes])

basic_dct  = arr("Basic", "DCT")
fast_dct   = arr("Fast(1D)", "DCT")
basic_idct = arr("Basic", "IDCT")
fast_idct  = arr("Fast(1D)", "IDCT")
basic_psnr = arr("Basic", "PSNR")
fast_psnr  = arr("Fast(1D)", "PSNR")

# === Helper function to save annotated plot ===
def save_plot(x, y1, y2, title, ylabel, filename, unit="s"):
    plt.figure(figsize=(8, 6))
    plt.plot(x, y1, "o-", label="Basic", color="royalblue")
    plt.plot(x, y2, "s-", label="Fast(1D)", color="orange")

    # x-axis: log2 scale with ticks labeled as 2,4,8,16,...
    plt.xscale("log", base=2)
    plt.xticks(x, [str(int(u)) for u in x])

    plt.title(title)
    plt.xlabel("uv_size (with log2 scale)")
    plt.ylabel(ylabel)
    plt.grid(True, ls="--", lw=0.5)
    plt.legend()

    # Annotate each data point
    for i, u in enumerate(x):
        if title.startswith("DCT"):
            plt.text(u, y1[i] +200, f"{y1[i]:.2f}{unit}",
                    fontsize=8, ha="center", va="bottom", color="blue")
            plt.text(u, y2[i] -150, f"{y2[i]:.2f}{unit}",
                    fontsize=8, ha="center", va="top", color="green")
        elif title.startswith("IDCT"):
            plt.text(u, y1[i] +100, f"{y1[i]:.2f}{unit}",
                    fontsize=8, ha="center", va="bottom", color="blue")
            plt.text(u, y2[i] -50, f"{y2[i]:.2f}{unit}",
                    fontsize=8, ha="center", va="top", color="green")
        elif title.startswith("PSNR"):
            plt.text(u, y1[i], f"{y1[i]:.2f}{unit}",
                    fontsize=8, ha="center", va="bottom", color="blue")
            plt.text(u, y2[i], f"{y2[i]:.2f}{unit}",
                    fontsize=8, ha="center", va="top", color="green")


    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    print(f"✅ Saved: {filename}")
    plt.close()

# === Save all three figures ===
save_plot(uv_sizes, basic_dct, fast_dct,
          "DCT Runtime vs uv_size", "Runtime (s)", "viz_dct_runtime.png")

save_plot(uv_sizes, basic_idct, fast_idct,
          "IDCT Runtime vs uv_size", "Runtime (s)", "viz_idct_runtime.png")

save_plot(uv_sizes, basic_psnr, fast_psnr,
          "PSNR vs uv_size", "PSNR (dB)", "viz_psnr.png", unit="dB")
