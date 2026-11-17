import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
fam = "Liberation Sans" if any("Liberation Sans" in f.name for f in font_manager.fontManager.ttflist) else "DejaVu Sans"
plt.rcParams["font.family"] = fam
counts = {"DDR": [6266, 630, 4477, 236, 913], "EyePACS": [25810, 2443, 5292, 873, 708], "APTOS": [1805, 370, 999, 193, 295]}
totals = {"DDR": 12522, "EyePACS": 35108, "APTOS": 3662}
colors = {"DDR": "#2F6FD0", "EyePACS": "#E8643C", "APTOS": "#1E9E6A"}
grades = ["0\nNo DR", "1\nMild", "2\nModerate", "3\nSevere", "4\nProliferative"]
fig, ax = plt.subplots(figsize=(5.8, 3.35), dpi=220)
ax.axvspan(0.5, 2.5, color="#FFF1C9", zorder=0)
ax.text(1.5, 76, "Early DR (target)", ha="center", va="top", fontsize=8.5, color="#6B5300", fontweight="bold")
w = 0.26
x = np.arange(5)
for k, (name, c) in enumerate(counts.items()):
    pct = np.array(c) / sum(c) * 100
    bars = ax.bar(x + (k - 1) * w, pct, w, color=colors[name], label=f"{name} (n = {totals[name]:,})", zorder=3)
    for xi, p in zip(x + (k - 1) * w, pct):
        ax.text(xi, p + 1.0, f"{p:.0f}" if p >= 9.5 else f"{p:.1f}", ha="center", va="bottom", fontsize=7, color="#222222", zorder=4)
ax.set_xticks(x); ax.set_xticklabels(grades, fontsize=8.5)
ax.set_ylim(0, 80); ax.set_xlim(-0.5, 4.5)
ax.set_ylabel("% of the dataset's images", fontsize=8.5)
ax.tick_params(axis="y", labelsize=8)
ax.yaxis.grid(True, color="#E3E6EA", linewidth=0.8, zorder=1); ax.set_axisbelow(True)
for s in ("top", "right"): ax.spines[s].set_visible(False)
ax.spines["left"].set_color("#9AA2AA"); ax.spines["bottom"].set_color("#9AA2AA")
ax.legend(loc="upper right", fontsize=8, frameon=False, bbox_to_anchor=(1.0, 0.98))
fig.tight_layout()
fig.savefig("charts/grade_mix.png", facecolor="white")
print(fam)
