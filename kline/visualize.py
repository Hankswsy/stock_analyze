"""把每個型態群的「代表性 K 線組合」畫出來。

代表 = 離群心最近的真實視窗(medoid),比直接畫群心更像真的 K 線。
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")  # 無視窗環境也能存圖
import matplotlib.pyplot as plt
import numpy as np

# Windows 內建的繁中字型,讓圖上中文正常顯示
plt.rcParams["font.sans-serif"] = ["Microsoft JhengHei", "Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False


def _plot_candles(ax, o, h, l, c, title):
    for i in range(len(o)):
        up = c[i] >= o[i]
        color = "#d33" if up else "#2a2"  # 台股習慣:紅漲綠跌
        ax.plot([i, i], [l[i], h[i]], color="#444", linewidth=1, zorder=1)
        lo, hi = (o[i], c[i]) if up else (c[i], o[i])
        ax.add_patch(plt.Rectangle((i - 0.3, lo), 0.6, max(hi - lo, 1e-9),
                                   color=color, zorder=2))
    ax.set_title(title, fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])


def plot_archetypes(df, idx, labels, discoverer, X, n_window: int,
                    out_path: str = "patterns.png"):
    """每群挑一個 medoid 視窗畫成小圖,拼成一張總覽。"""
    Xp = discoverer.pca.transform(discoverer.scaler.transform(X))
    centers = discoverer.model.cluster_centers_
    k = discoverer.best_k

    cols = min(k, 4)
    rows = (k + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.6, rows * 2.4))
    axes = np.atleast_1d(axes).ravel()

    ohlc = df[["Open", "High", "Low", "Close"]].to_numpy(dtype=float)
    for cl in range(k):
        mask = labels == cl
        members = np.where(mask)[0]
        # 找離群心最近的樣本 = medoid
        d = np.linalg.norm(Xp[members] - centers[cl], axis=1)
        medoid = members[d.argmin()]
        last = idx[medoid]
        win = ohlc[last - n_window + 1 : last + 1]
        _plot_candles(axes[cl], win[:, 0], win[:, 1], win[:, 2], win[:, 3],
                      f"群{cl}  n={mask.sum()}")

    for j in range(k, len(axes)):
        axes[j].axis("off")

    fig.suptitle(f"發現 {k} 種 K 線組合原型(每 {n_window} 根一組)", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path


def plot_archetypes_multi(frames, symbols, last_pos, labels, discoverer, X,
                          n_window: int, out_path: str = "patterns.png"):
    """多檔版:每群的 medoid 可能來自任一檔股票,標題附上是哪一檔。"""
    Xp = discoverer.pca.transform(discoverer.scaler.transform(X))
    centers = discoverer.model.cluster_centers_
    k = discoverer.best_k

    ohlc = {s: df[["Open", "High", "Low", "Close"]].to_numpy(dtype=float)
            for s, df in frames.items()}

    cols = min(k, 4)
    rows = (k + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.6, rows * 2.4))
    axes = np.atleast_1d(axes).ravel()

    for cl in range(k):
        mask = labels == cl
        members = np.where(mask)[0]
        d = np.linalg.norm(Xp[members] - centers[cl], axis=1)
        medoid = members[d.argmin()]
        sym = symbols[medoid]
        last = last_pos[medoid]
        win = ohlc[sym][last - n_window + 1 : last + 1]
        _plot_candles(axes[cl], win[:, 0], win[:, 1], win[:, 2], win[:, 3],
                      f"群{cl}  n={mask.sum()}\n{sym}")

    for j in range(k, len(axes)):
        axes[j].axis("off")

    fig.suptitle(f"多檔混合:{k} 種 K 線組合原型(每 {n_window} 根)", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path
