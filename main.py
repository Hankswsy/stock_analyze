"""K 線組合無監督辨識 —— 端到端示範。

用法:
    python main.py 2330.TW --window 5 --period 5y
"""
from __future__ import annotations

import argparse

from kline.data import fetch
from kline.features import build_windows
from kline.discover import PatternDiscoverer
from kline.analyze import forward_returns
from kline.visualize import plot_archetypes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("symbol", nargs="?", default="2330.TW", help="股票代碼,台股加 .TW")
    ap.add_argument("--window", type=int, default=5, help="每組 K 線根數")
    ap.add_argument("--period", default="5y")
    ap.add_argument("--horizon", type=int, default=5, help="未來報酬看幾根之後")
    ap.add_argument("--out", default="patterns.png")
    args = ap.parse_args()

    print(f"[1/4] 下載 {args.symbol} ({args.period}) ...")
    df = fetch(args.symbol, period=args.period)
    print(f"      共 {len(df)} 根 K 線")

    print(f"[2/4] 切滑動視窗(每 {args.window} 根)...")
    X, idx = build_windows(df, n=args.window)
    print(f"      特徵矩陣 {X.shape}")

    print("[3/4] 無監督分群(自動挑群數)...")
    disc = PatternDiscoverer().fit(X)
    labels = disc.labels(X)
    print(f"      最佳群數 k={disc.best_k}  silhouette={disc.best_score:.3f}")

    print("[4/4] 未來報酬分析 + 畫原型圖 ...")
    summary = forward_returns(df, idx, labels, horizon=args.horizon)
    print("\n各型態群未來報酬(依平均報酬排序):")
    print(summary.to_string(float_format=lambda x: f"{x:+.4f}"))

    path = plot_archetypes(df, idx, labels, disc, X, args.window, out_path=args.out)
    print(f"\n原型圖已存:{path}")


if __name__ == "__main__":
    main()
