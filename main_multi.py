"""K 線組合無監督辨識 —— 多檔混合訓練。

把多檔股票的視窗混在一起分群,找出「跨股票反覆出現」的組合,
避免單檔過擬合。用 breadth 欄位判斷型態是否真的有普遍性。

用法:
    python main_multi.py --symbols 2330.TW 2317.TW 2454.TW 0050.TW --window 5
    python main_multi.py --file symbols.txt --period 5y
"""
from __future__ import annotations

import argparse

from kline.data import fetch, fetch_many
from kline.features import build_windows_multi
from kline.discover import PatternDiscoverer
from kline.analyze import forward_returns_multi, forward_returns_multi_excess
from kline.visualize import plot_archetypes_multi

# 台股權值股 + 幾檔常見標的當預設籃子
DEFAULT_SYMBOLS = [
    "2330.TW", "2317.TW", "2454.TW", "2308.TW", "2412.TW",
    "2882.TW", "2881.TW", "1301.TW", "1303.TW", "0050.TW",
]


def load_symbols(args) -> list[str]:
    if args.file:
        with open(args.file, encoding="utf-8") as f:
            return [ln.strip() for ln in f if ln.strip() and not ln.startswith("#")]
    if args.symbols:
        return args.symbols
    return DEFAULT_SYMBOLS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", nargs="+", help="股票代碼清單(空白分隔)")
    ap.add_argument("--file", help="從檔案讀代碼(每行一個)")
    ap.add_argument("--window", type=int, default=5)
    ap.add_argument("--period", default="5y")
    ap.add_argument("--horizon", type=int, default=5)
    ap.add_argument("--benchmark", default="0050.TW",
                    help="去市場化用的大盤基準;設 none 則不去市場化")
    ap.add_argument("--beta", type=float, default=1.0,
                    help="市場中性化的 beta 係數(權值股實際常 >1)")
    ap.add_argument("--out", default="patterns_multi.png")
    args = ap.parse_args()

    symbols = load_symbols(args)
    print(f"[1/4] 下載 {len(symbols)} 檔 ({args.period}) ...")
    frames = fetch_many(symbols, period=args.period)
    total = sum(len(d) for d in frames.values())
    print(f"      成功 {len(frames)} 檔,共 {total} 根 K 線")

    print(f"[2/4] 多檔切視窗(每 {args.window} 根)...")
    X, syms, pos = build_windows_multi(frames, n=args.window)
    print(f"      合併特徵矩陣 {X.shape}")

    print("[3/4] 無監督分群(自動挑群數)...")
    disc = PatternDiscoverer().fit(X)
    labels = disc.labels(X)
    print(f"      最佳群數 k={disc.best_k}  silhouette={disc.best_score:.3f}")

    use_excess = args.benchmark.lower() != "none"
    if use_excess:
        print(f"[4/4] 去市場化未來報酬分析(基準 {args.benchmark}, beta={args.beta})...")
        bench = fetch(args.benchmark, period=args.period)
        summary = forward_returns_multi_excess(
            frames, syms, pos, labels, bench,
            horizon=args.horizon, beta=args.beta)
        metric = "超額報酬(已扣大盤)"
    else:
        print("[4/4] 跨股票未來報酬分析(未去市場化)...")
        summary = forward_returns_multi(frames, syms, pos, labels, horizon=args.horizon)
        metric = "原始報酬"
    print(f"\n各型態群 — {metric}(依 signal_t 排序;breadth=多少比例股票上為正):")
    print(summary.to_string(float_format=lambda x: f"{x:+.4f}"))

    path = plot_archetypes_multi(frames, syms, pos, labels, disc, X,
                                 args.window, out_path=args.out)
    print(f"\n原型圖已存:{path}")


if __name__ == "__main__":
    main()
