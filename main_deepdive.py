"""深挖最顯著的型態群 —— 端到端。

自動挑「signal_t 最負(最顯著偏空)」的群深入分析。
用法:
    python main_deepdive.py --period 5y --window 5
    python main_deepdive.py --cluster 2   # 指定要看哪一群
"""
from __future__ import annotations

import argparse

from kline.data import fetch, fetch_many
from kline.features import build_windows_multi
from kline.discover import PatternDiscoverer
from kline.analyze import forward_returns_multi_excess
from kline import deepdive as dd
from main_multi import DEFAULT_SYMBOLS, load_symbols


def describe_shape(shape, n_window):
    print(f"\n── 平均形狀(每 {n_window} 根;數值為視窗內正規化尺度 0~1)──")
    print(f"{'根':>3} {'方向':>8} {'實體':>6} {'上影線':>7} {'下影線':>7}")
    for b in shape["bars"]:
        print(f"{b['bar']:>3} {b['方向']:>8} {b['實體']:>6.3f} "
              f"{b['上影線']:>7.3f} {b['下影線']:>7.3f}")
    t = shape["trend"]
    dirn = "先高後低(下降)" if t < 0 else "先低後高(上升)"
    print(f"整段趨勢:{t:+.3f} → {dirn}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", nargs="+")
    ap.add_argument("--file")
    ap.add_argument("--window", type=int, default=5)
    ap.add_argument("--period", default="5y")
    ap.add_argument("--horizon", type=int, default=5)
    ap.add_argument("--benchmark", default="0050.TW")
    ap.add_argument("--beta", type=float, default=1.0)
    ap.add_argument("--cluster", type=int, default=None,
                    help="指定深挖哪一群;不給則自動挑最顯著偏空的")
    args = ap.parse_args()

    symbols = load_symbols(args)
    print(f"[1/3] 下載 {len(symbols)} 檔 + 基準 {args.benchmark} ...")
    frames = fetch_many(symbols, period=args.period)
    bench = fetch(args.benchmark, period=args.period)

    print(f"[2/3] 切視窗 + 分群 ...")
    X, syms, pos = build_windows_multi(frames, n=args.window)
    disc = PatternDiscoverer().fit(X)
    labels = disc.labels(X)
    summary = forward_returns_multi_excess(
        frames, syms, pos, labels, bench, horizon=args.horizon, beta=args.beta)
    print(f"      k={disc.best_k}  各群超額報酬 signal_t:")
    print(summary[["count", "mean_ret", "signal_t", "breadth"]].to_string(
        float_format=lambda x: f"{x:+.4f}"))

    # 挑目標群:指定優先,否則挑 signal_t 最負的
    if args.cluster is not None:
        target = args.cluster
    else:
        target = summary["signal_t"].idxmin()
    print(f"\n[3/3] 深挖群 {target}(signal_t={summary.loc[target,'signal_t']:+.3f})")

    # 1. 形狀
    shape = dd.average_shape(X, labels, target, args.window)
    describe_shape(shape, args.window)

    # 明細
    rec = dd.build_cluster_records(frames, syms, pos, labels, bench,
                                   target, horizon=args.horizon, beta=args.beta)

    # 2. 各股票
    print("\n── 各股票超額報酬(依平均由低到高)──")
    print(dd.per_symbol(rec).to_string(float_format=lambda x: f"{x:+.4f}"))

    # 3. 各年份
    print("\n── 各年份超額報酬 ──")
    print(dd.per_year(rec).to_string(float_format=lambda x: f"{x:+.4f}"))

    # 4. 月份分布
    print("\n── 出現次數月份分布 ──")
    print(dd.per_month(rec).to_string())


if __name__ == "__main__":
    main()
