"""Walk-forward 驗證 —— 端到端。

用法:
    python main_walkforward.py --period 8y --cutoff 2023-07-01
    python main_walkforward.py --period 5y   # 不給 cutoff 則用「70% 時點」自動切
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from kline.data import fetch, fetch_many
from kline.features import build_windows_multi
from kline import walkforward as wf
from main_multi import load_symbols


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", nargs="+")
    ap.add_argument("--file")
    ap.add_argument("--window", type=int, default=5)
    ap.add_argument("--period", default="8y")
    ap.add_argument("--horizon", type=int, default=5)
    ap.add_argument("--benchmark", default="0050.TW")
    ap.add_argument("--beta", type=float, default=1.0)
    ap.add_argument("--cutoff", default=None,
                    help="訓練/測試切分日 YYYY-MM-DD;不給則自動用 70% 時點")
    ap.add_argument("--method", choices=["excess", "xs"], default="excess",
                    help="excess=減大盤指數;xs=橫斷面去均值(不受指數成分偏差)")
    args = ap.parse_args()

    symbols = load_symbols(args)
    print(f"[1/3] 下載 {len(symbols)} 檔 + 基準 {args.benchmark} ({args.period}) ...")
    frames = fetch_many(symbols, period=args.period)
    bench = fetch(args.benchmark, period=args.period)

    print(f"[2/3] 切視窗 ...")
    X, syms, pos = build_windows_multi(frames, n=args.window)
    dates = wf.window_dates(frames, syms, pos)

    # 決定 cutoff
    if args.cutoff:
        cutoff = pd.Timestamp(args.cutoff)
    else:
        cutoff = pd.Timestamp(np.sort(dates)[int(len(dates) * 0.7)])
    print(f"      切分日:{cutoff.date()}  "
          f"(訓練 {(dates < cutoff).sum()} / 測試 {(dates >= cutoff).sum()} 視窗)")

    print(f"[3/3] 只用訓練期分群 → 套到測試期評估(方法={args.method})...")
    if args.method == "xs":
        ins, outs, disc = wf.run_xs(frames, syms, pos, X, cutoff,
                                    horizon=args.horizon)
    else:
        ins, outs, disc = wf.run(frames, syms, pos, X, bench, cutoff,
                                 horizon=args.horizon, beta=args.beta)
    print(f"      訓練期定義出 k={disc.best_k} 種型態")

    cmp = wf.compare(ins, outs)
    print("\n樣本內(訓練期) vs 樣本外(測試期)超額報酬對照:")
    print(cmp.to_string(float_format=lambda x: f"{x:+.3f}"))

    print("\n判讀:'延續'=樣本外仍顯著同向可信;'消失'=樣本內顯著但樣本外歸零"
          "(過擬合/特定時期)。")


if __name__ == "__main__":
    main()
