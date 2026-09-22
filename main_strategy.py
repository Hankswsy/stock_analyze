"""盤勢策略回測 —— 端到端。

只用訓練期定義盤勢與倉位規則,在測試期(樣本外)回測「依盤勢調倉」
對上「買進持有」,比較風險調整後報酬。

用法:
    python main_strategy.py --period 8y --lookback 20 --horizon 20 --k 4
"""
from __future__ import annotations

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from kline.data import fetch
from kline import regime as rg
from kline import backtest as bt

plt.rcParams["font.sans-serif"] = ["Microsoft JhengHei", "Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="0050.TW")
    ap.add_argument("--period", default="8y")
    ap.add_argument("--lookback", type=int, default=20)
    ap.add_argument("--horizon", type=int, default=20)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--floor", type=float, default=0.2, help="最低倉位")
    ap.add_argument("--cutoff", default=None)
    ap.add_argument("--fee", type=float, default=0.001425, help="手續費率(單邊)")
    ap.add_argument("--tax", type=float, default=0.003, help="證交稅(賣出)")
    ap.add_argument("--band", type=float, default=0.0, help="不交易帶(0=不設)")
    ap.add_argument("--out", default="strategy.png")
    args = ap.parse_args()

    print(f"[1/4] 下載 {args.symbol} ({args.period}) + 算盤勢特徵 ...")
    df = fetch(args.symbol, period=args.period)
    feat = rg.regime_features(df, lookback=args.lookback)

    if args.cutoff:
        cutoff = pd.Timestamp(args.cutoff)
    else:
        cutoff = feat.index[int(len(feat) * 0.7)]
    tr = feat.index < cutoff
    te = feat.index >= cutoff
    print(f"      切分日 {cutoff.date()}(訓練 {tr.sum()} / 測試 {te.sum()} 日)")

    print(f"[2/4] 訓練期定義 {args.k} 種盤勢 + 倉位規則 ...")
    scaler, km = bt.fit_regimes(feat[tr], args.k)
    lab_tr = bt.assign(scaler, km, feat[tr])
    g_train = rg.forward_stats(df, feat.index[tr], lab_tr, horizon=args.horizon)
    weights = bt.regime_weights(g_train, floor=args.floor)
    names = rg.label_regimes(feat[tr], g_train)
    print("      各盤勢倉位:")
    for reg in sorted(weights):
        print(f"        {names[reg]:>10}  預期波動 {g_train.loc[reg,'fwd_vol_mean']:.3f}"
              f"  → 倉位 {weights[reg]:.2f}")

    print("[3/4] 測試期(樣本外)回測(含交易成本)...")
    lab_te = bt.assign(scaler, km, feat[te])
    res = bt.backtest(df, feat.index[te], lab_te, weights,
                      fee=args.fee, tax=args.tax, band=args.band)

    m_net = bt.metrics(res["strat_ret"])
    m_gross = bt.metrics(res["gross_ret"])
    m_bh = bt.metrics(res["mkt_ret"])
    table = pd.DataFrame({
        "調倉(含成本)": m_net,
        "調倉(無成本)": m_gross,
        "買進持有": m_bh,
    })
    print("\n樣本外績效對照:")
    print(table.to_string(float_format=lambda x: f"{x:+.4f}"))

    ts = bt.turnover_stats(res)
    print(f"\n換手統計:調倉 {ts['調倉次數']} 次,年化換手 {ts['年化換手']:.2f} 倍,"
          f"累計成本 {ts['累計成本']:.4f}(佔本金 {ts['累計成本']*100:.2f}%)")

    print("[4/4] 畫淨值曲線 ...")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(res.index, res["equity_bh"], label="買進持有", color="#888", lw=1.5)
    ax.plot(res.index, res["equity_strat"], label="依盤勢調倉", color="#cc6677", lw=1.8)
    ax.set_title(f"{args.symbol} 樣本外淨值:依盤勢調倉 vs 買進持有")
    ax.set_ylabel("淨值(起始=1)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(args.out, dpi=130)
    plt.close(fig)
    print(f"      淨值圖已存:{args.out}")


if __name__ == "__main__":
    main()
