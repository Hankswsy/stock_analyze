"""多段滾動 walk-forward(WFA)—— 盤勢策略的黃金標準驗證。

訓練 N 年 → 測試 M 年 → 往前滾,每段只用當段訓練期定義盤勢與倉位,
把各段「樣本外」報酬接成一條連續曲線,涵蓋多頭/空頭/震盪各種環境,
再和買進持有比。這比單一切分公平得多。

用法:
    python main_wfa.py --period 12y --train_years 3 --test_years 1 --band 0.15
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

TDAYS = 252  # 一年約略交易日數


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="0050.TW")
    ap.add_argument("--period", default="12y")
    ap.add_argument("--lookback", type=int, default=20)
    ap.add_argument("--horizon", type=int, default=20)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--floor", type=float, default=0.2)
    ap.add_argument("--band", type=float, default=0.15)
    ap.add_argument("--fee", type=float, default=0.001425)
    ap.add_argument("--tax", type=float, default=0.003)
    ap.add_argument("--train_years", type=int, default=3)
    ap.add_argument("--test_years", type=int, default=1)
    ap.add_argument("--out", default="wfa.png")
    args = ap.parse_args()

    print(f"[1/3] 下載 {args.symbol} ({args.period}) + 算盤勢特徵 ...")
    df = fetch(args.symbol, period=args.period)
    feat = rg.regime_features(df, lookback=args.lookback)

    train_n = args.train_years * TDAYS
    test_n = args.test_years * TDAYS
    n = len(feat)
    print(f"      {n} 個交易日;訓練 {train_n} / 測試 {test_n} 每段")

    print("[2/3] 滾動各段:訓練期定盤勢 → 測試期回測 ...")
    all_res = []
    fold = 0
    start = 0
    while start + train_n + test_n <= n:
        tr_idx = feat.index[start: start + train_n]
        te_idx = feat.index[start + train_n: start + train_n + test_n]

        scaler, km = bt.fit_regimes(feat.loc[tr_idx], args.k)
        lab_tr = bt.assign(scaler, km, feat.loc[tr_idx])
        g_train = rg.forward_stats(df, tr_idx, lab_tr, horizon=args.horizon)
        weights = bt.regime_weights(g_train, floor=args.floor)

        lab_te = bt.assign(scaler, km, feat.loc[te_idx])
        res = bt.backtest(df, te_idx, lab_te, weights,
                          fee=args.fee, tax=args.tax, band=args.band)
        all_res.append(res)
        fold += 1
        print(f"      fold {fold}: 測試 {te_idx[0].date()} ~ {te_idx[-1].date()}")
        start += test_n  # 往前滾一個測試期

    if not all_res:
        raise ValueError("資料太短,拉長 --period 或縮短 train/test 年數")

    full = pd.concat(all_res).sort_index()
    full = full[~full.index.duplicated(keep="first")]

    m_net = bt.metrics(full["strat_ret"])
    m_bh = bt.metrics(full["mkt_ret"])
    table = pd.DataFrame({"依盤勢調倉(含成本)": m_net, "買進持有": m_bh})
    print(f"\n[3/3] 全樣本外({fold} 段接續,{full.index[0].date()} ~ "
          f"{full.index[-1].date()})績效:")
    print(table.to_string(float_format=lambda x: f"{x:+.4f}"))

    # 連續 OOS 淨值曲線
    eq_s = (1 + full["strat_ret"].fillna(0)).cumprod()
    eq_b = (1 + full["mkt_ret"].fillna(0)).cumprod()
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(full.index, eq_b, label="買進持有", color="#888", lw=1.5)
    ax.plot(full.index, eq_s, label="依盤勢調倉(含成本)", color="#cc6677", lw=1.8)
    ax.set_title(f"{args.symbol} 多段滾動樣本外淨值({fold} 段接續)")
    ax.set_ylabel("淨值(起始=1)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(args.out, dpi=130)
    plt.close(fig)
    print(f"\n淨值圖已存:{args.out}")


if __name__ == "__main__":
    main()
