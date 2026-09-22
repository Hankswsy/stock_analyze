"""盤勢偵測 —— 端到端,含 walk-forward 驗證。

在大盤(預設 0050)上把每天分成幾種市場狀態,看每種狀態的未來波動度,
並用 walk-forward 驗證「盤勢分類能否延續到樣本外」。

用法:
    python main_regime.py --period 8y --lookback 20 --horizon 20
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from kline.data import fetch
from kline import regime as rg


def fit_regimes(feat_train, k):
    scaler = StandardScaler().fit(feat_train.to_numpy())
    km = KMeans(n_clusters=k, n_init=10, random_state=42)
    km.fit(scaler.transform(feat_train.to_numpy()))
    return scaler, km


def assign(scaler, km, feat):
    return km.predict(scaler.transform(feat.to_numpy()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="0050.TW", help="要偵測盤勢的標的(通常放大盤)")
    ap.add_argument("--period", default="8y")
    ap.add_argument("--lookback", type=int, default=20)
    ap.add_argument("--horizon", type=int, default=20)
    ap.add_argument("--k", type=int, default=4, help="分幾種盤勢")
    ap.add_argument("--cutoff", default=None, help="walk-forward 切分日;不給用 70%")
    args = ap.parse_args()

    print(f"[1/4] 下載 {args.symbol} ({args.period}) ...")
    df = fetch(args.symbol, period=args.period)

    print(f"[2/4] 算盤勢特徵(回看 {args.lookback} 日)...")
    feat = rg.regime_features(df, lookback=args.lookback)
    print(f"      {len(feat)} 個交易日 x {feat.shape[1]} 特徵")

    # 決定 cutoff
    if args.cutoff:
        cutoff = pd.Timestamp(args.cutoff)
    else:
        cutoff = feat.index[int(len(feat) * 0.7)]
    tr = feat.index < cutoff
    te = feat.index >= cutoff
    print(f"      切分日 {cutoff.date()}(訓練 {tr.sum()} / 測試 {te.sum()} 日)")

    print(f"[3/4] 只用訓練期定義 {args.k} 種盤勢 ...")
    scaler, km = fit_regimes(feat[tr], args.k)
    lab_tr = assign(scaler, km, feat[tr])
    lab_te = assign(scaler, km, feat[te])

    g_in = rg.forward_stats(df, feat.index[tr], lab_tr, horizon=args.horizon)
    g_out = rg.forward_stats(df, feat.index[te], lab_te, horizon=args.horizon)
    names = rg.label_regimes(feat[tr], g_in)

    print("\n[4/4] 樣本內(訓練期)各盤勢的未來表現:")
    disp_in = g_in.copy()
    disp_in.insert(0, "名稱", [names[i] for i in disp_in.index])
    print(disp_in.to_string(float_format=lambda x: f"{x:+.4f}"))

    print("\n樣本外(測試期)同一套盤勢分類的未來表現:")
    disp_out = g_out.copy()
    disp_out.insert(0, "名稱", [names.get(i, "?") for i in disp_out.index])
    print(disp_out.to_string(float_format=lambda x: f"{x:+.4f}"))

    # 驗證重點:未來波動度的排序是否延續
    in_order = g_in.sort_values("fwd_vol_mean").index.tolist()
    out_order = g_out.sort_values("fwd_vol_mean").index.tolist()
    same = in_order == out_order
    print(f"\n未來波動度排序 — 訓練:{in_order}  測試:{out_order}")
    print("→ 排序一致,盤勢的波動預測力延續到樣本外 ✔"
          if same else "→ 排序有變動,部分延續(見數字)")

    path = rg.plot_validation(g_in, g_out, names, out_path="regime.png")
    print(f"\n驗證圖已存:{path}")


if __name__ == "__main__":
    main()
