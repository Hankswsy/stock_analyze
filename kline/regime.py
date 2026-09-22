"""盤勢偵測:把每一天分類成不同「市場狀態」,並看它預測什麼。

和前面「型態預測方向」不同,這裡預測的是「未來波動度」。
波動度有叢聚性(高波動後常接高波動),是市場少數會延續的性質,
所以盤勢偵測遠比方向擇時穩健 —— 這正是無監督分群能發揮的地方。

盤勢特徵(都設計成無因次、可跨時比較):
  vol_20        近 20 日報酬標準差(波動水準)
  vol_ratio     近 5 日 / 近 20 日波動(波動在升溫還是降溫)
  trend_20      近 20 日累積報酬(方向)
  trend_strength trend_20 / vol_20(趨勢有多「乾淨」,類 t 值)
  drawdown      距近 20 日高點的回檔幅度
  vol_z         成交量相對 60 日均量的位階
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def regime_features(df: pd.DataFrame, lookback: int = 20) -> pd.DataFrame:
    """算每日盤勢特徵。回傳與 df 對齊、去除暖機期 NaN 的特徵表。"""
    close = df["Close"]
    ret = np.log(close).diff()

    vol_20 = ret.rolling(lookback).std()
    vol_5 = ret.rolling(5).std()
    feat = pd.DataFrame(index=df.index)
    feat["vol_20"] = vol_20
    feat["vol_ratio"] = vol_5 / vol_20
    feat["trend_20"] = close / close.shift(lookback) - 1.0
    feat["trend_strength"] = feat["trend_20"] / (vol_20 * np.sqrt(lookback))
    roll_max = close.rolling(lookback).max()
    feat["drawdown"] = close / roll_max - 1.0
    vol_ma = df["Volume"].rolling(60).mean()
    feat["vol_z"] = (df["Volume"] - vol_ma) / df["Volume"].rolling(60).std()

    return feat.dropna()


def forward_stats(df: pd.DataFrame, feat_index, labels: np.ndarray,
                  horizon: int = 20) -> pd.DataFrame:
    """每種盤勢的「未來」表現:未來波動度、未來報酬。

    未來波動度 = 之後 horizon 日的報酬標準差(年化)。
    這是盤勢偵測真正該預測、也預測得動的東西。
    """
    close = df["Close"].reindex(feat_index)
    ret = np.log(df["Close"]).diff()

    pos = df.index.get_indexer(feat_index)
    all_ret = ret.to_numpy()
    all_close = df["Close"].to_numpy()

    rows = []
    for p, lab in zip(pos, labels):
        fut_end = p + horizon
        if fut_end < len(all_close):
            fwd_vol = np.nanstd(all_ret[p + 1: fut_end + 1]) * np.sqrt(252)
            fwd_ret = all_close[fut_end] / all_close[p] - 1.0
            rows.append((lab, fwd_vol, fwd_ret))

    res = pd.DataFrame(rows, columns=["regime", "fwd_vol", "fwd_ret"])
    g = res.groupby("regime").agg(
        count=("fwd_vol", "count"),
        fwd_vol_mean=("fwd_vol", "mean"),
        fwd_ret_mean=("fwd_ret", "mean"),
        fwd_ret_win=("fwd_ret", lambda s: (s > 0).mean()),
    )
    return g


def plot_validation(g_in, g_out, names, out_path="regime.png"):
    """把訓練 vs 測試的未來波動度並排長條,直觀看延續性。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Microsoft JhengHei", "Microsoft YaHei"]
    plt.rcParams["axes.unicode_minus"] = False

    order = g_in.sort_values("fwd_vol_mean").index.tolist()
    labels = [names[i] for i in order]
    vin = [g_in.loc[i, "fwd_vol_mean"] for i in order]
    vout = [g_out.loc[i, "fwd_vol_mean"] for i in order]

    x = np.arange(len(order))
    w = 0.38
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(x - w / 2, vin, w, label="訓練期(樣本內)", color="#4477aa")
    ax.bar(x + w / 2, vout, w, label="測試期(樣本外)", color="#cc6677")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("未來波動度(年化)")
    ax.set_title("各盤勢的未來波動度:訓練 vs 測試(排序延續 = 可用訊號)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path


def label_regimes(feat: pd.DataFrame, g: pd.DataFrame) -> dict:
    """依 fwd_vol 高低,給每個 regime 一個好懂的名字(給人看用)。"""
    order = g.sort_values("fwd_vol_mean").index.tolist()
    names = {}
    tags = ["低波", "中低波", "中波", "中高波", "高波", "極高波"]
    for rank, reg in enumerate(order):
        tag = tags[rank] if rank < len(tags) else f"波動{rank}"
        trend = "偏多" if g.loc[reg, "fwd_ret_mean"] > 0 else "偏空"
        names[reg] = f"{tag}/{trend}"
    return names
