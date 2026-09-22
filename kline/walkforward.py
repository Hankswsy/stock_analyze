"""Walk-forward 驗證:型態在「沒看過的未來」還有沒有效?

做法:
  1. 只用訓練期(較早的資料)分群,定義並固定型態(群心)。
  2. 用這套固定型態,對訓練期與測試期各自貼標籤。
  3. 分別算「樣本內(in-sample, 訓練期)」與「樣本外(out-of-sample,
     測試期)」的超額報酬,逐群比對。
  若某群樣本內顯著、樣本外卻退回 0,代表它是過擬合 / 特定時期現象,
  不能拿來實戰擇時。

關鍵:分群(KMeans)不看報酬、只看形狀,所以「用訓練期分群」這步
不構成未來函數偏誤;報酬僅在事後評估使用。
群編號在訓練/測試間一致(同一個 fitted 模型),比對才有意義。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from kline.discover import PatternDiscoverer
from kline.analyze import forward_returns_multi_excess, forward_returns_cross_sectional


def window_dates(frames: dict, symbols: np.ndarray, last_pos: np.ndarray):
    """每個視窗「最後一根」的日期(用來切訓練/測試)。"""
    return np.array([frames[s].index[p] for s, p in zip(symbols, last_pos)])


def split_by_date(dates, cutoff: pd.Timestamp):
    """回傳 (訓練遮罩, 測試遮罩)。訓練=cutoff 之前,測試=cutoff 之後。"""
    cutoff = pd.Timestamp(cutoff)
    train = dates < cutoff
    test = dates >= cutoff
    return train, test


def run(frames, symbols, last_pos, X, benchmark, cutoff,
        horizon=5, beta=1.0, k_range=range(4, 13)):
    """執行一次 walk-forward,回傳 (in_sample_df, out_sample_df, discoverer)。"""
    dates = window_dates(frames, symbols, last_pos)
    tr, te = split_by_date(dates, cutoff)
    if tr.sum() < 500 or te.sum() < 200:
        raise ValueError(f"切分後樣本太少(訓練{tr.sum()}, 測試{te.sum()}),換 cutoff")

    # 只用訓練期資料分群 → 固定型態
    disc = PatternDiscoverer(k_range=k_range).fit(X[tr])
    labels_tr = disc.labels(X[tr])
    labels_te = disc.labels(X[te])  # 用同一套模型套到測試期

    ins = forward_returns_multi_excess(
        frames, symbols[tr], last_pos[tr], labels_tr, benchmark,
        horizon=horizon, beta=beta)
    outs = forward_returns_multi_excess(
        frames, symbols[te], last_pos[te], labels_te, benchmark,
        horizon=horizon, beta=beta)
    return ins, outs, disc


def run_xs(frames, symbols, last_pos, X, cutoff, horizon=5, k_range=range(4, 13)):
    """橫斷面去均值版的 walk-forward(不需 benchmark)。"""
    dates = window_dates(frames, symbols, last_pos)
    tr, te = split_by_date(dates, cutoff)
    if tr.sum() < 500 or te.sum() < 200:
        raise ValueError(f"切分後樣本太少(訓練{tr.sum()}, 測試{te.sum()})")

    disc = PatternDiscoverer(k_range=k_range).fit(X[tr])
    labels_tr = disc.labels(X[tr])
    labels_te = disc.labels(X[te])

    ins = forward_returns_cross_sectional(
        frames, symbols[tr], last_pos[tr], labels_tr, horizon=horizon)
    outs = forward_returns_cross_sectional(
        frames, symbols[te], last_pos[te], labels_te, horizon=horizon)
    return ins, outs, disc


def compare(ins: pd.DataFrame, outs: pd.DataFrame) -> pd.DataFrame:
    """把樣本內外的關鍵指標並排,方便看哪些群「撐不過去」。"""
    cols = [c for c in ["count", "mean_ret", "signal_t", "breadth"] if c in ins.columns]
    a = ins[cols].add_suffix("_in")
    b = outs[cols].add_suffix("_out")
    cmp = a.join(b, how="outer")
    # 訊號是否延續:同號且樣本外仍達 |t|>1.5 算「延續」
    def verdict(row):
        ti, to = row["signal_t_in"], row["signal_t_out"]
        if np.sign(ti) == np.sign(to) and abs(to) > 1.5 and abs(ti) > 1.5:
            return "延續"
        if abs(ti) > 1.5 and abs(to) < 1.0:
            return "消失"
        return "微弱/不定"
    cmp["判定"] = cmp.apply(verdict, axis=1)
    return cmp.sort_values("signal_t_in")
