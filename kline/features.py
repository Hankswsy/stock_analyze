"""特徵工程:把連續 N 根 K 線的視窗轉成尺度不變的特徵向量。

核心概念:一個「K 線組合」的意義來自形狀,而非絕對價位。
所以每個視窗都用視窗內的價格範圍做正規化,讓 2330 在 1000 元
和某小型股在 20 元、只要形狀一樣就落在特徵空間的同一區。
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _window_features(win: pd.DataFrame) -> np.ndarray:
    """把一個 N 根視窗轉成特徵向量。

    正規化基準:整個視窗的最高最低價 → [0,1]。
    每根 K 線輸出 4 個值(正規化後的 O/H/L/C)。
    另外附上每根的成交量(用視窗內 z-score 正規化)。
    """
    o = win["Open"].to_numpy(dtype=float)
    h = win["High"].to_numpy(dtype=float)
    l = win["Low"].to_numpy(dtype=float)
    c = win["Close"].to_numpy(dtype=float)
    v = win["Volume"].to_numpy(dtype=float)

    lo, hi = l.min(), h.max()
    rng = hi - lo
    if rng <= 0:  # 整段完全沒波動,跳過
        return None

    def norm(x):
        return (x - lo) / rng

    price_feat = np.concatenate([norm(o), norm(h), norm(l), norm(c)])

    # 成交量:視窗內 z-score(捕捉量能變化的形狀)
    vmean, vstd = v.mean(), v.std()
    vol_feat = (v - vmean) / vstd if vstd > 0 else np.zeros_like(v)

    return np.concatenate([price_feat, vol_feat])


def build_windows(df: pd.DataFrame, n: int = 5):
    """滑動視窗切出所有連續 N 根組合。

    回傳:
      X      -- (樣本數, 特徵維度) 特徵矩陣
      idx    -- 每個視窗「最後一根」在原始 df 的整數位置(給後續算未來報酬用)
    """
    feats, idx = [], []
    values = df[["Open", "High", "Low", "Close", "Volume"]]
    for start in range(len(df) - n + 1):
        win = values.iloc[start : start + n]
        f = _window_features(win)
        if f is not None:
            feats.append(f)
            idx.append(start + n - 1)  # 視窗最後一根
    return np.asarray(feats), np.asarray(idx)


def build_windows_multi(frames: dict, n: int = 5):
    """多檔混合:把每檔各自切視窗後合併成一個大特徵矩陣。

    正規化在「單一視窗內」做,所以不同股票、不同價位的相同形狀
    會落在特徵空間同一區 —— 這正是多檔混合能成立的前提。

    回傳:
      X       -- (總視窗數, 特徵維度)
      symbols -- 每個視窗來自哪一檔(字串陣列)
      last_pos-- 每個視窗最後一根在「該檔 df」裡的整數位置
    後兩者要成對使用,才能在正確的股票序列上算未來報酬。
    """
    feats, syms, poss = [], [], []
    for sym, df in frames.items():
        X, idx = build_windows(df, n=n)
        if len(X) == 0:
            continue
        feats.append(X)
        syms.extend([sym] * len(X))
        poss.extend(idx.tolist())
    if not feats:
        raise ValueError("沒有任何可用視窗(資料太短?)")
    return np.vstack(feats), np.asarray(syms), np.asarray(poss)


if __name__ == "__main__":
    from data import fetch

    d = fetch("2330.TW", period="1y")
    X, idx = build_windows(d, n=5)
    print(f"特徵矩陣:{X.shape}(視窗數 x 特徵維度)")
