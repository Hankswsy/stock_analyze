"""監督式方向模型的資料準備(給 XGBoost 教學範例用)。

特徵 = K 線形狀(25 維)+ 盤勢情境(6 維)= 31 維
label = 橫斷面去均值後的未來報酬正負(1=跑贏當天同儕,0=跑輸)

用橫斷面去均值當 label,避開先前發現的指數成分偏差;
問的是「這個形狀+情境,能不能預測該股跑贏當天平均」。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from kline.features import build_windows
from kline.regime import regime_features

CTX_COLS = ["vol_20", "vol_ratio", "trend_20", "trend_strength", "drawdown", "vol_z"]


def build_dataset(frames: dict, window: int = 5, horizon: int = 20,
                  lookback: int = 20):
    """組出監督式訓練資料。

    回傳:
      X      -- (樣本數, 31) 特徵矩陣(numpy)
      y      -- (樣本數,) 0/1 標籤
      dates  -- 每筆的日期(用來 walk-forward 切分)
      names  -- 特徵名稱(給重要度圖用)
    """
    n = window
    shape_names = ([f"O{i}" for i in range(n)] + [f"H{i}" for i in range(n)] +
                   [f"L{i}" for i in range(n)] + [f"C{i}" for i in range(n)] +
                   [f"V{i}" for i in range(n)])
    names = shape_names + CTX_COLS

    feats, labels, dts = [], [], []
    for sym, df in frames.items():
        Xs, idx = build_windows(df, n=n)
        ctx = regime_features(df, lookback=lookback)          # 依日期索引
        close = df["Close"].to_numpy(dtype=float)
        dates_idx = df.index

        for row, p in zip(Xs, idx):
            d = dates_idx[p]
            fut = p + horizon
            if d not in ctx.index or fut >= len(close):
                continue                                       # 暖機期或尾端跳過
            ctx_row = ctx.loc[d, CTX_COLS].to_numpy(dtype=float)
            fwd = close[fut] / close[p] - 1.0
            feats.append(np.concatenate([row, ctx_row]))
            labels.append(fwd)                                 # 先存原始報酬
            dts.append((d, sym))

    X = np.asarray(feats)
    raw = pd.Series([l for l in labels])
    meta = pd.DataFrame(dts, columns=["date", "symbol"])
    meta["raw"] = raw.values

    # 橫斷面去均值:減掉同一天全股池平均,再取正負當 label
    day_mean = meta.groupby("date")["raw"].transform("mean")
    y = (meta["raw"].to_numpy() - day_mean.to_numpy() > 0).astype(int)

    return X, y, meta["date"].to_numpy(), names
