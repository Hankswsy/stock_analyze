"""深挖單一型態群:把某個群拆開看清楚它是什麼。

回答四個問題:
  1. 這型態的「平均形狀」長怎樣(翻成人看得懂的描述)
  2. 在哪些股票上最有效
  3. 在哪些年份最有效
  4. 出現時間是否集中在特定月份(側面看是不是季節/財報效應)
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def average_shape(X: np.ndarray, labels: np.ndarray, cluster: int,
                  n_window: int) -> dict:
    """算該群所有成員的平均正規化 K 線,翻成形狀描述。

    特徵前 4*n 維是 norm(O),norm(H),norm(L),norm(C) 各 n 個。
    """
    members = X[labels == cluster]
    mean_feat = members.mean(axis=0)
    n = n_window
    O = mean_feat[0:n]
    H = mean_feat[n:2 * n]
    L = mean_feat[2 * n:3 * n]
    C = mean_feat[3 * n:4 * n]

    bars = []
    for i in range(n):
        body = C[i] - O[i]                       # 正=紅(收>開),負=綠
        upper = H[i] - max(O[i], C[i])           # 上影線
        lower = min(O[i], C[i]) - L[i]           # 下影線
        bars.append({
            "bar": i + 1,
            "方向": "紅(漲)" if body >= 0 else "綠(跌)",
            "實體": round(abs(body), 3),
            "上影線": round(upper, 3),
            "下影線": round(lower, 3),
        })
    trend = C[-1] - C[0]                          # 整段趨勢(正規化尺度)
    return {"bars": bars, "trend": round(trend, 3), "O": O, "H": H, "L": L, "C": C}


def per_symbol(res_cluster: pd.DataFrame) -> pd.DataFrame:
    """該群在每檔股票上的超額報酬統計(依平均排序)。"""
    g = res_cluster.groupby("symbol")["ret"].agg(
        count="count", mean="mean",
        win_rate=lambda s: (s > 0).mean(), std="std",
    )
    g["t"] = g["mean"] / g["std"] * np.sqrt(g["count"])
    return g.sort_values("mean")


def per_year(res_cluster: pd.DataFrame) -> pd.DataFrame:
    """該群在每一年的超額報酬統計。"""
    g = res_cluster.groupby("year")["ret"].agg(
        count="count", mean="mean",
        win_rate=lambda s: (s > 0).mean(),
    )
    return g


def per_month(res_cluster: pd.DataFrame) -> pd.DataFrame:
    """出現次數的月份分布(看是否集中在特定月份)。"""
    return res_cluster.groupby("month").size().rename("出現次數")


def build_cluster_records(frames, symbols, last_pos, labels, benchmark,
                          cluster, horizon=5, beta=1.0) -> pd.DataFrame:
    """組出該群每一筆出現的明細:日期、年、月、超額報酬。"""
    from kline.analyze import _align_benchmark

    closes = {s: df["Close"].to_numpy(dtype=float) for s, df in frames.items()}
    dates = {s: df.index for s, df in frames.items()}
    bench = _align_benchmark(frames, benchmark)

    rows = []
    for sym, pos, lab in zip(symbols, last_pos, labels):
        if lab != cluster:
            continue
        c = closes[sym]
        b = bench[sym]
        fut = pos + horizon
        if fut < len(c) and not (np.isnan(b[pos]) or np.isnan(b[fut])):
            excess = (c[fut] / c[pos] - 1.0) - beta * (b[fut] / b[pos] - 1.0)
            d = dates[sym][pos]
            rows.append((sym, d, d.year, d.month, excess))
    return pd.DataFrame(rows, columns=["symbol", "date", "year", "month", "ret"])
