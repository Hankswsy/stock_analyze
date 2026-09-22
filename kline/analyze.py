"""後分析:每個型態群「出現後的未來報酬」統計。

這一步把無監督結果變得有用 —— 哪些組合出現後,接下來
h 根 K 線的報酬顯著偏正 / 偏負?這才是型態有沒有交易價值的判準。
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def forward_returns(df: pd.DataFrame, idx: np.ndarray, labels: np.ndarray,
                    horizon: int = 5) -> pd.DataFrame:
    """算每個視窗在 horizon 根之後的報酬,並依群彙整。

    idx: 每個視窗最後一根的整數位置。
    報酬 = close[i+horizon] / close[i] - 1。
    """
    close = df["Close"].to_numpy(dtype=float)
    rows = []
    for pos, lab in zip(idx, labels):
        fut = pos + horizon
        if fut < len(close):
            ret = close[fut] / close[pos] - 1.0
            rows.append((lab, ret))

    res = pd.DataFrame(rows, columns=["cluster", "fwd_ret"])
    summary = res.groupby("cluster")["fwd_ret"].agg(
        count="count",
        mean_ret="mean",
        median_ret="median",
        win_rate=lambda s: (s > 0).mean(),
        std="std",
    )
    # 用 mean/std*sqrt(n) 當簡單的訊號強度指標(類 t 值)
    summary["signal_t"] = summary["mean_ret"] / summary["std"] * np.sqrt(summary["count"])
    return summary.sort_values("mean_ret", ascending=False)


def forward_returns_multi(frames: dict, symbols: np.ndarray, last_pos: np.ndarray,
                          labels: np.ndarray, horizon: int = 5) -> pd.DataFrame:
    """多檔版:在各自股票序列上算未來報酬,再依群彙整。

    除了整體統計,多算兩個「普遍性」欄位:
      n_syms   -- 這個群涵蓋幾檔股票(越多越通用)
      breadth  -- 各檔平均報酬>0 的比例(型態在多少比例的股票上有效)
    breadth 高才代表這是跨股票的真型態,而非單檔巧合。
    """
    closes = {s: df["Close"].to_numpy(dtype=float) for s, df in frames.items()}
    rows = []
    for sym, pos, lab in zip(symbols, last_pos, labels):
        c = closes[sym]
        fut = pos + horizon
        if fut < len(c):
            rows.append((lab, sym, c[fut] / c[pos] - 1.0))

    res = pd.DataFrame(rows, columns=["cluster", "symbol", "fwd_ret"])

    # 先算每群每檔的平均,用來衡量 breadth
    per = res.groupby(["cluster", "symbol"])["fwd_ret"].mean().reset_index()
    breadth = per.groupby("cluster").agg(
        n_syms=("symbol", "nunique"),
        breadth=("fwd_ret", lambda s: (s > 0).mean()),
    )

    summary = res.groupby("cluster")["fwd_ret"].agg(
        count="count",
        mean_ret="mean",
        median_ret="median",
        win_rate=lambda s: (s > 0).mean(),
        std="std",
    )
    summary["signal_t"] = summary["mean_ret"] / summary["std"] * np.sqrt(summary["count"])
    summary = summary.join(breadth)
    return summary.sort_values("signal_t", ascending=False)


def forward_returns_cross_sectional(frames: dict, symbols: np.ndarray,
                                    last_pos: np.ndarray, labels: np.ndarray,
                                    horizon: int = 5) -> pd.DataFrame:
    """橫斷面去均值:每筆報酬減掉「同一天所有股票的平均報酬」。

    比「減大盤指數」更乾淨 —— 不受指數成分偏差(如 0050 被台積電主導)影響,
    等於拿每個型態去跟「當天同儕的等權平均」比。若某群仍顯著,才是真的
    形狀帶來的相對強弱,而非整體漂移。
    """
    dates = {s: df.index for s, df in frames.items()}
    closes = {s: df["Close"].to_numpy(dtype=float) for s, df in frames.items()}

    recs = []
    for sym, pos, lab in zip(symbols, last_pos, labels):
        c = closes[sym]
        fut = pos + horizon
        if fut < len(c):
            recs.append((dates[sym][pos], lab, c[fut] / c[pos] - 1.0))
    r = pd.DataFrame(recs, columns=["date", "cluster", "raw"])

    # 每一天的等權平均報酬(全股池橫斷面)
    day_mean = r.groupby("date")["raw"].transform("mean")
    r["xs"] = r["raw"] - day_mean   # 去均值後的相對報酬

    per = r.groupby(["cluster"])["xs"]
    summary = per.agg(
        count="count", mean_ret="mean", median_ret="median",
        win_rate=lambda s: (s > 0).mean(), std="std",
    )
    summary["signal_t"] = summary["mean_ret"] / summary["std"] * np.sqrt(summary["count"])
    return summary.sort_values("signal_t", ascending=False)


def _align_benchmark(frames: dict, benchmark: pd.DataFrame) -> dict:
    """把大盤(如 0050)的收盤價對齊到每檔股票的日期軸。

    為什麼要對齊:last_pos 是股票 df 內的整數位置,但各檔停牌日不同,
    整數位置不能直接拿去索引大盤。用 reindex 依日期對齊後,
    大盤陣列就和該股票 df 一樣長、可用同一個整數位置索引。
    ffill 補上大盤剛好沒開盤那天的值。
    """
    bench_close = benchmark["Close"]
    aligned = {}
    for sym, df in frames.items():
        aligned[sym] = bench_close.reindex(df.index).ffill().to_numpy(dtype=float)
    return aligned


def forward_returns_multi_excess(frames: dict, symbols: np.ndarray,
                                 last_pos: np.ndarray, labels: np.ndarray,
                                 benchmark: pd.DataFrame, horizon: int = 5,
                                 beta: float = 1.0) -> pd.DataFrame:
    """去市場化版:未來報酬扣掉同期大盤報酬,只留型態自己的超額報酬。

    excess = 個股報酬 - beta × 大盤同期報酬
    beta=1 是最單純的市場中性假設;權值股實際 beta 常 >1,
    要更精準可先估各檔 beta 再帶入(見 --beta 說明)。
    """
    closes = {s: df["Close"].to_numpy(dtype=float) for s, df in frames.items()}
    bench = _align_benchmark(frames, benchmark)

    rows = []
    for sym, pos, lab in zip(symbols, last_pos, labels):
        c = closes[sym]
        b = bench[sym]
        fut = pos + horizon
        if fut < len(c) and not (np.isnan(b[pos]) or np.isnan(b[fut])):
            stock_ret = c[fut] / c[pos] - 1.0
            mkt_ret = b[fut] / b[pos] - 1.0
            rows.append((lab, sym, stock_ret - beta * mkt_ret))

    res = pd.DataFrame(rows, columns=["cluster", "symbol", "excess_ret"])

    per = res.groupby(["cluster", "symbol"])["excess_ret"].mean().reset_index()
    breadth = per.groupby("cluster").agg(
        n_syms=("symbol", "nunique"),
        breadth=("excess_ret", lambda s: (s > 0).mean()),
    )

    summary = res.groupby("cluster")["excess_ret"].agg(
        count="count",
        mean_ret="mean",
        median_ret="median",
        win_rate=lambda s: (s > 0).mean(),
        std="std",
    )
    summary["signal_t"] = summary["mean_ret"] / summary["std"] * np.sqrt(summary["count"])
    summary = summary.join(breadth)
    return summary.sort_values("signal_t", ascending=False)
