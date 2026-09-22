"""盤勢策略回測:用「波動目標」依盤勢調整倉位,和買進持有比較。

邏輯(全程 walk-forward,無未來函數):
  1. 只用訓練期定義盤勢 + 每種盤勢的預期未來波動(來自訓練期)。
  2. 測試期每天照當天盤勢查表得預期波動,用「波動目標法」定倉位:
       倉位 = clip(目標波動 / 該盤勢預期波動, floor, 1.0)
     預期波動越高 → 倉位越低(在高風險盤勢自動減碼)。
  3. 倉位在當天收盤決定,隔天才套用報酬(t 的部位吃 t+1 的報酬)。

只做多、不加槓桿(倉位上限 1.0),所以策略頂多跟大盤同倉,
價值在於「高波時減碼」能不能改善風險調整後報酬(Sharpe / 回撤)。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from kline import regime as rg


def fit_regimes(feat_train: pd.DataFrame, k: int):
    scaler = StandardScaler().fit(feat_train.to_numpy())
    km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(
        scaler.transform(feat_train.to_numpy()))
    return scaler, km


def assign(scaler, km, feat: pd.DataFrame) -> np.ndarray:
    return km.predict(scaler.transform(feat.to_numpy()))


def regime_weights(g_train: pd.DataFrame, floor: float = 0.2) -> dict:
    """把訓練期各盤勢的預期波動轉成目標倉位。

    目標波動設為訓練期最低盤勢的波動(最穩的盤勢 → 滿倉 1.0),
    其餘盤勢依預期波動反比縮小,下限 floor 保持部分持股。
    """
    target = g_train["fwd_vol_mean"].min()
    w = {}
    for reg in g_train.index:
        raw = target / g_train.loc[reg, "fwd_vol_mean"]
        w[reg] = float(np.clip(raw, floor, 1.0))
    return w


def _apply_band(target: pd.Series, band: float) -> pd.Series:
    """不交易帶:目標與現有部位差距 < band 就維持不動,壓低換手。"""
    if band <= 0:
        return target
    held = []
    cur = 0.0
    for t in target.to_numpy():
        if abs(t - cur) >= band:
            cur = t          # 差距夠大才真的調到目標
        held.append(cur)
    return pd.Series(held, index=target.index, name="weight")


def backtest(df: pd.DataFrame, feat_index, labels: np.ndarray,
             weights: dict, fee: float = 0.001425, tax: float = 0.003,
             band: float = 0.0) -> pd.DataFrame:
    """在測試期跑回測,含台股交易成本與不交易帶。

    fee: 手續費率(買賣各收一次)。tax: 證交稅(僅賣出)。
    band: 不交易帶,目標倉位變動小於 band 就不調(降換手)。
    每天倉位變化 = 換手;加碼(買)扣 fee,減碼(賣)扣 fee+tax。
    """
    close = df["Close"]
    daily_ret = close.pct_change()

    target = pd.Series(
        [weights[l] for l in labels], index=feat_index, name="weight")
    pos_series = _apply_band(target, band)   # 套不交易帶後的實際部位

    idx = feat_index
    ret = daily_ret.reindex(idx)
    weight_lag = pos_series.shift(1).fillna(0.0)

    # 換手:今天的目標部位 vs 昨天,拆成買進量與賣出量
    dpos = pos_series.fillna(0.0).diff().fillna(pos_series.fillna(0.0))
    buy = dpos.clip(lower=0)
    sell = (-dpos).clip(lower=0)
    cost = buy * fee + sell * (fee + tax)   # 當天換倉的成本(佔資金比例)

    gross = weight_lag * ret
    strat_ret = gross - cost                 # 成本在調倉當天扣
    out = pd.DataFrame({
        "weight": pos_series,
        "mkt_ret": ret,
        "gross_ret": gross,
        "cost": cost,
        "strat_ret": strat_ret,
    })
    out["equity_strat"] = (1 + out["strat_ret"].fillna(0)).cumprod()
    out["equity_gross"] = (1 + out["gross_ret"].fillna(0)).cumprod()
    out["equity_bh"] = (1 + out["mkt_ret"].fillna(0)).cumprod()
    return out


def turnover_stats(res: pd.DataFrame) -> dict:
    """換手與成本統計,判斷成本吃掉多少。"""
    total_cost = res["cost"].sum()
    n_trades = (res["weight"].diff().abs() > 1e-9).sum()
    ann_turnover = res["weight"].diff().abs().sum() / len(res) * 252
    return {
        "調倉次數": int(n_trades),
        "年化換手": ann_turnover,
        "累計成本": total_cost,
    }


def metrics(returns: pd.Series, periods_per_year: int = 252) -> dict:
    """年化報酬、年化波動、Sharpe、最大回撤。"""
    r = returns.dropna()
    if len(r) == 0:
        return {}
    ann_ret = (1 + r).prod() ** (periods_per_year / len(r)) - 1
    ann_vol = r.std() * np.sqrt(periods_per_year)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else np.nan
    equity = (1 + r).cumprod()
    dd = (equity / equity.cummax() - 1).min()
    return {
        "年化報酬": ann_ret,
        "年化波動": ann_vol,
        "Sharpe": sharpe,
        "最大回撤": dd,
    }
