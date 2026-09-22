"""資料抓取:用 yfinance 下載 OHLCV 歷史 K 線。"""
from __future__ import annotations

import pandas as pd
import yfinance as yf


def fetch(symbol: str, period: str = "5y", interval: str = "1d") -> pd.DataFrame:
    """下載單一標的的 OHLCV。

    symbol: 台股用 '2330.TW'、'0050.TW';美股直接 'AAPL'。
    period: '1y' / '2y' / '5y' / 'max' ...
    interval: '1d' / '1h' / '1wk' ...
    回傳欄位:Open, High, Low, Close, Volume(去除有缺值的列)。
    """
    df = yf.download(
        symbol,
        period=period,
        interval=interval,
        auto_adjust=True,
        progress=False,
    )
    if df.empty:
        raise ValueError(f"抓不到資料:{symbol}(檢查代碼,台股要加 .TW)")

    # yfinance 新版可能回傳 MultiIndex 欄位,壓平成單層
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
    df.index.name = "Date"
    return df


def fetch_many(symbols: list[str], period: str = "5y",
               interval: str = "1d") -> dict[str, pd.DataFrame]:
    """一次抓多檔,回傳 {symbol: df}。抓不到的檔會跳過並印出警告。"""
    out: dict[str, pd.DataFrame] = {}
    for s in symbols:
        try:
            out[s] = fetch(s, period=period, interval=interval)
        except Exception as e:  # 單檔失敗不該中斷整批
            print(f"  [略過] {s}: {e}")
    if not out:
        raise ValueError("所有標的都抓不到資料")
    return out


if __name__ == "__main__":
    d = fetch("2330.TW", period="1y")
    print(d.tail())
    print(f"共 {len(d)} 根 K 線")
