"""kline 套件初始化。

順便把標準輸出設成 UTF-8,讓 Windows cp950 主控台也能正常印中文,
不必每次執行前置 PYTHONIOENCODING=utf-8。
"""
import sys

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except Exception:
        pass  # 舊版 Python 或非標準串流則略過,不影響計算
