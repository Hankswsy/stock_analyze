"""XGBoost 用法教學範例 —— 監督式 K 線方向分類。

重點在「XGBoost 怎麼用」:sklearn API、關鍵參數、early stopping、
特徵重要度、分類評估(AUC / 準確率)。方向本身我們已知大概率無效,
正好示範「XGBoost 遇到沒訊號的資料會怎樣」——不會無中生有。

用法:
    python main_xgb.py --file symbols.txt --period 8y --cutoff 2023-01-01
"""
from __future__ import annotations

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report
from xgboost import XGBClassifier

from kline.data import fetch_many
from kline.supervised import build_dataset
from main_multi import load_symbols

plt.rcParams["font.sans-serif"] = ["Microsoft JhengHei", "Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", nargs="+")
    ap.add_argument("--file")
    ap.add_argument("--window", type=int, default=5)
    ap.add_argument("--horizon", type=int, default=20)
    ap.add_argument("--period", default="8y")
    ap.add_argument("--cutoff", default="2023-01-01")
    ap.add_argument("--out", default="xgb_importance.png")
    args = ap.parse_args()

    symbols = load_symbols(args)
    print(f"[1/4] 下載 {len(symbols)} 檔 + 組監督式資料 ...")
    frames = fetch_many(symbols, period=args.period)
    X, y, dates, names = build_dataset(frames, window=args.window,
                                       horizon=args.horizon)
    print(f"      樣本 {X.shape[0]} 筆 x {X.shape[1]} 特徵;正例比例 {y.mean():.3f}")

    # ── walk-forward 切分:cutoff 之前訓練,之後測試 ──
    cutoff = pd.Timestamp(args.cutoff)
    tr = dates < cutoff
    te = dates >= cutoff
    Xtr, ytr = X[tr], y[tr]
    Xte, yte = X[te], y[te]

    # 訓練集再切最後 20%(依時間)當驗證集,給 early stopping 用
    order = np.argsort(dates[tr])
    Xtr, ytr = Xtr[order], ytr[order]
    cut = int(len(Xtr) * 0.8)
    Xt, yt = Xtr[:cut], ytr[:cut]
    Xv, yv = Xtr[cut:], ytr[cut:]
    print(f"      訓練 {len(Xt)} / 驗證 {len(Xv)} / 測試 {len(Xte)}")

    print("[2/4] 訓練 XGBoost ...")
    # ── XGBoost 關鍵參數(sklearn 介面)──
    model = XGBClassifier(
        n_estimators=400,        # 最多幾棵樹(配 early stopping,實際會更少)
        max_depth=4,             # 每棵樹深度;越深越容易過擬合
        learning_rate=0.03,      # 學習率;小=穩但需更多樹
        subsample=0.8,           # 每棵樹隨機抽 80% 樣本(防過擬合)
        colsample_bytree=0.8,    # 每棵樹隨機抽 80% 特徵
        min_child_weight=5,      # 葉節點最小樣本權重;大=更保守
        reg_lambda=1.0,          # L2 正則化
        objective="binary:logistic",   # 二元分類,輸出機率
        eval_metric="auc",       # 用 AUC 當驗證指標
        early_stopping_rounds=30,       # 驗證 AUC 連 30 輪沒進步就停
        random_state=42,
        n_jobs=-1,
    )
    # eval_set 提供驗證集;verbose=False 不洗版
    model.fit(Xt, yt, eval_set=[(Xv, yv)], verbose=False)
    print(f"      實際用了 {model.best_iteration + 1} 棵樹(early stopping)")

    print("[3/4] 樣本外評估 ...")
    proba = model.predict_proba(Xte)[:, 1]      # 預測「跑贏」的機率
    pred = (proba >= 0.5).astype(int)
    auc = roc_auc_score(yte, proba)
    acc = accuracy_score(yte, pred)
    print(f"      測試 AUC = {auc:.4f}(0.5=亂猜)")
    print(f"      測試準確率 = {acc:.4f}(基準={max(yte.mean(),1-yte.mean()):.4f})")
    print("\n分類報告:")
    print(classification_report(yte, pred, target_names=["跑輸", "跑贏"], digits=3))

    print("[4/4] 特徵重要度 ...")
    imp = pd.Series(model.feature_importances_, index=names).sort_values()
    top = imp.tail(15)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(top.index, top.values, color="#4477aa")
    ax.set_title(f"XGBoost 特徵重要度 Top15(測試 AUC={auc:.3f})")
    ax.set_xlabel("重要度(gain)")
    fig.tight_layout()
    fig.savefig(args.out, dpi=130)
    plt.close(fig)
    print(f"      重要度圖已存:{args.out}")
    print("\n最重要的 5 個特徵:")
    print(imp.tail(5).iloc[::-1].to_string(float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":
    main()
