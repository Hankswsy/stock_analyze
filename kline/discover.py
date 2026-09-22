"""無監督型態發現:KMeans 分群 + 自動挑群數。

流程:標準化 -> (可選)PCA 降噪 -> KMeans。
用 silhouette 分數在候選群數中挑最佳,避免拍腦袋定 k。
"""
from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


class PatternDiscoverer:
    def __init__(self, k_range=range(4, 13), pca_var: float = 0.95, seed: int = 42):
        self.k_range = k_range
        self.pca_var = pca_var
        self.seed = seed
        self.scaler = StandardScaler()
        self.pca = None
        self.model: KMeans | None = None
        self.best_k: int | None = None

    def fit(self, X: np.ndarray):
        Xs = self.scaler.fit_transform(X)

        # PCA 保留 95% 變異,降噪並加速分群
        self.pca = PCA(n_components=self.pca_var, random_state=self.seed)
        Xp = self.pca.fit_transform(Xs)

        best_score, best_model, best_k = -1.0, None, None
        # silhouette 在大樣本上很慢,超過門檻就抽樣評分
        sample = None
        if len(Xp) > 5000:
            rng = np.random.default_rng(self.seed)
            sample = rng.choice(len(Xp), 5000, replace=False)

        for k in self.k_range:
            km = KMeans(n_clusters=k, n_init=10, random_state=self.seed)
            labels = km.fit_predict(Xp)
            if sample is not None:
                score = silhouette_score(Xp[sample], labels[sample])
            else:
                score = silhouette_score(Xp, labels)
            if score > best_score:
                best_score, best_model, best_k = score, km, k

        self.model = best_model
        self.best_k = best_k
        self.best_score = best_score
        return self

    def labels(self, X: np.ndarray) -> np.ndarray:
        Xp = self.pca.transform(self.scaler.transform(X))
        return self.model.predict(Xp)
