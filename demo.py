import numpy as np
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA as SkPCA
from pca import PCA

X = load_iris().data
mine = PCA(2).fit(X)
ref = SkPCA(2).fit(X)
print("explained variance ratio (mine):   ", np.round(mine.explained_variance_ratio, 4))
print("explained variance ratio (sklearn):", np.round(ref.explained_variance_ratio_, 4))
Z, R = mine.transform(X), ref.transform(X)
print("max |difference| ignoring sign:", float(np.abs(np.abs(Z) - np.abs(R)).max()))
err = np.mean((X - mine.inverse_transform(Z)) ** 2)
print(f"reconstruction MSE from 2 components: {err:.4f}")
