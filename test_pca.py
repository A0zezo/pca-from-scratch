import numpy as np
from sklearn.decomposition import PCA as SkPCA
from pca import PCA

X = np.random.default_rng(0).normal(size=(100, 5)) @ np.random.default_rng(1).normal(size=(5, 5))


def test_matches_sklearn_variance():
    assert np.allclose(PCA(3).fit(X).explained_variance_ratio, SkPCA(3).fit(X).explained_variance_ratio_)


def test_matches_sklearn_projection_up_to_sign():
    assert np.allclose(np.abs(PCA(2).fit(X).transform(X)), np.abs(SkPCA(2).fit(X).transform(X)))


def test_full_rank_reconstruction():
    p = PCA(5).fit(X)
    assert np.allclose(p.inverse_transform(p.transform(X)), X)


def test_components_orthonormal():
    c = PCA(3).fit(X).components
    assert np.allclose(c @ c.T, np.eye(3))
