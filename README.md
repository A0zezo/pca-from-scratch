# pca-from-scratch

PCA via SVD in NumPy, checked against scikit-learn on the Iris dataset.

## Run

```bash
pip install -r requirements.txt
python demo.py
pytest
```

## Output (reproduced)

```
$ python3 demo.py
explained variance ratio (mine):    [0.9246 0.0531]
explained variance ratio (sklearn): [0.9246 0.0531]
max |difference| ignoring sign: 3.6854200247127267e-14
reconstruction MSE from 2 components: 0.0253

$ pytest -q
4 passed in 0.60s
```

## Limits

- Components match scikit-learn up to sign. No whitening or sparse support.
