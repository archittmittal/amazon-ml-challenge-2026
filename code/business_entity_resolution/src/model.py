"""Gradient-boosted matcher, cross-fitted by Source-1 entity.

Two interchangeable back-ends, both permissively licensed:
  lgbm  LightGBM (MIT), CPU
  xgb   XGBoost (Apache-2.0), CUDA when a GPU is present: fast enough to train on 100% of
        candidate pairs instead of a 35% sample.

`get(rows)` returns the feature matrix for the given row indices; this lets stage 2 stack
[pair | residue-encoding | context] features lazily instead of materialising a huge matrix.
"""
import shutil
import subprocess

import lightgbm as lgb
import numpy as np

LGB_PARAMS = dict(objective='binary', learning_rate=0.08, num_leaves=255, min_data_in_leaf=200,
                  feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=2.0,
                  max_bin=255, verbose=-1, seed=7)
XGB_PARAMS = dict(objective='binary:logistic', eval_metric='logloss', tree_method='hist',
                  eta=0.06, max_depth=0, max_leaves=255, grow_policy='lossguide',
                  min_child_weight=20, subsample=0.8, colsample_bytree=0.8, reg_lambda=2.0,
                  max_bin=256, seed=7)


def has_gpu():
    if not shutil.which('nvidia-smi'):
        return False
    try:
        return subprocess.run(['nvidia-smi', '-L'], capture_output=True, timeout=20).returncode == 0
    except Exception:
        return False


def resolve_backend(requested):
    if requested == 'lgbm':
        return 'lgbm'
    try:
        import xgboost  # noqa: F401
    except Exception:
        return 'lgbm'
    if requested == 'xgb':
        return 'xgb'
    return 'xgb' if has_gpu() else 'lgbm'          # auto


def _batches(xgb, get, rows, y, batch=1_000_000):
    class Batches(xgb.DataIter):
        def __init__(self):
            self.i = 0
            super().__init__()

        def next(self, input_data):
            if self.i >= len(rows):
                return False
            r = rows[self.i:self.i + batch]
            input_data(data=get(r), label=y[r])
            self.i += batch
            return True

        def reset(self):
            self.i = 0

    return Batches()


def fit(get, y, rows, qhash, names, threads, rounds, log, backend='lgbm'):
    """Train on `rows`; 4% of queries (by hash) are held out for early stopping."""
    es = (qhash[rows] % 25) == 0
    tr, va = rows[~es], rows[es]
    if backend == 'xgb':
        import xgboost as xgb
        params = dict(XGB_PARAMS, nthread=threads, device='cuda' if has_gpu() else 'cpu')
        # Stream the training rows in batches: the full float32 matrix (17M rows x ~95 features,
        # ~6.5GB, plus hstack temporaries) never exists in RAM; XGBoost only keeps the quantised
        # 1-byte-per-value copy.
        dtr = xgb.QuantileDMatrix(_batches(xgb, get, tr, y), max_bin=256)
        dtr.feature_names = names
        dva = xgb.QuantileDMatrix(get(va), label=y[va], feature_names=names, ref=dtr)
        m = xgb.train(params, dtr, num_boost_round=rounds, evals=[(dva, 'valid')],
                      early_stopping_rounds=50, verbose_eval=200)
        score = m.best_score
    else:
        params = dict(LGB_PARAMS, num_threads=threads)
        dtr = lgb.Dataset(get(tr), label=y[tr], feature_name=names, free_raw_data=True)
        dva = lgb.Dataset(get(va), label=y[va], reference=dtr)
        m = lgb.train(params, dtr, num_boost_round=rounds, valid_sets=[dva],
                      callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(200)])
        score = m.best_score['valid_0']['binary_logloss']
    log(f'  model[{backend}]: {len(tr):,} rows ({y[tr].mean():.3f} pos), '
        f'best_iter={m.best_iteration}, valid logloss={score:.5f}')
    return m


def _raw_predict(m, X):
    if isinstance(m, lgb.Booster):
        return m.predict(X, num_iteration=m.best_iteration)
    return m.inplace_predict(X, iteration_range=(0, m.best_iteration + 1))


def predict(m, get, rows, chunk=2_000_000, log=None, tag=''):
    out = np.empty(len(rows), dtype=np.float32)
    for i in range(0, len(rows), chunk):
        out[i:i + chunk] = _raw_predict(m, get(rows[i:i + chunk]))
        if log is not None:
            log(f'  {tag}predict {min(i + chunk, len(rows)):,}/{len(rows):,}')
    return out


def crossfit(get, n, y, fold, sample_mask, qhash, names, threads, rounds, log, backend='lgbm'):
    """2-fold by Source-1 entity. Returns OOF predictions and both fold models."""
    oof = np.zeros(n, dtype=np.float32)
    models = []
    for f in (0, 1):
        m = fit(get, y, np.flatnonzero((fold != f) & sample_mask), qhash, names, threads, rounds,
                log, backend)
        rows = np.flatnonzero(fold == f)
        oof[rows] = predict(m, get, rows, log=log, tag=f'fold{f} OOF ')
        models.append(m)
    return oof, models


def predict_avg(models, get, n, log=None):
    rows = np.arange(n)
    preds = [predict(m, get, rows, log=log, tag=f'model{i} ') for i, m in enumerate(models)]
    return np.mean(preds, axis=0).astype(np.float32)


def importance(m, names):
    if isinstance(m, lgb.Booster):
        return dict(zip(names, m.feature_importance('gain')))
    return m.get_score(importance_type='gain')


def save(m, path_stem):
    m.save_model(path_stem + ('.txt' if isinstance(m, lgb.Booster) else '.json'))


def load_model(path):
    if path.endswith('.txt'):
        return lgb.Booster(model_file=path)       # saved at best_iteration; predicts with all trees
    import xgboost as xgb
    m = xgb.Booster(model_file=path)             # best_iteration is restored from model attributes
    m.set_param({'device': 'cuda' if has_gpu() else 'cpu'})
    return m
