"""From pair probabilities to per-entity match lists.

1. Exclusivity: each S2/S3 record is kept only for its arg-max S1 candidate.
2. Expected-F0.5 decoding: the leaderboard averages F0.5 *per Source-1 entity*, so a
   global threshold is not the optimal decision rule. For each S1 entity with sorted
   candidate probabilities p_1 >= ... >= p_L we pick the prefix size k that maximises

        E[F0.5 | k] = sum_{a,b} P(TP=a | top-k) P(FN=b | rest) * 1.25a / (0.25(a+b) + k)

   with k = 0 scoring P(no true match at all), i.e. the singleton case. TP and FN are
   Poisson-binomial; both distributions come from an O(L^2) DP, vectorised over entities.
   This trades recall for precision per entity, e.g. it returns an empty list when
   predicting anything would likely cost more than it gains.
"""
import numpy as np
import polars as pl

MAX_L = 16


def exclusive(cand_ids, p, min_p=0.02):
    c = cand_ids.select('rid2', 'rid1').with_columns(pl.Series('p', p))
    c = c.with_columns(pl.col('p').rank('ordinal', descending=True).over('rid2').alias('_r'))
    return c.filter((pl.col('_r') == 1) & (pl.col('p') >= min_p)).drop('_r')


def _expected_f_k(P, beta2=0.25):
    G, L = P.shape
    F = np.zeros((L + 1, G, L + 1), dtype=np.float64)
    B = np.zeros((L + 1, G, L + 1), dtype=np.float64)
    F[0][:, 0] = 1.0
    for k in range(L):
        pk = P[:, k:k + 1]
        F[k + 1] = F[k] * (1 - pk)
        F[k + 1][:, 1:] += F[k][:, :-1] * pk
    B[L][:, 0] = 1.0
    for k in range(L - 1, -1, -1):
        pk = P[:, k:k + 1]
        B[k] = B[k + 1] * (1 - pk)
        B[k][:, 1:] += B[k + 1][:, :-1] * pk
    E = np.zeros((G, L + 1))
    E[:, 0] = B[0][:, 0]
    a = np.arange(L + 1, dtype=np.float64)
    for k in range(1, L + 1):
        W = (1 + beta2) * a[:, None] / (beta2 * (a[:, None] + a[None, :]) + k)
        E[:, k] = np.einsum('ga,gb,ab->g', F[k], B[k], W)
    return E


def select_expected_f(ex, chunk=100_000):
    """ex: exclusive frame (rid2, rid1, p). Returns selected (rid1, rid2)."""
    g = (ex.sort('p', descending=True)
           .group_by('rid1', maintain_order=True)
           .agg(pl.col('rid2').head(MAX_L), pl.col('p').head(MAX_L)))
    rid1 = g['rid1'].to_numpy()
    # polars .to_numpy() on a nullable-typed integer column (list length here) can return
    # float64 even with zero nulls; force int64 explicitly so it's safe to use as an index.
    lens = g['p'].list.len().to_numpy().astype(np.int64)
    flat_p = g['p'].explode().to_numpy().astype(np.float64)
    flat_r = g['rid2'].explode().to_numpy()
    starts = np.concatenate([[0], np.cumsum(lens)[:-1]]).astype(np.int64)
    P = np.zeros((len(g), MAX_L))
    row = np.repeat(np.arange(len(g), dtype=np.int64), lens)
    col = (np.arange(len(flat_p), dtype=np.int64) - np.repeat(starts, lens)).astype(np.int64)
    P[row, col] = flat_p
    K = np.zeros(len(g), dtype=np.int64)
    for i in range(0, len(g), chunk):
        K[i:i + chunk] = _expected_f_k(P[i:i + chunk]).argmax(1)
    keep = col < np.repeat(K, lens)
    return pl.DataFrame({'rid1': np.repeat(rid1, lens)[keep], 'rid2': flat_r[keep]})


def select_threshold(ex, t):
    return ex.filter(pl.col('p') >= t).select('rid1', 'rid2')


# ------------------------------------------------------------------------------------------
# Learned entity decoder
# ------------------------------------------------------------------------------------------
# The leaderboard scores every S1 entity separately, so the costliest mistakes are cluster-SIZE
# decisions (an entity with matches predicted empty, or a true singleton given one false match,
# scores 0). Expected-F decoding picks the size from pair probabilities assuming they are
# calibrated; under the train/test prior shift they are not exactly. A small model sees the whole
# sorted probability profile of an entity (plus its expected-F curve) and learns, out of fold,
# how far the optimal size is from the expected-F choice. It is only used if it beats expected-F
# on out-of-fold data.
N_P, N_E, MAX_DELTA = 10, 11, 3


def _groups(ex):
    g = (ex.sort('p', descending=True).group_by('rid1', maintain_order=True)
           .agg(pl.col('rid2').head(MAX_L), pl.col('p').head(MAX_L)))
    rid1 = g['rid1'].to_numpy()
    lens = g['p'].list.len().to_numpy().astype(np.int64)
    flat_p = g['p'].explode().to_numpy().astype(np.float64)
    flat_r = g['rid2'].explode().to_numpy()
    starts = np.concatenate([[0], np.cumsum(lens)[:-1]]).astype(np.int64)
    row = np.repeat(np.arange(len(g), dtype=np.int64), lens)
    col = (np.arange(len(flat_p), dtype=np.int64) - np.repeat(starts, lens)).astype(np.int64)
    P = np.zeros((len(g), MAX_L))
    P[row, col] = flat_p
    return rid1, lens, flat_r, row, col, P


def _entity_features(P, lens, chunk=100_000):
    E = np.zeros((len(P), MAX_L + 1), dtype=np.float32)
    for i in range(0, len(P), chunk):
        E[i:i + chunk] = _expected_f_k(P[i:i + chunk])
    k_ef = E.argmax(1)
    X = np.column_stack([P[:, :N_P], E[:, :N_E], k_ef, lens, P.sum(1), (P > 0.5).sum(1),
                         (P > 0.2).sum(1), (P > 0.8).sum(1)]).astype(np.float32)
    return X, k_ef


def _select(rid1, lens, flat_r, col, k):
    keep = col < np.repeat(k, lens)
    return pl.DataFrame({'rid1': np.repeat(rid1, lens)[keep], 'rid2': flat_r[keep]})


def fit_entity_decoder(ex, truth, n_s1, log, threads=4, min_gain=2e-4):
    import lightgbm as lgb
    rid1, lens, flat_r, row, col, P = _groups(ex)
    X, k_ef = _entity_features(P, lens)
    G = len(rid1)
    # oracle: best prefix size along our own ranking, given the true labels
    flat = pl.DataFrame({'rid1': np.repeat(rid1, lens), 'rid2': flat_r}).with_row_index('i')
    hit = flat.join(truth.with_columns(pl.lit(True).alias('t')), on=['rid1', 'rid2'], how='left') \
              .sort('i')['t'].fill_null(False).to_numpy()
    Y = np.zeros((G, MAX_L))
    Y[row, col] = hit
    nt = pl.DataFrame({'rid1': rid1}).join(truth.group_by('rid1').len(), on='rid1', how='left') \
           ['len'].fill_null(0).to_numpy().astype(np.float64)
    tp = np.cumsum(Y, axis=1)
    k = np.arange(1, MAX_L + 1)
    F = np.zeros((G, MAX_L + 1))
    F[:, 0] = (nt == 0)
    F[:, 1:] = np.where(nt[:, None] > 0, 1.25 * tp / (0.25 * nt[:, None] + k[None, :]), 0.0)
    k_or = F.argmax(1)
    y = (np.clip(k_or - k_ef, -MAX_DELTA, MAX_DELTA) + MAX_DELTA).astype(np.int32)
    # short, regularised boosting: some size-correction classes are very rare (<0.1% of entities)
    # and long multiclass training drove them to unstable extremes (in-sample 0.924 vs OOF 0.966
    # on a synthetic check); 60 rounds / 31 leaves / L2=5 gave fresh-data score == OOF score.
    params = dict(objective='multiclass', num_class=2 * MAX_DELTA + 1, learning_rate=0.1,
                  num_leaves=31, min_data_in_leaf=100, feature_fraction=0.9, lambda_l2=5.0,
                  verbose=-1, num_threads=threads, seed=11)
    rounds = 60
    fold = (pl.Series(rid1).hash(41) % 2).to_numpy()
    k_oof = k_ef.copy()
    for f in (0, 1):
        tr, te = fold != f, fold == f
        m = lgb.train(params, lgb.Dataset(X[tr], label=y[tr]), num_boost_round=rounds)
        d = m.predict(X[te]).argmax(1) - MAX_DELTA
        k_oof[te] = np.clip(k_ef[te] + d, 0, lens[te])
    idx = np.arange(G)
    f_ef, f_lr = F[idx, k_ef].mean(), F[idx, k_oof].mean()
    gain = (f_lr - f_ef) * G / max(n_s1, 1)          # in macro-F0.5 units over all S1 entities
    changed = (k_oof != k_ef).mean()
    log(f'  learned entity decoder: OOF macro-F0.5 change {gain:+.5f} '
        f'({changed:.3%} of entities resized; oracle upper bound {(F[idx, k_or].mean() - f_ef) * G / n_s1:+.5f})')
    if gain < min_gain:
        return None, gain, None
    final = lgb.train(params, lgb.Dataset(X, label=y), num_boost_round=rounds)
    return final, gain, _select(rid1, lens, flat_r, col, k_oof)


def select_learned(ex, model):
    rid1, lens, flat_r, row, col, P = _groups(ex)
    X, k_ef = _entity_features(P, lens)
    d = model.predict(X).argmax(1) - MAX_DELTA
    return _select(rid1, lens, flat_r, col, np.clip(k_ef + d, 0, lens))
