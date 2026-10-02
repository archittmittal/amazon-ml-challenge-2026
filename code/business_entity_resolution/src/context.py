"""Stage-2 collective features built from stage-1 probabilities (p1).

Pairwise matching treats every (S2/S3, S1) pair in isolation. But the problem has
structure that a pairwise model cannot see:

* Exclusivity: an S2/S3 record belongs to at most ONE S1 entity, so candidates for the
  same query compete (margin to the best rival, rank within the query).
* Cluster coherence: all records of one entity resemble *each other*. A record with an
  unrecognisable DBA name ("Viodeltamira") or a hashed-out address can still be pulled in
  when it closely matches the entity's most confident member (its "anchor").

For each pair (q, s) the anchor is the highest-p1 other record whose top choice is s.
We then measure q-vs-anchor similarity. p1 comes from out-of-fold predictions on train
and from the fold-model average on test, so the stage-2 inputs have the same
distribution in both.
"""
import gc

import numpy as np
import polars as pl
from rapidfuzz import fuzz
from rapidfuzz.process import cpdist

from features import take
from memguard import next_chunk

CONTEXT_FEATURES = ['p1', 'p1_qmax', 'p1_qrank', 'p1_margin', 'p1_qsum', 'p1_ssum', 's_nconf',
                    'p1_srank', 's_ncand', 'anc_p1', 'anc_has', 'anc_n_tset', 'anc_k_tset',
                    'anc_a_tset', 'anc_m_tset', 'anc_self_s_n', 'anc2_a_tset',
                    # sibling-aware: how confident are the records that share q's number
                    # signature vs. the records that share s's own number signature
                    'cx_sig_p1sum_other', 'cx_sig_p1max', 'cx_exact_p1max', 'cx_exact_nconf']


def _sim(scorer, a, b, workers):
    return np.asarray(cpdist(a, b, scorer=scorer, workers=workers, dtype=np.float32), dtype=np.float32) / 100.0


def context_features(cand, p1, rec, workers, chunk, log):
    """Window aggregations are evaluated one at a time on a slim frame and moved straight into
    numpy: computing ~10 windows in a single expression over 40M+ pairs built that many hash
    tables at once and could exceed a 30GB machine."""
    p1f = p1.astype(np.float32)
    c = cand.select('rid2', 'rid1', 'qsig_h', 'ssig_h').with_columns(pl.Series('p1', p1f))
    n = c.height
    out = {k: np.full(n, np.nan, dtype=np.float32) for k in CONTEXT_FEATURES}
    out['p1'] = p1f

    def put(name, expr):
        out[name] = c.select(expr.cast(pl.Float32).alias(name)).to_series().to_numpy()
        gc.collect()

    put('p1_qmax', pl.col('p1').max().over('rid2'))
    put('p1_qsum', pl.col('p1').sum().over('rid2'))
    put('p1_ssum', pl.col('p1').sum().over('rid1'))
    put('p1_srank', pl.col('p1').rank('ordinal', descending=True).over('rid1'))
    put('s_ncand', pl.len().over('rid1'))
    put('p1_qrank', pl.col('p1').rank('ordinal', descending=True).over('rid2'))
    c = c.with_columns(pl.Series('p1_qrank', out['p1_qrank']))
    exact = (pl.col('qsig_h') == pl.col('ssig_h')).fill_null(False)
    put('cx_sig_p1sum_other', pl.when(pl.col('qsig_h').is_not_null())
        .then(pl.col('p1').sum().over('rid1', 'qsig_h') - pl.col('p1')))
    put('cx_sig_p1max', pl.when(pl.col('qsig_h').is_not_null()).then(pl.col('p1').max().over('rid1', 'qsig_h')))
    put('cx_exact_p1max', pl.col('p1').filter(exact).max().over('rid1').fill_null(0))
    put('cx_exact_nconf', (exact & (pl.col('p1') > 0.5)).sum().over('rid1'))
    put('s_nconf', ((pl.col('p1_qrank') == 1) & (pl.col('p1') > 0.5)).sum().over('rid1'))
    q2 = c.select(pl.col('p1').filter(pl.col('p1_qrank') == 2).max().over('rid2').fill_null(0)
                  .cast(pl.Float32)).to_series().to_numpy()
    out['p1_margin'] = np.where(out['p1_qrank'] == 1, p1f - q2, p1f - out['p1_qmax']).astype(np.float32)
    del q2

    # anchors: top-3 records (by p1) that chose s as their best S1
    top = (c.filter(pl.col('p1_qrank') == 1).select('rid1', 'rid2', 'p1').sort('p1', descending=True)
            .group_by('rid1', maintain_order=True)
            .agg(pl.col('rid2').head(3).alias('ar'), pl.col('p1').head(3).alias('ap')))
    top = top.with_columns(
        pl.col('ar').list.get(0, null_on_oob=True).alias('a1'),
        pl.col('ar').list.get(1, null_on_oob=True).alias('a2'),
        pl.col('ar').list.get(2, null_on_oob=True).alias('a3'),
        pl.col('ap').list.get(0, null_on_oob=True).alias('p_a1'),
        pl.col('ap').list.get(1, null_on_oob=True).alias('p_a2'),
    ).select('rid1', 'a1', 'a2', 'a3', 'p_a1', 'p_a2')
    c = (c.select('rid2', 'rid1').with_row_index('row')
          .join(top, on='rid1', how='left').sort('row'))
    del top
    gc.collect()
    c = c.select(
        'rid2', 'rid1',
        pl.when(pl.col('a1') != pl.col('rid2')).then(pl.col('a1')).otherwise(pl.col('a2')).alias('anc'),
        pl.when(pl.col('a1') != pl.col('rid2')).then(pl.col('p_a1')).otherwise(pl.col('p_a2')).alias('anc_p1'),
        # second anchor (distinct from q and first anchor)
        pl.when(pl.col('a1') == pl.col('rid2')).then(pl.col('a3'))
          .when(pl.col('a2') == pl.col('rid2')).then(pl.col('a3'))
          .otherwise(pl.col('a2')).alias('anc2'),
    )
    out['anc_p1'] = c['anc_p1'].fill_null(np.nan).cast(pl.Float32).to_numpy()
    has = c['anc'].is_not_null().to_numpy()
    out['anc_has'] = has.astype(np.float32)

    cols = rec.select('n_core', 'n_skel', 'a_words', 'a_nums').with_columns(pl.all().fill_null(''))
    rows = np.flatnonzero(has)
    # explicit int64: polars .to_numpy() on a nullable-typed integer column can silently
    # return float64 even with zero remaining nulls, which breaks use as a row index.
    rid2 = c['rid2'].to_numpy().astype(np.int64)
    rid1 = c['rid1'].to_numpy().astype(np.int64)
    anc = c['anc'].fill_null(0).to_numpy().astype(np.int64)
    i = 0
    while i < len(rows):
        chunk = next_chunk(chunk, 250_000, log, 'context')
        r = rows[i:i + chunk]
        i += chunk
        q = take(cols, rid2[r])
        a = take(cols, anc[r])
        s = take(cols, rid1[r])
        out['anc_n_tset'][r] = _sim(fuzz.token_set_ratio, q['n_core'].to_list(), a['n_core'].to_list(), workers)
        out['anc_k_tset'][r] = _sim(fuzz.token_set_ratio, q['n_skel'].to_list(), a['n_skel'].to_list(), workers)
        out['anc_a_tset'][r] = _sim(fuzz.token_set_ratio, q['a_words'].to_list(), a['a_words'].to_list(), workers)
        out['anc_m_tset'][r] = _sim(fuzz.token_set_ratio, q['a_nums'].to_list(), a['a_nums'].to_list(), workers)
        # how well the anchor itself matches s (is the anchor trustworthy?)
        out['anc_self_s_n'][r] = _sim(fuzz.token_set_ratio, a['n_core'].to_list(), s['n_core'].to_list(), workers)
        log(f'  context anchor-sim {min(i, len(rows)):,}/{len(rows):,}')
    has2 = c['anc2'].is_not_null().to_numpy()
    rows2 = np.flatnonzero(has2)
    anc2 = c['anc2'].fill_null(0).to_numpy().astype(np.int64)
    i = 0
    while i < len(rows2):
        chunk = next_chunk(chunk, 250_000, log, 'context')
        r = rows2[i:i + chunk]
        i += chunk
        q = take(cols, rid2[r])
        a = take(cols, anc2[r])
        out['anc2_a_tset'][r] = _sim(fuzz.token_set_ratio, q['a_words'].to_list(), a['a_words'].to_list(), workers)
        log(f'  context anchor2-sim {min(i, len(rows2)):,}/{len(rows2):,}')
    log(f'  context features: {n:,} pairs, {has.mean():.3f} with anchor')
    return np.column_stack([out[k] for k in CONTEXT_FEATURES])
