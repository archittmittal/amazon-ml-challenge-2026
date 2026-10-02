"""Sibling-entity discrimination.

Test-set diagnostics showed the dominant error is not noise on a true match but a *sibling*:
a different business with the same base name plus one extra word ("Applied Retail Systems
Holdings", "... Public Limited") at a nearby house number on the same street (13565 vs 13558,
C-316 vs C-303). Two signals separate siblings from genuinely corrupted copies:

1. Number agreement across the candidate set (label-free). Corruption of a house number is
   independent per record, so two sources rarely share the same *wrong* number; a sibling's
   records all share *their* number. For each (record q, entity s) we count how many other
   candidates of s carry q's number signature, and whether any candidate matches s's own.

2. Extra-word semantics (supervised, out-of-fold). For each word present in one name but not
   the other we learn, from training labels, how often pairs with that residue are true
   matches: 'llc' / 'the' residue -> noise; 'holdings' / 'public' / 'traders' -> sibling.
   Encodings for fold f come only from the other fold's S1 entities (no leakage); test uses
   all of train. Unseen words (e.g. new French words) fall back to the prior.
"""
import gc

import numpy as np
import polars as pl

from features import take
from memguard import next_chunk

TE_FEATURES = ['te_x_min', 'te_x_mean', 'te_x_n', 'te_x_unseen', 'te_m_min', 'te_m_mean', 'te_m_n']


def _hash_or_null(col):
    return pl.when(pl.col(col).fill_null('') != '').then(pl.col(col).hash(5)).otherwise(None)


def agreement_features(cand, rec):
    """Memory-lean: works on a slim (ids + hashes) frame and evaluates one window aggregation at a
    time. Evaluating all windows in one with_columns over the full candidate table built several
    30M-row hash tables concurrently and peaked at ~29GB on the full training set."""
    sig = rec.select(
        (pl.col('a_nums').fill_null('').str.split(' ')
           .list.eval(pl.element().filter(pl.element() != ''))
           .list.unique().list.sort().list.join(' ')).alias('sig'),
        pl.col('a_nums').fill_null('').str.split(' ').list.first().alias('p'),
        pl.col('n_core').fill_null('').alias('nc'),
    ).select(_hash_or_null('sig').alias('sig_h'), _hash_or_null('p').alias('p_h'),
             _hash_or_null('nc').alias('nc_h'))
    # exact core-name frequency among S1 records of the same country (label-free)
    freq = (rec.filter(pl.col('src') == 1).group_by('country', 'n_core').len()
               .rename({'len': 's1freq'}))
    nf = (rec.select('rid', 'country', 'n_core').join(freq, on=['country', 'n_core'], how='left')
             .sort('rid')['s1freq'].fill_null(0).cast(pl.Float32))
    sig = sig.with_columns(nf.alias('s1freq'))
    del freq, nf
    q = take(sig, cand['rid2'].to_numpy().astype(np.int64))
    s = take(sig, cand['rid1'].to_numpy().astype(np.int64))
    del sig
    freq_cols = [q['s1freq'].alias('ag_q_s1freq'), s['s1freq'].alias('ag_s_s1freq')]
    slim = pl.DataFrame({'rid1': cand['rid1'], 'rid2': cand['rid2'],
                         'qsig_h': q['sig_h'], 'ssig_h': s['sig_h'],
                         '_qp': q['p_h'], '_sp': s['p_h'], '_qnc': q['nc_h']})
    del q, s
    slim = slim.with_columns((pl.col('qsig_h') == pl.col('ssig_h')).fill_null(False).alias('_se'),
                             (pl.col('_qp') == pl.col('_sp')).fill_null(False).alias('_pe'))
    exprs = {
        'ag_s_sig_eq': pl.col('_se').sum().over('rid1'),
        'ag_q_sig_cluster': pl.when(pl.col('qsig_h').is_not_null()).then(pl.len().over('rid1', 'qsig_h')),
        'ag_s_p_eq': pl.col('_pe').sum().over('rid1'),
        'ag_q_p_cluster': pl.when(pl.col('_qp').is_not_null()).then(pl.len().over('rid1', '_qp')),
        'ag_n_cluster': pl.when(pl.col('_qnc').is_not_null()).then(pl.len().over('rid1', '_qnc')),
        'ag_q_nsig_match': pl.col('_se').sum().over('rid2'),
    }
    new = {}
    for name, e in exprs.items():                       # one hash table alive at a time
        new[name] = slim.select(e.cast(pl.Float32).alias(name)).to_series()
        gc.collect()
    se, pe = slim['_se'], slim['_pe']
    new['ag_sib_sig'] = (~se & (new['ag_q_sig_cluster'] >= 2)).cast(pl.Float32).alias('ag_sib_sig')
    new['ag_sib_p'] = (~pe & (new['ag_q_p_cluster'] >= 2)).cast(pl.Float32).alias('ag_sib_p')
    out = cand.with_columns(slim['qsig_h'], slim['ssig_h'], *new.values(), *freq_cols)
    del slim, new
    gc.collect()
    return out


# ------------------------------------------------------------------------------------------
# out-of-fold target encoding of name residue words
# ------------------------------------------------------------------------------------------
def _residue_tokens(rec):
    return rec.select(pl.col('n_full').fill_null('').str.split(' ')
                        .list.eval(pl.element().filter(pl.element() != '')).list.unique().alias('t'))


def _iter_residues(cand, toks, chunk):
    """Yields exploded (row, side, word-hash) frames one pair-chunk at a time, so memory is bounded
    by the chunk and not by the ~100M+ residue rows of the full candidate set."""
    i = 0
    while i < cand.height:
        chunk = next_chunk(chunk, 250_000, None, 'residues')
        P = cand.slice(i, chunk)
        q = take(toks, P['rid2'].to_numpy().astype(np.int64))['t']
        s = take(toks, P['rid1'].to_numpy().astype(np.int64))['t']
        d = pl.DataFrame({'x': q.list.set_difference(s), 'm': s.list.set_difference(q)}) \
              .with_row_index('row', offset=i)
        i += chunk
        yield pl.concat([d.select('row', pl.col(col).alias('w')).explode('w').drop_nulls('w')
                           .select('row', pl.lit(side, dtype=pl.UInt8).alias('side'),
                                   pl.col('w').hash(3).alias('w'))
                         for side, col in ((0, 'x'), (1, 'm'))])


def _new_out(n):
    out = {k: np.full(n, np.nan, dtype=np.float32) for k in TE_FEATURES}
    for k in ('te_x_n', 'te_x_unseen', 'te_m_n'):
        out[k][:] = 0
    return out


def _aggregate_into(out, j, prior, alpha):
    j = j.with_columns(pl.col('pos').fill_null(0.0), pl.col('cnt').fill_null(0.0))
    pr = pl.when(pl.col('side') == 0).then(prior[0]).otherwise(prior[1])
    j = j.with_columns(((pl.col('pos') + alpha * pr) / (pl.col('cnt') + alpha)).alias('r'))
    g = j.group_by('row', 'side').agg(pl.col('r').min().alias('rmin'), pl.col('r').mean().alias('rmean'),
                                      pl.len().alias('n'), (pl.col('cnt') < 5).sum().alias('unseen'))
    for side, pre in ((0, 'te_x_'), (1, 'te_m_')):
        gs = g.filter(pl.col('side') == side)
        r = gs['row'].to_numpy().astype(np.int64)
        out[pre + 'min'][r] = gs['rmin'].to_numpy()
        out[pre + 'mean'][r] = gs['rmean'].to_numpy()
        out[pre + 'n'][r] = gs['n'].to_numpy()
        if side == 0:
            out['te_x_unseen'][r] = gs['unseen'].to_numpy()


def target_encode_train(cand, rec, y, fold, chunk, alpha=20.0, log=print):
    """Two streaming passes: (1) accumulate per-(fold, side, word) counts chunk by chunk,
    (2) encode each chunk with the *other* fold's counts. Peak memory is one chunk plus the
    (few-million-row) statistics table."""
    toks = _residue_tokens(rec)
    acc, pos_sum, cnt_sum = None, [0.0, 0.0], [0, 0]
    for e in _iter_residues(cand, toks, chunk):
        rows = e['row'].to_numpy().astype(np.int64)
        e = e.with_columns(pl.Series('y', y[rows].astype(np.float64)), pl.Series('f', fold[rows].astype(np.int8)))
        for side in (0, 1):
            es = e.filter(pl.col('side') == side)
            pos_sum[side] += float(es['y'].sum())
            cnt_sum[side] += es.height
        part = e.group_by('f', 'side', 'w').agg(pl.col('y').sum().alias('pos'), pl.len().cast(pl.Float64).alias('cnt'))
        acc = part if acc is None else (pl.concat([acc, part]).group_by('f', 'side', 'w')
                                          .agg(pl.col('pos').sum(), pl.col('cnt').sum()))
        del e, part
    prior = {side: (pos_sum[side] / cnt_sum[side]) if cnt_sum[side] else 0.5 for side in (0, 1)}
    other = acc.with_columns((1 - pl.col('f')).cast(pl.Int8).alias('f'))      # fold g stats -> rows of fold 1-g
    out = _new_out(cand.height)
    for e in _iter_residues(cand, toks, chunk):
        rows = e['row'].to_numpy().astype(np.int64)
        e = e.with_columns(pl.Series('f', fold[rows].astype(np.int8)))
        _aggregate_into(out, e.join(other, on=['f', 'side', 'w'], how='left'), prior, alpha)
    full = acc.group_by('side', 'w').agg(pl.col('pos').sum(), pl.col('cnt').sum())
    del acc, other
    gc.collect()
    log(f'  residue-word encoding: {full.height:,} distinct residue words, '
        f'prior P(match|extra word)={prior[0]:.3f}, P(match|missing word)={prior[1]:.3f}')
    return np.column_stack([out[k] for k in TE_FEATURES]), {'stats': full, 'prior': prior, 'alpha': alpha}


def target_encode_apply(cand, rec, enc, chunk):
    toks = _residue_tokens(rec)
    out = _new_out(cand.height)
    for e in _iter_residues(cand, toks, chunk):
        _aggregate_into(out, e.join(enc['stats'], on=['side', 'w'], how='left'), enc['prior'], enc['alpha'])
    return np.column_stack([out[k] for k in TE_FEATURES])
