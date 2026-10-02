"""Pairwise features, fully vectorised (rapidfuzz.cpdist in C++ threads + polars set ops).

Nothing here depends on the country value, so France is scored with exactly the same
function as the training countries.
"""
import gc

import numpy as np
import polars as pl
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler, Levenshtein
from rapidfuzz.process import cpdist

from memguard import next_chunk
from text import LEGAL_FORMS

REC_COLS = ['n_full', 'n_core', 'n_skel', 'a_words', 'a_nums', 'a_codes', 'indic', 'domain', 'src',
            'n_core_w', 'a_words_w']
BLOCK_COLS = ['bscore', 'nkeys', 'brank', 'q_bmax', 'q_ncand', 's_nq', 's_nq_top1', 'b_ratio', 'b_margin',
              'b_sim', 'b_extra']
# candidate-set agreement features (label-free; computed in sibling.agreement_features)
AGREE_FEATURES = ['ag_s_sig_eq', 'ag_q_sig_cluster', 'ag_sib_sig', 'ag_s_p_eq', 'ag_q_p_cluster',
                  'ag_sib_p', 'ag_n_cluster', 'ag_q_nsig_match',
                  # how many S1 entities (same country) carry exactly this core name: a no-address
                  # record named like exactly one S1 is strong evidence; like 40 S1s, none at all
                  'ag_q_s1freq', 'ag_s_s1freq']
_LEGAL_LIST = sorted(LEGAL_FORMS)

PAIR_FEATURES = BLOCK_COLS + AGREE_FEATURES + [
    # name
    'n_ratio', 'n_tsort', 'n_tset', 'n_partial', 'n_jw', 'k_ratio', 'k_tset', 'ns_ratio', 'ns_partial',
    'f_tset', 'n_jacc', 'n_wjacc', 'n_inter', 'k_jacc', 'q_ntok', 's_ntok', 'initials',
    'q_indic', 'q_domain', 'q_src',
    # address
    'a_tset', 'a_tsort', 'a_partial', 'a_ratio', 'a_jacc', 'a_wjacc', 'a_inter', 'q_naw', 's_naw',
    'm_ratio', 'm_inter', 'm_jacc', 'm_first_eq', 'm_maxlen', 'q_nnum', 's_nnum', 'q_addr_empty',
    # joint
    'all_tset',
    # sibling discrimination: house-number relation, composite codes, name differences
    'num_sig_eq', 'num_first_contains', 'num_first_edit', 'num_first_logdiff',
    'code_inter', 'code_jacc', 'name_extra_n', 'name_missing_n', 'name_extra_idf',
    'name_missing_idf', 'legal_conflict', 'legal_eq',
]


def add_idf_weights(rec):
    """Per-record sum of token idf (computed on the split's own corpus: unsupervised)."""
    idfs = {}
    n = rec.height
    for col in ('n_core', 'a_words'):
        # per-record de-duplication before exploding (no global unique over ~60M rows), and the
        # weights are scattered into an array: joining + re-sorting `rec` copied the whole
        # 12M-row table twice and was the ~5GB memory spike on the full training set.
        t = (rec.select('rid', pl.col(col).str.split(' ').list.unique().alias('t')).explode('t')
                .filter(pl.col('t').is_not_null() & (pl.col('t') != '')))
        idf = (t.group_by('t').len()
                .with_columns((np.log(n) - pl.col('len').cast(pl.Float64).log()).cast(pl.Float32).alias('idf'))
                .select('t', 'idf'))
        w = t.join(idf, on='t').group_by('rid').agg(pl.col('idf').sum())
        arr = np.zeros(n, dtype=np.float32)
        arr[w['rid'].to_numpy().astype(np.int64)] = w['idf'].to_numpy()
        del t, w
        gc.collect()
        rec = rec.with_columns(pl.Series(col + '_w', arr))     # rid == row position
        idfs[col] = idf
    return rec, idfs


def take(rec, idx):
    """Row gather by dense rid (rid == row position)."""
    idx = np.asarray(idx)
    if not np.issubdtype(idx.dtype, np.integer):
        idx = idx.astype(np.int64)
    return rec.select(pl.all().gather(pl.Series(idx, dtype=pl.UInt32)))


def _sim(scorer, a, b, workers, scale=100.0):
    out = cpdist(a, b, scorer=scorer, workers=workers, dtype=np.float32)
    return np.asarray(out, dtype=np.float32) / scale


def _nan_if_empty(x, *empties):
    m = np.zeros(len(x), dtype=bool)
    for e in empties:
        m |= e
    x[m] = np.nan
    return x


def _set_feats(ql, sl, idf, qw, sw):
    d = pl.DataFrame({'a': ql, 'b': sl}).with_row_index('i')
    d = d.with_columns(pl.col('a').list.set_intersection(pl.col('b')).alias('t'),
                       pl.col('a').list.unique().list.len().alias('na'),
                       pl.col('b').list.unique().list.len().alias('nb'))
    inter = d['t'].list.len().to_numpy().astype(np.float32)
    na = d['na'].to_numpy().astype(np.float32)
    nb = d['nb'].to_numpy().astype(np.float32)
    jacc = inter / np.maximum(na + nb - inter, 1)
    wjacc = None
    if idf is not None:
        e = (d.select('i', 't').explode('t').drop_nulls('t').join(idf, on='t', how='left')
              .group_by('i').agg(pl.col('idf').sum()))
        wi = np.zeros(len(d), dtype=np.float32)
        wi[e['i'].to_numpy()] = e['idf'].fill_null(0).to_numpy()
        wjacc = wi / np.maximum(qw + sw - wi, 1e-3)
    return inter, jacc, wjacc, d


def pair_features(P, rec, idfs, workers):
    """P: candidate frame chunk (rid2, rid1, blocking columns). Returns float32 matrix."""
    q = take(rec, P['rid2'].to_numpy())
    s = take(rec, P['rid1'].to_numpy())
    F = {c: P[c].cast(pl.Float32).to_numpy() for c in BLOCK_COLS + AGREE_FEATURES}

    def L(df, c):
        return df[c].fill_null('').to_list()

    qn, sn = L(q, 'n_core'), L(s, 'n_core')
    q_nempty = q['n_core'].fill_null('').str.len_chars().to_numpy() == 0
    F['n_ratio'] = _sim(fuzz.ratio, qn, sn, workers)
    F['n_tsort'] = _sim(fuzz.token_sort_ratio, qn, sn, workers)
    F['n_tset'] = _sim(fuzz.token_set_ratio, qn, sn, workers)
    F['n_partial'] = _sim(fuzz.partial_ratio, qn, sn, workers)
    F['n_jw'] = _sim(JaroWinkler.normalized_similarity, qn, sn, workers, 1.0)
    F['k_ratio'] = _sim(fuzz.ratio, L(q, 'n_skel'), L(s, 'n_skel'), workers)
    F['k_tset'] = _sim(fuzz.token_set_ratio, L(q, 'n_skel'), L(s, 'n_skel'), workers)
    qns = [x.replace(' ', '') for x in qn]
    sns = [x.replace(' ', '') for x in sn]
    F['ns_ratio'] = _sim(fuzz.ratio, qns, sns, workers)
    F['ns_partial'] = _sim(fuzz.partial_ratio, qns, sns, workers)
    F['f_tset'] = _sim(fuzz.token_set_ratio, L(q, 'n_full'), L(s, 'n_full'), workers)
    for k in ('n_ratio', 'n_tsort', 'n_tset', 'n_partial', 'n_jw', 'ns_ratio', 'ns_partial'):
        _nan_if_empty(F[k], q_nempty)

    qt = q['n_core'].fill_null('').str.split(' ')
    st = s['n_core'].fill_null('').str.split(' ')
    inter, jacc, wjacc, _ = _set_feats(qt, st, idfs['n_core'], q['n_core_w'].to_numpy(), s['n_core_w'].to_numpy())
    F['n_inter'], F['n_jacc'], F['n_wjacc'] = inter, jacc, wjacc
    _, F['k_jacc'], _, _ = _set_feats(q['n_skel'].fill_null('').str.split(' '),
                                      s['n_skel'].fill_null('').str.split(' '), None, None, None)
    F['q_ntok'] = qt.list.len().to_numpy().astype(np.float32)
    F['s_ntok'] = st.list.len().to_numpy().astype(np.float32)
    s_init = st.list.eval(pl.element().str.slice(0, 1)).list.join('').to_numpy()
    qns_arr = np.array(qns, dtype=object)
    F['initials'] = ((qns_arr == s_init) & (np.array([len(x) for x in qns]) >= 2)).astype(np.float32)
    F['q_indic'] = q['indic'].cast(pl.Float32).to_numpy()
    F['q_domain'] = q['domain'].cast(pl.Float32).to_numpy()
    F['q_src'] = q['src'].cast(pl.Float32).to_numpy()

    qa, sa = L(q, 'a_words'), L(s, 'a_words')
    q_aempty = np.array([len(x) == 0 for x in qa])
    s_aempty = np.array([len(x) == 0 for x in sa])
    F['a_tset'] = _nan_if_empty(_sim(fuzz.token_set_ratio, qa, sa, workers), q_aempty, s_aempty)
    F['a_tsort'] = _nan_if_empty(_sim(fuzz.token_sort_ratio, qa, sa, workers), q_aempty, s_aempty)
    F['a_partial'] = _nan_if_empty(_sim(fuzz.partial_ratio, qa, sa, workers), q_aempty, s_aempty)
    F['a_ratio'] = _nan_if_empty(_sim(fuzz.ratio, qa, sa, workers), q_aempty, s_aempty)
    qal = q['a_words'].fill_null('').str.split(' ')
    sal = s['a_words'].fill_null('').str.split(' ')
    inter, jacc, wjacc, _ = _set_feats(qal, sal, idfs['a_words'], q['a_words_w'].to_numpy(), s['a_words_w'].to_numpy())
    F['a_inter'], F['a_jacc'], F['a_wjacc'] = inter, jacc, wjacc
    F['q_naw'] = qal.list.len().to_numpy().astype(np.float32)
    F['s_naw'] = sal.list.len().to_numpy().astype(np.float32)

    qm, sm = L(q, 'a_nums'), L(s, 'a_nums')
    q_mempty = np.array([len(x) == 0 for x in qm])
    s_mempty = np.array([len(x) == 0 for x in sm])
    F['m_ratio'] = _nan_if_empty(_sim(fuzz.token_set_ratio, qm, sm, workers), q_mempty, s_mempty)
    qml = q['a_nums'].fill_null('').str.split(' ')
    sml = s['a_nums'].fill_null('').str.split(' ')
    inter, jacc, _, d = _set_feats(qml, sml, None, None, None)
    F['m_inter'], F['m_jacc'] = inter, _nan_if_empty(jacc, q_mempty, s_mempty)
    F['m_first_eq'] = _nan_if_empty(
        (qml.list.first().fill_null('') == sml.list.first().fill_null('')).to_numpy().astype(np.float32),
        q_mempty, s_mempty)
    F['m_maxlen'] = (d['t'].list.eval(pl.element().filter(pl.element() != '').str.len_chars()).list.max()
                     .fill_null(0).to_numpy().astype(np.float32))
    F['q_nnum'] = qml.list.len().to_numpy().astype(np.float32) * (~q_mempty)
    F['s_nnum'] = sml.list.len().to_numpy().astype(np.float32) * (~s_mempty)
    F['q_addr_empty'] = (q_aempty & q_mempty).astype(np.float32)

    qall = [a + ' ' + b + ' ' + c for a, b, c in zip(qn, qa, qm)]
    sall = [a + ' ' + b + ' ' + c for a, b, c in zip(sn, sa, sm)]
    F['all_tset'] = _sim(fuzz.token_set_ratio, qall, sall, workers)

    # ---- sibling discrimination -------------------------------------------------------
    # Genuine corruption of a house number is random per record (2454->245, 1014->014,
    # 508->0508); a sibling branch sits at a *different* nearby number (13558 vs 13565).
    miss_m = q_mempty | s_mempty
    qsig = _clean_list(qml).list.unique().list.sort().list.join(' ')
    ssig = _clean_list(sml).list.unique().list.sort().list.join(' ')
    F['num_sig_eq'] = _nan_if_empty((qsig == ssig).to_numpy().astype(np.float32), miss_m)
    qf = qml.list.first().fill_null('').to_list()
    sf = sml.list.first().fill_null('').to_list()
    F['num_first_contains'] = _nan_if_empty(np.array(
        [float(a != b and (a in b or b in a)) if a and b else 0.0 for a, b in zip(qf, sf)],
        dtype=np.float32), miss_m)
    F['num_first_edit'] = _nan_if_empty(np.asarray(
        cpdist(qf, sf, scorer=Levenshtein.distance, workers=workers), dtype=np.float32), miss_m)
    qi = pl.Series(qf).str.slice(0, 15).str.to_integer(strict=False).cast(pl.Float64).to_numpy()
    si = pl.Series(sf).str.slice(0, 15).str.to_integer(strict=False).cast(pl.Float64).to_numpy()
    F['num_first_logdiff'] = np.log1p(np.abs(qi - si)).astype(np.float32)   # NaN when missing

    qc = _clean_list(q['a_codes'].fill_null('').str.split(' '))
    sc = _clean_list(s['a_codes'].fill_null('').str.split(' '))
    c_empty = (qc.list.len() == 0).to_numpy() | (sc.list.len() == 0).to_numpy()
    inter, jacc, _, _ = _set_feats(qc, sc, None, None, None)
    F['code_inter'] = _nan_if_empty(inter, c_empty)
    F['code_jacc'] = _nan_if_empty(jacc, c_empty)

    qft = _clean_list(q['n_full'].fill_null('').str.split(' ')).list.unique()
    sft = _clean_list(s['n_full'].fill_null('').str.split(' ')).list.unique()
    F['name_extra_n'] = qft.list.set_difference(sft).list.len().to_numpy().astype(np.float32)
    F['name_missing_n'] = sft.list.set_difference(qft).list.len().to_numpy().astype(np.float32)
    F['name_extra_idf'] = _idf_sum(qt.list.set_difference(st), idfs['n_core'])
    F['name_missing_idf'] = _idf_sum(st.list.set_difference(qt), idfs['n_core'])
    ql = qft.list.eval(pl.element().filter(pl.element().is_in(_LEGAL_LIST)))
    sl = sft.list.eval(pl.element().filter(pl.element().is_in(_LEGAL_LIST)))
    both = ((ql.list.len() > 0) & (sl.list.len() > 0)).to_numpy()
    shared = ql.list.set_intersection(sl).list.len().to_numpy()
    F['legal_conflict'] = (both & (shared == 0)).astype(np.float32)
    F['legal_eq'] = (both & (shared == ql.list.len().to_numpy())
                     & (shared == sl.list.len().to_numpy())).astype(np.float32)
    return np.column_stack([np.asarray(F[c], dtype=np.float32) for c in PAIR_FEATURES])


def _clean_list(lst):
    return lst.list.eval(pl.element().filter(pl.element() != ''))


def _idf_sum(lists, idf):
    d = pl.DataFrame({'t': lists}).with_row_index('i')
    e = (d.explode('t').drop_nulls('t').filter(pl.col('t') != '')
          .join(idf, on='t', how='left').group_by('i').agg(pl.col('idf').sum()))
    out = np.zeros(len(d), dtype=np.float32)
    out[e['i'].to_numpy()] = e['idf'].fill_null(0).to_numpy()
    return out


def build_matrix(cand, rec, idfs, path, workers, chunk, log):
    n, nf = cand.height, len(PAIR_FEATURES)
    X = np.lib.format.open_memmap(path, mode='w+', dtype=np.float32, shape=(n, nf))
    i = 0
    while i < n:
        chunk = next_chunk(chunk, 250_000, log, 'features')
        X[i:i + chunk] = pair_features(cand.slice(i, chunk), rec, idfs, workers)
        i += chunk
        log(f'  features {min(i, n):,}/{n:,}')
    X.flush()
    return X
