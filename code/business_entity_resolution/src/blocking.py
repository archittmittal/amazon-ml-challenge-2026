"""Candidate generation: IDF-weighted conjunctive-key inverted index.

EDA showed the vocabulary is small and heavily reused (only ~59% of Source-1 core names
are unique, while name+address is unique), so single tokens are either too frequent to be
selective or too fragile to be robust. Identity lives in *combinations*. Each record emits:

  NN  unordered pair of name skeleton tokens       (order / typo / script robust)
  NA  name skeleton token x address word            (name + locality)
  MA  address number x address word                 (house number + street)
  AA  unordered pair of address words               (street + city; survives number noise)
  N1 / A1 / SP  single name token / address word / 8-char space-free name prefix
                (SP catches domain-style names such as "coastaltungsten.com")
  AC / CA  composite address code ("c303", "21149a61") alone and x address word:
           unit identifiers are the most selective thing an Indian address carries

Keys are scoped by the country string (open set: the label is hashed into the key and never
enumerated). Keys whose Source-1 document frequency exceeds `cap` are dropped; a
(query, S1) pair is scored by the sum of log(N/df) over shared keys.

Retrieval runs S2/S3 -> S1: each S2/S3 record belongs to at most one S1 entity, so a small
per-query top-K is a natural, bounded candidate list.
"""
import gc

import numpy as np
import polars as pl

from rapidfuzz import fuzz
from rapidfuzz.process import cpdist

from features import take
from memguard import next_chunk

MAX_NAME_TOK = 5
MAX_ADDR_TOK = 7
MAX_NUMS = 3


def _tokens(df, col, maxn, minlen=2):
    return (df.select('rid', 'country', pl.col(col).str.split(' ').list.head(maxn).alias('t'))
              .explode('t')
              .filter(pl.col('t').is_not_null() & (pl.col('t').str.len_chars() >= minlen))
              .unique(['rid', 't']))


def _pair_keys(a, b, tag, unordered):
    j = a.join(b.select('rid', pl.col('t').alias('u')), on='rid')
    if unordered:
        j = j.filter(pl.col('t') < pl.col('u'))
    return j.select('rid', pl.concat_str([pl.lit(tag), pl.col('country'), pl.col('t'), pl.col('u')],
                                         separator='|').hash(11).alias('key'))


def _single_keys(a, tag):
    return a.select('rid', pl.concat_str([pl.lit(tag), pl.col('country'), pl.col('t')],
                                         separator='|').hash(11).alias('key'))


def record_keys(df):
    nk = _tokens(df, 'n_skel', MAX_NAME_TOK)
    nt = _tokens(df, 'n_core', MAX_NAME_TOK)
    aw = _tokens(df, 'a_words', MAX_ADDR_TOK)
    an = _tokens(df, 'a_nums', MAX_NUMS, minlen=1)
    ac = _tokens(df, 'a_codes', MAX_NUMS + 1, minlen=3)
    sp = (df.select('rid', 'country', pl.col('n_core').str.replace_all(' ', '').str.slice(0, 8).alias('t'))
            .filter(pl.col('t').str.len_chars() >= 5))
    parts = [
        _pair_keys(nk, nk, 'NN', True),
        _pair_keys(nk, aw, 'NA', False),
        _pair_keys(an, aw, 'MA', False),
        _pair_keys(aw, aw, 'AA', True),
        _single_keys(nt, 'N1'),
        _single_keys(aw, 'A1'),
        _single_keys(sp, 'SP'),
        _single_keys(ac, 'AC'),
        _pair_keys(ac, aw, 'CA', False),
    ]
    return pl.concat(parts).unique()


class KeyIndex:
    """Inverted index over Source-1 records (only keys with df <= cap are kept)."""

    def __init__(self, s1, cap, chunk=400_000):
        keys = pl.concat([record_keys(s1.slice(i, chunk)) for i in range(0, len(s1), chunk)])
        df = keys.group_by('key').len().rename({'len': 'df'}).filter(pl.col('df') <= cap)
        n = max(len(s1), 2)
        df = df.with_columns((np.log(n) - pl.col('df').cast(pl.Float32).log()).cast(pl.Float32).alias('w'))
        self.post = keys.join(df.select('key', 'w'), on='key').rename({'rid': 'rid1'})

    def query(self, q, topk, min_ratio, blob=None, wide=50, wide_ratio=0.1, extra_k=2, extra_min_sim=0.7,
              workers=-1):
        """Key-score retrieval. With `blob` (dense per-record text), two-stage: keep the usual
        top-k by key score, and additionally rescue up to `extra_k` pairs per query from the
        top-`wide` pool by plain string similarity. The blocking lab showed ~1% of true pairs are
        retrieved but cut (rank > 10 or low score ratio), largely among ties of generic names."""
        qk = record_keys(q).rename({'rid': 'rid2'})
        j = (qk.join(self.post, on='key')
               .group_by('rid2', 'rid1')
               .agg(pl.col('w').sum().alias('bscore'), pl.len().cast(pl.Float32).alias('nkeys')))
        del qk
        j = j.with_columns(pl.col('bscore').rank('ordinal', descending=True).over('rid2').alias('brank'),
                           pl.col('bscore').max().over('rid2').alias('qmax'))
        base = (pl.col('brank') <= topk) & (pl.col('bscore') >= min_ratio * pl.col('qmax'))
        if blob is None:
            out = j.filter(base).with_columns(pl.lit(None, dtype=pl.Float32).alias('b_sim'),
                                              pl.lit(0.0, dtype=pl.Float32).alias('b_extra'))
        else:
            w = j.filter((pl.col('brank') <= wide) & (pl.col('bscore') >= wide_ratio * pl.col('qmax')))
            sim = cpdist(take(blob, w['rid2'].to_numpy())['blob'].to_list(),
                         take(blob, w['rid1'].to_numpy())['blob'].to_list(),
                         scorer=fuzz.token_set_ratio, workers=workers, dtype=np.float32)
            w = w.with_columns(pl.Series('b_sim', np.asarray(sim, dtype=np.float32) / 100.0), base.alias('_base'))
            ex = (w.filter(~pl.col('_base') & (pl.col('b_sim') >= extra_min_sim))
                   .with_columns(pl.col('b_sim').rank('ordinal', descending=True).over('rid2').alias('_sr'))
                   .filter(pl.col('_sr') <= extra_k).drop('_sr')
                   .with_columns(pl.lit(1.0, dtype=pl.Float32).alias('b_extra')))
            out = pl.concat([w.filter(pl.col('_base')).with_columns(pl.lit(0.0, dtype=pl.Float32).alias('b_extra')),
                             ex]).drop('_base')
            del w, ex
        out = out.with_columns(pl.col('brank').cast(pl.Float32)).rechunk()
        del j
        return out


def generate_candidates(rec, cap, topk, min_ratio, chunk, log, rerank=True, query_frac=None, workers=-1):
    cols = ['rid', 'country', 'n_core', 'n_skel', 'a_words', 'a_nums', 'a_codes']
    s1 = rec.filter(pl.col('src') == 1).select(cols)
    idx = KeyIndex(s1, cap)
    log(f'blocking index: {len(s1):,} S1 records, {idx.post.height:,} postings (cap={cap}, rerank={rerank})')
    q = rec.filter(pl.col('src') != 1).select(cols)
    if query_frac is not None:          # label-free sample pass (used to mine maps before blocking)
        q = q.filter(pl.col('rid').hash(29) % 1000 < query_frac * 1000)
    blob = None
    if rerank:
        blob = rec.select(pl.concat_str([pl.col('n_core').fill_null(''), pl.col('a_words').fill_null(''),
                                         pl.col('a_nums').fill_null('')], separator=' ').alias('blob'))
    out = []
    i = step = 0
    while i < len(q):
        chunk = next_chunk(chunk, 20_000, log, 'blocking')
        out.append(idx.query(q.slice(i, chunk), topk, min_ratio, blob=blob, workers=workers))
        i += chunk
        step += 1
        if step % 5 == 1:
            log(f'  queried {min(i, len(q)):,}/{len(q):,}')
        if step % 10 == 0:
            gc.collect()   # nudge the allocator to return freed memory to the OS
    del idx, blob
    gc.collect()
    c = pl.concat(out).drop('qmax')
    if rerank:
        log(f'  two-stage blocking: {int(c["b_extra"].sum()):,} pairs rescued by string re-ranking '
            f'({c["b_extra"].mean():.3f} of candidates)')
    # query- and entity-level blocking context (unsupervised; identical at train and test time)
    c = c.with_columns(
        pl.col('bscore').max().over('rid2').alias('q_bmax'),
        pl.col('bscore').filter(pl.col('brank') == 2).max().over('rid2').fill_null(0).alias('q_b2'),
        pl.len().over('rid2').cast(pl.Float32).alias('q_ncand'),
        pl.len().over('rid1').cast(pl.Float32).alias('s_nq'),
        (pl.col('brank') == 1).sum().over('rid1').cast(pl.Float32).alias('s_nq_top1'),
    ).with_columns(
        (pl.col('bscore') / pl.col('q_bmax')).alias('b_ratio'),
        pl.when(pl.col('brank') == 1).then(pl.col('bscore') - pl.col('q_b2'))
          .otherwise(pl.col('bscore') - pl.col('q_bmax')).alias('b_margin'),
    )
    return c.sort('rid2', 'rid1')
