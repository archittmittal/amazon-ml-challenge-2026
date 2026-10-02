"""Self-supervised token-equivalence mining (label-free, runs on train and test alike).

Take high-confidence blocking pairs (query's top-1 S1 with many shared keys). When the two
token sets differ by exactly one token on each side, that residue is almost always a
surface variant of the same concept: 'st'/'street', 'mh'/'maharashtra', 'ohio'/'oh',
transliteration residue ('praibhet'/'private'), French 'av'/'avenue', 'r'/'rue', ...
Counting such residues over millions of pairs and keeping only frequent, dominant ones
gives a corpus-specific substitution table with no labels and no external dictionary.
This is what lets the pipeline adapt to France, which never appears in training.
"""
import polars as pl
from rapidfuzz import fuzz

from text import skeleton


def _subseq(a, b):
    it = iter(b)
    return all(ch in it for ch in a)


def plausible(x, y):
    """A mined substitution must look like a surface variant, not a co-occurring different word.
    Accepts 'tx'/'texas', 'mh'/'maharashtra', 'kanstrakshan'/'construction', 'lndia'/'india';
    rejects 'atlantique'/'pays'."""
    if fuzz.ratio(x, y) >= 80:
        return True
    sx, sy = skeleton(x) or x, skeleton(y) or y
    if sx[:1] != sy[:1]:
        return False
    return _subseq(sx, sy) or _subseq(sy, sx) or fuzz.ratio(sx, sy) >= 70


def _take(rec, idx, col):
    return rec.select(pl.col(col).gather(idx)).to_series()


def mine(cand, rec, cols, min_count=25, min_share=0.6, min_keys=5, max_pairs=4_000_000, log=print):
    conf = cand.filter((pl.col('brank') == 1) & (pl.col('nkeys') >= min_keys) & (pl.col('b_margin') > 0))
    if conf.height > max_pairs:
        conf = conf.sample(max_pairs, seed=0)
    i2, i1 = conf['rid2'], conf['rid1']
    maps = {}
    for col in cols:
        q = _take(rec, i2, col).str.split(' ')
        s = _take(rec, i1, col).str.split(' ')
        d = pl.DataFrame({'x': q.list.set_difference(s), 'y': s.list.set_difference(q)})
        d = (d.filter((pl.col('x').list.len() == 1) & (pl.col('y').list.len() == 1))
              .select(pl.col('x').list.first(), pl.col('y').list.first())
              .filter((pl.col('x') != '') & (pl.col('y') != '') & (pl.col('x') != pl.col('y'))))
        cnt = d.group_by('x', 'y').len()
        tot = d.group_by('x').len().rename({'len': 'tot'})
        good = (cnt.join(tot, on='x')
                   .filter((pl.col('len') >= min_count) & (pl.col('len') / pl.col('tot') >= min_share))
                   .sort('len', descending=True))
        m = dict(zip(good['x'].to_list(), good['y'].to_list()))
        rejected = [(k, v) for k, v in m.items() if not plausible(k, v)]
        m = {k: v for k, v in m.items() if plausible(k, v)}
        m = {k: v for k, v in m.items() if v not in m}          # no chains / cycles
        if rejected:
            log(f'  rejected {len(rejected)} implausible {col} merges, e.g. {rejected[:6]}')
        maps[col] = m
        top = list(m.items())[:12]
        log(f'  mined {len(m):,} equivalences for {col}; e.g. {top}')
    return maps


def merge(base, extra):
    out = {}
    for col in set(base) | set(extra):
        m = dict(base.get(col, {}))
        m.update(extra.get(col, {}))
        out[col] = {k: v for k, v in m.items() if v not in m}
    return out


def apply(rec, maps):
    exprs = []
    for col, m in maps.items():
        if not m:
            continue
        old, new = list(m.keys()), list(m.values())
        exprs.append(pl.col(col).str.split(' ')
                     .list.eval(pl.element().replace(old, new))
                     .list.join(' ').alias(col))
    return rec.with_columns(exprs) if exprs else rec
