"""Blocking lab: measure candidate-generation recall variants on labelled training data (CPU only).

The matcher can never recover a true pair that blocking did not propose, and at 97.7% pair recall
blocking is now the binding ceiling. This script answers, with labels and without any model
training, *why* true pairs are missed and which change recovers them:

  1. baseline (current settings) on ALL queries   -> exact recall, and label-free pseudo-members
  2. miss taxonomy on a query sample: ranked beyond top-k / cut by score ratio / never retrieved,
     and a profile of the never-retrieved pairs (missing address? name similarity? country?)
  3. variants on the same sample: wider top-k / lower ratio, higher key cap
  4. record-to-cluster retrieval: index every S1 *together with its confident members* (chosen
     label-free from baseline blocking), so a new record can match any member's wording

    python src/blocking_lab.py [--data_dir ...] [--sample 0.2]
"""
import argparse
import gc
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault('_RJEM_MALLOC_CONF', 'background_thread:true,dirty_decay_ms:1000,muzzy_decay_ms:0')

from run import find_data_dir, log, s1_dropout  # noqa: E402  (first: its bootstrap installs rapidfuzz/polars)

import numpy as np          # noqa: E402
import polars as pl         # noqa: E402
from rapidfuzz import fuzz  # noqa: E402
from rapidfuzz.process import cpdist  # noqa: E402

import blocking             # noqa: E402
from prepare import load_split, load_truth  # noqa: E402

COLS = ['rid', 'country', 'n_core', 'n_skel', 'a_words', 'a_nums', 'a_codes']


class Index(blocking.KeyIndex):
    """KeyIndex over arbitrary documents; `doc_rid` is the S1 id each document votes for."""

    def __init__(self, docs, cap, chunk=400_000):
        keys = pl.concat([blocking.record_keys(docs.slice(i, chunk)) for i in range(0, len(docs), chunk)]).unique()
        n_s1 = docs['rid'].n_unique()
        df = keys.group_by('key').len().rename({'len': 'df'}).filter(pl.col('df') <= cap)
        df = df.with_columns((np.log(max(n_s1, 2)) - pl.col('df').cast(pl.Float32).log()).cast(pl.Float32).alias('w'))
        self.post = keys.join(df.select('key', 'w'), on='key').rename({'rid': 'rid1'})


def run_queries(idx, q, topk, min_ratio, chunk=100_000):
    out = []
    for i in range(0, len(q), chunk):
        out.append(idx.query(q.slice(i, chunk), topk, min_ratio))
    return pl.concat(out).with_columns((pl.col('bscore') / pl.col('qmax')).alias('ratio'))


def recall(c, truth_q):
    hit = truth_q.join(c.select('rid1', 'rid2'), on=['rid1', 'rid2'], how='semi').height
    return hit / max(truth_q.height, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', default=None)
    ap.add_argument('--work_dir', default='/tmp/ber_work')
    ap.add_argument('--workers', type=int, default=os.cpu_count())
    ap.add_argument('--sample', type=float, default=0.2, help='fraction of queries for variant tests')
    ap.add_argument('--out', default='/kaggle/working/blocking_lab.txt' if os.path.isdir('/kaggle/working') else 'blocking_lab.txt')
    args = ap.parse_args()
    data_dir = args.data_dir or find_data_dir()
    report = []

    def say(msg):
        log(msg)
        report.append(msg)
        with open(args.out, 'w') as f:
            f.write('\n'.join(report) + '\n')

    rec = load_split(data_dir, 'train', args.work_dir, args.workers)
    truth = load_truth(data_dir, rec)
    rec, truth, rep = s1_dropout(rec, truth, data_dir)          # same conditions as training
    say(f'S1 dropout (as in training): {rep}; true pairs {truth.height:,}')
    rec = rec.select(COLS + ['src'])
    s1 = rec.filter(pl.col('src') == 1).select(COLS)
    q_all = rec.filter(pl.col('src') != 1).select(COLS)
    qs = q_all.filter(pl.col('rid').hash(23) % 1000 < args.sample * 1000)
    truth_s = truth.join(qs.select(pl.col('rid').alias('rid2')), on='rid2', how='semi')
    say(f'queries: all {q_all.height:,}, sample {qs.height:,} ({truth_s.height:,} true pairs in sample)')

    # ---- 1. baseline on all queries -------------------------------------------------------
    t = time.time()
    idx = Index(s1, 200)
    base = run_queries(idx, q_all, 10, 0.3)
    say(f'[baseline cap=200 topk=10 ratio=0.3] recall={recall(base, truth):.5f} '
        f'pairs/query={base.height / q_all.height:.2f} ({(time.time() - t) / 60:.0f} min)')

    # ---- 2. miss taxonomy on the sample ---------------------------------------------------
    wide = run_queries(idx, qs, 50, 0.0)
    j = truth_s.join(wide.select('rid1', 'rid2', 'brank', 'ratio'), on=['rid1', 'rid2'], how='left')
    kept = (pl.col('brank') <= 10) & (pl.col('ratio') >= 0.3)
    tax = j.select(
        kept.fill_null(False).mean().alias('kept'),
        ((pl.col('brank') > 10) & (pl.col('ratio') >= 0.3)).fill_null(False).mean().alias('rank>10'),
        ((pl.col('ratio') < 0.3)).fill_null(False).mean().alias('ratio<0.3'),
        pl.col('brank').is_null().mean().alias('never_retrieved(top50)'),
    )
    say(f'miss taxonomy (share of true pairs): {tax.to_dicts()[0]}')
    for k, r in ((15, 0.3), (20, 0.2), (30, 0.15)):
        c = wide.filter((pl.col('brank') <= k) & (pl.col('ratio') >= r))
        say(f'  [cap=200 topk={k} ratio={r}] sample recall={recall(c, truth_s):.5f} pairs/query={c.height / qs.height:.2f}')
    say(f'  [cap=200 topk=10 ratio=0.3] sample recall={recall(wide.filter(kept), truth_s):.5f} (baseline on sample)')

    miss = j.filter(pl.col('brank').is_null()).select('rid1', 'rid2')
    if miss.height:
        ms = miss.sample(min(50_000, miss.height), seed=1)
        a = rec.select(pl.all().gather(pl.Series(ms['rid2'].to_numpy().astype(np.int64))))
        b = rec.select(pl.all().gather(pl.Series(ms['rid1'].to_numpy().astype(np.int64))))
        nm = np.asarray(cpdist(a['n_core'].fill_null('').to_list(), b['n_core'].fill_null('').to_list(),
                               scorer=fuzz.token_set_ratio, workers=-1))
        ad = np.asarray(cpdist(a['a_words'].fill_null('').to_list(), b['a_words'].fill_null('').to_list(),
                               scorer=fuzz.token_set_ratio, workers=-1))
        no_addr = ((a['a_words'].fill_null('') == '') & (a['a_nums'].fill_null('') == '')).to_numpy()
        prof = pl.DataFrame({'country': a['country'], 'no_addr': no_addr, 'name_sim': nm, 'addr_sim': ad})
        say('never-retrieved profile: ' + str(prof.select(
            pl.col('no_addr').mean().alias('no_address'),
            (pl.col('name_sim') < 50).mean().alias('name_sim<50'),
            (pl.col('name_sim') >= 80).mean().alias('name_sim>=80'),
            (pl.col('addr_sim') < 50).mean().alias('addr_sim<50'),
            ((pl.col('name_sim') < 50) & (pl.col('no_addr') | (pl.col('addr_sim') < 50))).mean().alias('both_weak'),
        ).to_dicts()[0]))
        say('never-retrieved by country: ' + str(prof.group_by('country').len().sort('country').rows()))
        ex = pl.DataFrame({'q_name': a['n_core'], 'q_addr': a['a_words'], 's_name': b['n_core'], 's_addr': b['a_words']})
        for r in ex.sample(min(12, ex.height), seed=2).rows():
            say(f'    miss: {r[0]!r} | {r[1]!r}   <->   S1 {r[2]!r} | {r[3]!r}')
    del wide, j
    gc.collect()

    # ---- 3. higher key cap ----------------------------------------------------------------
    for cap in (400,):
        t = time.time()
        idx2 = Index(s1, cap)
        c = run_queries(idx2, qs, 10, 0.3)
        say(f'  [cap={cap} topk=10 ratio=0.3] sample recall={recall(c, truth_s):.5f} '
            f'pairs/query={c.height / qs.height:.2f} postings={idx2.post.height:,} ({(time.time() - t) / 60:.0f} min)')
        del idx2, c
        gc.collect()

    # ---- 4. record-to-cluster retrieval (label-free members) ------------------------------
    t = time.time()
    # label-free members: a query's clear top-1 S1 (>=5 shared keys, strictly ahead of its runner-up);
    # one per S1 (the strongest) keeps the index affordable on a 30GB machine -> recall gain measured
    # here is a lower bound of what indexing more members would give.
    b2 = base.with_columns(pl.col('bscore').filter(pl.col('brank') == 2).max().over('rid2').fill_null(0).alias('b2'))
    members = (b2.filter((pl.col('brank') == 1) & (pl.col('nkeys') >= 5) & (pl.col('bscore') > pl.col('b2')))
                 .sort('bscore', descending=True).unique('rid1', keep='first').select('rid2', 'rid1'))
    del b2
    mem_docs = (rec.select(pl.all().gather(pl.Series(members['rid2'].to_numpy().astype(np.int64))))
                   .select(COLS).with_columns(pl.Series('rid', members['rid1'].to_numpy()).cast(pl.UInt32)))
    precision_members = members.join(truth, on=['rid1', 'rid2'], how='semi').height / max(members.height, 1)
    say(f'record-to-cluster: {members.height:,} label-free members (true-member precision {precision_members:.4f})')
    idx3 = Index(pl.concat([s1, mem_docs]), 200)
    c3 = run_queries(idx3, qs, 10, 0.3)
    base_s = base.join(qs.select(pl.col('rid').alias('rid2')), on='rid2', how='semi')
    union = pl.concat([base_s.select('rid1', 'rid2'), c3.select('rid1', 'rid2')]).unique()
    say(f'  [cluster index alone] sample recall={recall(c3, truth_s):.5f} pairs/query={c3.height / qs.height:.2f}')
    say(f'  [baseline UNION cluster] sample recall={recall(union, truth_s):.5f} '
        f'pairs/query={union.height / qs.height:.2f} ({(time.time() - t) / 60:.0f} min)')
    say('done')


if __name__ == '__main__':
    main()
