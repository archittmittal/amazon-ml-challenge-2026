"""Label-free diagnostics of a submission, per country.

The test set has no labels, but the data generator is the same for every country, so a
healthy submission should reproduce the *training* ground-truth statistics in each country:
singleton rate, matches per S1 entity, and the fraction of S2/S3 records that belong to some
S1 entity. Deviations localise the loss without needing labels:

  test S2/S3 assigned%  << train matched%   -> recall loss (model or blocking)
  test S2/S3 in-candidates% << train matched% -> blocking loss specifically
  test preds/S1 >> train matches/S1           -> false merges (precision loss)

    python src/diagnose.py [data_dir] [output_dir]
"""
import glob
import os
import sys

import polars as pl

pl.Config.set_fmt_str_lengths(90)
pl.Config.set_tbl_width_chars(250)
pl.Config.set_tbl_rows(40)
pl.Config.set_tbl_cols(20)


def find_data_dir():
    for base in ('/kaggle/input', '.'):
        hits = glob.glob(os.path.join(base, '**', 'train', 'train_source1.tsv'), recursive=True)
        if hits:
            return os.path.dirname(os.path.dirname(os.path.abspath(hits[0])))
    raise FileNotFoundError('pass data_dir explicitly')


def read(path, cols=None):
    df = pl.read_csv(path, separator='\t', quote_char=None, infer_schema=False)
    return df.select(cols) if cols else df


def explode_ids(df, col, out):
    return (df.with_columns(pl.col(col).fill_null('').str.split(','))
              .explode(col).filter(pl.col(col) != '')
              .rename({col: out}))


def per_s1(s1, pairs, name):
    cnt = pairs.group_by('source1_entity_id').len().rename({'len': name})
    return (s1.join(cnt, on='source1_entity_id', how='left')
              .with_columns(pl.col(name).fill_null(0)))


def main():
    data_dir = sys.argv[1] if len(sys.argv) > 1 else find_data_dir()
    out_dir = sys.argv[2] if len(sys.argv) > 2 else '/kaggle/working/output'
    print('data:', data_dir, '| outputs:', out_dir, flush=True)

    # ---------------- training reference (ground truth) ----------------
    tr1 = read(os.path.join(data_dir, 'train', 'train_source1.tsv'), ['entity_id', 'country']) \
        .rename({'entity_id': 'source1_entity_id'})
    tr23 = pl.concat([read(os.path.join(data_dir, 'train', f'train_source{k}.tsv'), ['entity_id', 'country'])
                      for k in (2, 3)])
    gt = explode_ids(read(os.path.join(data_dir, 'train', 'train_ground_truth.tsv')), 'matched_entity_ids', 'e2')
    g = per_s1(tr1, gt, 'k')
    ref = g.group_by('country').agg(
        pl.len().alias('n_s1'),
        (pl.col('k') == 0).mean().alias('singleton%'),
        pl.col('k').mean().alias('matches/S1'))
    matched = tr23.join(gt.select(pl.col('e2').alias('entity_id')), on='entity_id', how='semi')
    ref = ref.join(matched.group_by('country').len().rename({'len': 'm'}), on='country') \
             .join(tr23.group_by('country').len().rename({'len': 'n23'}), on='country') \
             .with_columns((pl.col('m') / pl.col('n23')).alias('S23 matched%')).drop('m', 'n23')
    print('\n=== TRAIN ground truth, per country (what a perfect submission looks like) ===')
    print(ref.sort('country'))
    print('train matches-per-S1 histogram (share of S1 entities):')
    print(g.group_by('country', 'k').len()
           .with_columns((pl.col('len') / pl.col('len').sum().over('country')).round(4).alias('share'))
           .pivot(on='country', index='k', values='share').sort('k'))
    del tr1, tr23, gt, g, matched

    # ---------------- test submission ----------------
    te1 = read(os.path.join(data_dir, 'test', 'test_source1.tsv')).rename({'entity_id': 'source1_entity_id'})
    te23 = pl.concat([read(os.path.join(data_dir, 'test', f'test_source{k}.tsv')) for k in (2, 3)])
    m = read(os.path.join(out_dir, 'matching_results.tsv'))
    c = read(os.path.join(out_dir, 'candidate_pairs.tsv'))
    mp = explode_ids(m, 'matched_entity_ids', 'e2')
    cp = explode_ids(c, 'candidate_entity_ids', 'e2')
    s1 = te1.select('source1_entity_id', 'country')
    t = per_s1(per_s1(s1, mp, 'np'), cp, 'nc')
    st = t.group_by('country').agg(
        pl.len().alias('n_s1'),
        (pl.col('np') == 0).mean().alias('pred-empty%'),
        pl.col('np').mean().alias('preds/S1'),
        (pl.col('nc') == 0).mean().alias('cand-empty%'),
        pl.col('nc').mean().alias('cands/S1'))
    c23 = te23.select(pl.col('entity_id').alias('e2'), pl.col('country').alias('c2'),
                      pl.col('entity_id').str.slice(0, 2).alias('src'))
    tot = c23.group_by('c2').len().rename({'len': 'n23', 'c2': 'country'})
    asg = mp.select('e2').unique().join(c23, on='e2').group_by('c2').len().rename({'len': 'a', 'c2': 'country'})
    inc = cp.select('e2').unique().join(c23, on='e2').group_by('c2').len().rename({'len': 'ic', 'c2': 'country'})
    st = (st.join(tot, on='country').join(asg, on='country', how='left').join(inc, on='country', how='left')
            .with_columns((pl.col('a') / pl.col('n23')).alias('S23 assigned%'),
                          (pl.col('ic') / pl.col('n23')).alias('S23 in-cands%'))
            .drop('a', 'ic', 'n23'))
    print('\n=== TEST submission, per country (compare column-by-column with TRAIN above) ===')
    print(st.sort('country'))
    print('test predicted-matches-per-S1 histogram:')
    print(t.group_by('country', 'np').len()
           .with_columns((pl.col('len') / pl.col('len').sum().over('country')).round(4).alias('share'))
           .pivot(on='country', index='np', values='share').sort('np'))
    by_src = (c23.join(mp.select('e2').unique().with_columns(pl.lit(1).alias('a')), on='e2', how='left')
                 .group_by('c2', 'src').agg(pl.col('a').fill_null(0).mean().alias('assigned%'))
                 .sort('c2', 'src'))
    print('test S2/S3 assigned% by source:')
    print(by_src)
    xc = (mp.join(s1, on='source1_entity_id').join(c23, on='e2')
            .filter(pl.col('country') != pl.col('c2')).height)
    print(f'cross-country matches (must be 0): {xc}')

    # ---------------- samples to eyeball ----------------
    rec = pl.concat([te1.rename({'source1_entity_id': 'entity_id'}), te23])

    def show(ids, title):
        print(f'\n--- {title} ---')
        for sid in ids:
            r = rec.filter(pl.col('entity_id') == sid).row(0)
            print('S1 ', r[1], ' | ', r[2])
            for kind, frame in (('pred', mp), ('cand', cp)):
                e = frame.filter(pl.col('source1_entity_id') == sid)['e2']
                if kind == 'cand':
                    e = e.filter(~e.is_in(mp.filter(pl.col('source1_entity_id') == sid)['e2'].implode()))
                for x in rec.join(pl.DataFrame({'entity_id': e}), on='entity_id').rows():
                    print(f'   {kind}', x[0], x[1], ' | ', x[2])

    for cty in sorted(st['country'].to_list()):
        sub = t.filter(pl.col('country') == cty)
        a = sub.filter((pl.col('np') == 0) & (pl.col('nc') > 0))
        show(a.sample(min(5, a.height), seed=1)['source1_entity_id'].to_list(),
             f'{cty}: predicted EMPTY but had candidates (recall suspects)')
        b = sub.filter(pl.col('np') >= 7)
        show(b.sample(min(3, b.height), seed=1)['source1_entity_id'].to_list(),
             f'{cty}: >=7 predicted matches (over-merge suspects)')
        un = (c23.filter(pl.col('c2') == cty).join(cp.select('e2').unique(), on='e2', how='anti'))
        print(f'\n--- {cty}: S2/S3 records in NO candidate list ({un.height:,}; ~26% expected to be decoys) ---')
        for x in te23.join(un.select(pl.col('e2').alias('entity_id')).sample(min(10, un.height), seed=2),
                           on='entity_id').rows():
            print('  ', x[0], x[1], ' | ', x[2])


if __name__ == '__main__':
    main()
