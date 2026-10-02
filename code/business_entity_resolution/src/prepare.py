"""Load raw TSVs, canonicalise every record in parallel, cache as parquet."""
import os
from multiprocessing import Pool

import numpy as np
import polars as pl

from text import normalize_record

COLS = ['n_full', 'n_core', 'n_skel', 'a_words', 'a_nums', 'a_codes', 'indic', 'domain']
CACHE_VERSION = 'v2'   # bump whenever text.py normalisation changes


def read_tsv(path):
    # quote_char=None: names contain stray quotes; every field is tab-delimited.
    df = pl.read_csv(path, separator='\t', quote_char=None, infer_schema=False)
    return df.with_columns(pl.col('business_name').fill_null(''),
                           pl.col('business_address').fill_null(''),
                           pl.col('country').fill_null('').str.strip_chars())


def _work(chunk):
    return [normalize_record(n, a, c) for n, a, c in chunk]


def normalize_frame(df, workers):
    rows = list(zip(df['business_name'].to_list(), df['business_address'].to_list(),
                    df['country'].to_list()))
    step = max(1, len(rows) // (workers * 8) + 1)
    chunks = [rows[i:i + step] for i in range(0, len(rows), step)]
    with Pool(workers) as pool:
        res = [r for part in pool.imap(_work, chunks) for r in part]
    cols = list(zip(*res)) if res else [[] for _ in COLS]
    out = {c: list(v) for c, v in zip(COLS, cols)}
    return df.with_columns(
        *[pl.Series(c, out[c], dtype=pl.Utf8) for c in COLS[:6]],
        pl.Series('indic', out['indic'], dtype=pl.Boolean),
        pl.Series('domain', out['domain'], dtype=pl.Boolean),
    )


def load_split(data_dir, split, cache_dir, workers):
    """Returns one frame with all three sources; `src` in {1,2,3}, `rid` dense row id."""
    os.makedirs(cache_dir, exist_ok=True)
    cache = os.path.join(cache_dir, f'{split}_records_{CACHE_VERSION}.parquet')
    if os.path.exists(cache):
        return pl.read_parquet(cache)
    frames = []
    for k in (1, 2, 3):
        df = read_tsv(os.path.join(data_dir, split, f'{split}_source{k}.tsv'))
        df = normalize_frame(df, workers).with_columns(pl.lit(k, dtype=pl.UInt8).alias('src'))
        frames.append(df)
        print(f'  [{split}] source{k}: {len(df):,} records normalised', flush=True)
    rec = pl.concat(frames).with_row_index('rid')
    rec = rec.with_columns(pl.col('rid').cast(pl.UInt32))
    rec.write_parquet(cache)
    return rec


def load_truth(data_dir, rec):
    """Ground-truth as (rid1, rid2) pairs on the dense row ids of `rec`."""
    gt = pl.read_csv(os.path.join(data_dir, 'train', 'train_ground_truth.tsv'), separator='\t',
                     quote_char=None, infer_schema=False)
    gt = (gt.with_columns(pl.col('matched_entity_ids').fill_null('').str.split(','))
            .explode('matched_entity_ids').filter(pl.col('matched_entity_ids') != ''))
    ids = rec.select('entity_id', 'rid')
    return (gt.join(ids.rename({'entity_id': 'source1_entity_id', 'rid': 'rid1'}), on='source1_entity_id')
              .join(ids.rename({'entity_id': 'matched_entity_ids', 'rid': 'rid2'}), on='matched_entity_ids')
              .select('rid1', 'rid2'))
