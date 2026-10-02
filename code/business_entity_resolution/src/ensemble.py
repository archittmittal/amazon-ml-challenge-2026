"""Gap-filling ensemble of two submissions.

The primary submission's decisions are never overridden. A Source-2/3 record is added to an
entity only when the primary run left that record unmatched entirely AND the secondary run
matched it to that entity. Records the primary assigned elsewhere are untouched (exclusivity
holds: every record still belongs to at most one entity).

    python src/ensemble.py --primary out_a --secondary out_b --out_dir out_ens
"""
import argparse
import os

import polars as pl


def read(path):
    return pl.read_csv(path, separator='\t', quote_char=None, infer_schema=False)


def explode(df, col):
    return (df.with_columns(pl.col(col).fill_null('').str.split(','))
              .explode(col).filter(pl.col(col) != '').rename({col: 'entity_id'}))


def merge(df, col, extra):
    allp = pl.concat([explode(df, col), extra]).unique()
    g = allp.group_by('source1_entity_id').agg(pl.col('entity_id').sort().str.join(',').alias(col))
    return (df.select('source1_entity_id').join(g, on='source1_entity_id', how='left')
              .with_columns(pl.col(col).fill_null('')))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--primary', required=True)
    ap.add_argument('--secondary', required=True)
    ap.add_argument('--out_dir', required=True)
    args = ap.parse_args()
    pm = read(os.path.join(args.primary, 'matching_results.tsv'))
    pc = read(os.path.join(args.primary, 'candidate_pairs.tsv'))
    sm = read(os.path.join(args.secondary, 'matching_results.tsv'))
    P, S = explode(pm, 'matched_entity_ids'), explode(sm, 'matched_entity_ids')
    agree = P.join(S, on=['source1_entity_id', 'entity_id']).height
    conflict = P.join(S, on='entity_id', suffix='_s').filter(
        pl.col('source1_entity_id') != pl.col('source1_entity_id_s')).height
    add = S.join(P.select('entity_id'), on='entity_id', how='anti')
    print(f'primary pairs {P.height:,} | secondary pairs {S.height:,} | agree {agree:,} '
          f'| same record, different entity {conflict:,} | gap-fill additions {add.height:,}')
    empty = pm.filter(pl.col('matched_entity_ids').fill_null('') == '')['source1_entity_id']
    print(f'  additions into entities the primary predicted EMPTY: '
          f'{add.filter(pl.col("source1_entity_id").is_in(empty.implode())).height:,}')
    os.makedirs(args.out_dir, exist_ok=True)
    merge(pm, 'matched_entity_ids', add).write_csv(os.path.join(args.out_dir, 'matching_results.tsv'),
                                                   separator='\t', quote_style='never')
    merge(pc, 'candidate_entity_ids', add).write_csv(os.path.join(args.out_dir, 'candidate_pairs.tsv'),
                                                     separator='\t', quote_style='never')
    print(f'wrote {args.out_dir}/')


if __name__ == '__main__':
    main()
