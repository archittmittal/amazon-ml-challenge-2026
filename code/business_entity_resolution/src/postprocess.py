"""Post-processing: attach address-less records to the entity their exact-name twin belongs to.

A Source-2/3 record with NO address carries only its name. When that name is generic, the matcher
(rightly) refuses to link it on name alone. But if another record with exactly the same normalised
name, in the same country, was matched to exactly ONE Source-1 entity, the address-less record is
almost certainly another copy of that same business.

Measured on the labelled training set (truth used for the twins): 57,345 such records, rule
precision 0.991 (India 0.990, US 0.991); only 0.9% are decoys. Records already matched are left
untouched, and added ids are also appended to that entity's candidate list (this rule is the
last candidate-generation step, so the matched ids stay a subset of the candidates).

    python src/postprocess.py --test_dir <dataset>/test --in_dir output --out_dir output_pp
"""
import argparse
import os

import polars as pl


def read(path):
    return pl.read_csv(path, separator='\t', quote_char=None, infer_schema=False)


def explode(df, col):
    return (df.with_columns(pl.col(col).fill_null('').str.split(','))
              .explode(col).filter(pl.col(col) != '').rename({col: 'entity_id'}))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--test_dir', required=True)
    ap.add_argument('--in_dir', required=True, help='folder with matching_results.tsv and candidate_pairs.tsv')
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--min_name_len', type=int, default=8)
    args = ap.parse_args()

    m = read(os.path.join(args.in_dir, 'matching_results.tsv'))
    c = read(os.path.join(args.in_dir, 'candidate_pairs.tsv'))
    s23 = pl.concat([read(os.path.join(args.test_dir, f'test_source{k}.tsv')) for k in (2, 3)])
    key = lambda col: pl.col(col).fill_null('').str.to_lowercase().str.replace_all(r'[^a-z0-9]', '')
    s23 = s23.select('entity_id', 'country', key('business_name').alias('nk'), key('business_address').alias('ak'))

    mp = explode(m, 'matched_entity_ids')
    assigned = s23.join(mp, on='entity_id')
    twins = (assigned.filter(pl.col('nk') != '').group_by('country', 'nk')
                     .agg(pl.col('source1_entity_id').n_unique().alias('ns'),
                          pl.col('source1_entity_id').first().alias('source1_entity_id')))
    add = (s23.filter((pl.col('ak') == '') & (pl.col('nk').str.len_chars() >= args.min_name_len))
              .join(mp.select('entity_id'), on='entity_id', how='anti')              # not already matched
              .join(twins.filter(pl.col('ns') == 1), on=['country', 'nk'])
              .select('source1_entity_id', 'entity_id'))
    print(f'adding {add.height:,} address-less records to {add["source1_entity_id"].n_unique():,} entities')
    was_empty = m.filter(pl.col('matched_entity_ids').fill_null('') == '')['source1_entity_id']
    print(f'  of which {add.filter(pl.col("source1_entity_id").is_in(was_empty.implode())).height:,} '
          f'go to entities that were predicted EMPTY')

    def merge(df, col, extra):
        base = explode(df, col)
        allp = pl.concat([base, extra]).unique()
        g = allp.group_by('source1_entity_id').agg(pl.col('entity_id').sort().str.join(',').alias(col))
        return (df.select('source1_entity_id').join(g, on='source1_entity_id', how='left')
                  .with_columns(pl.col(col).fill_null('')))

    os.makedirs(args.out_dir, exist_ok=True)
    m2 = merge(m, 'matched_entity_ids', add)
    c2 = merge(c, 'candidate_entity_ids', add)
    m2.write_csv(os.path.join(args.out_dir, 'matching_results.tsv'), separator='\t', quote_style='never')
    c2.write_csv(os.path.join(args.out_dir, 'candidate_pairs.tsv'), separator='\t', quote_style='never')
    print(f'wrote {args.out_dir}/matching_results.tsv ({m2.height:,} rows) and candidate_pairs.tsv')


if __name__ == '__main__':
    main()
