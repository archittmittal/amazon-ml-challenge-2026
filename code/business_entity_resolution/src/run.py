"""End-to-end Business Entity Resolution pipeline (Kaggle-ready).

See README.md for full options, hardware requirements, and reproduction steps.

    python src/run.py                       # auto-detects the dataset under /kaggle/input
    python src/run.py --data_dir .../dataset --out_dir output --work_dir /tmp/ber
    python src/run.py --smoke               # fast train-only sanity run on a 10% slice

Stages
  1  canonicalise records (transliteration, de-obfuscation)        prepare.py / text.py
  2  conjunctive-key IDF blocking, S2/S3 -> S1                      blocking.py
  3  self-supervised token-equivalence mining + application         equivalence.py
  4  pairwise features                                              features.py
  5  stage-1 LightGBM, 2-fold cross-fitted by S1 entity             model.py
  6  collective context features from stage-1 OOF probabilities     context.py
  7  stage-2 LightGBM                                               model.py
  8  exclusivity + expected-F0.5 decoding (vs threshold, chosen OOF) decode.py
  9  write matching_results.tsv / candidate_pairs.tsv, validate
 10  (separate script) address-less exact-name twins                 postprocess.py
"""
import argparse
import gc
import glob
import importlib
import json
import os
import pickle
import subprocess
import sys
import time

# Polars allocates through jemalloc (symbols prefixed `_rjem_`). By default freed pages are kept
# mapped for reuse, so process memory barely drops after large intermediates are released and the
# next stage starts from an inflated baseline. Ask it to hand freed memory back to the OS promptly.
# Must be set before polars is imported; inherited by worker processes and the test-stage re-exec.
os.environ.setdefault('_RJEM_MALLOC_CONF', 'background_thread:true,dirty_decay_ms:1000,muzzy_decay_ms:0')


def _bootstrap():
    """Make sure recent-enough libraries exist (Kaggle images vary)."""
    need = {'polars': '1.10', 'rapidfuzz': '3.6', 'lightgbm': '4.0'}
    missing = []
    for pkg, lo in need.items():
        try:
            from importlib.metadata import version
            v = version(pkg)
            if tuple(int(x) for x in v.split('.')[:2]) < tuple(int(x) for x in lo.split('.')):
                missing.append(f'{pkg}>={lo}')
        except Exception:
            missing.append(f'{pkg}>={lo}')
    if missing:
        print('installing', missing, flush=True)
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', *missing])
        importlib.invalidate_caches()


_bootstrap()

import numpy as np          # noqa: E402
import polars as pl         # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import equivalence          # noqa: E402
from blocking import generate_candidates                                  # noqa: E402
from context import CONTEXT_FEATURES, context_features                    # noqa: E402
from decode import (exclusive, fit_entity_decoder, select_expected_f,  # noqa: E402
                    select_learned, select_threshold)
from features import PAIR_FEATURES, REC_COLS, add_idf_weights, build_matrix  # noqa: E402
from metrics import macro_f05                                              # noqa: E402
from model import (crossfit, has_gpu, importance, load_model, predict_avg,  # noqa: E402
                   resolve_backend, save)
from prepare import load_split, load_truth                                 # noqa: E402
from sibling import (TE_FEATURES, agreement_features, target_encode_apply,  # noqa: E402
                     target_encode_train)

T0 = time.time()


def log(msg):
    """[elapsed  process-RSS  machine-free] message. `free` is the real headroom: RSS also counts
    allocator-retained pages the kernel can reclaim, so free memory is the number to watch."""
    try:
        import psutil
        mem = (f'{psutil.Process().memory_info().rss / 1e9:5.1f}GB '
               f'free {psutil.virtual_memory().available / 1e9:4.1f}GB')
    except Exception:
        mem = ''
    print(f'[{(time.time() - T0) / 60:6.1f}m {mem}] {msg}', flush=True)


def find_data_dir():
    # Only ever glob well-known, small roots. Never fall back to '..'/'../..': on Kaggle those
    # resolve to broad, sometimes slow-to-enumerate filesystem trees and a recursive '**' glob
    # there can look exactly like a hang instead of failing fast.
    bases = ['/kaggle/input', '.']
    for base in bases:
        if not os.path.isdir(base):
            continue
        hits = glob.glob(os.path.join(base, '**', 'train', 'train_source1.tsv'), recursive=True)
        if hits:
            return os.path.dirname(os.path.dirname(os.path.abspath(hits[0])))
    mounted = sorted(os.listdir('/kaggle/input')) if os.path.isdir('/kaggle/input') else []
    raise FileNotFoundError(
        "Could not find 'train/train_source1.tsv' under /kaggle/input or the current directory.\n"
        f"Datasets currently attached under /kaggle/input: {mounted or '(none)'}\n"
        "Attach the student_resource data as a Kaggle Dataset (Notebook -> Add Input), or pass "
        "--data_dir explicitly pointing at the folder that contains train/ and test/.")


def reindex(rec, truth, keep):
    """Keep only `keep` record ids and renumber them densely (rid == row position is an invariant
    the whole pipeline relies on). Truth pairs whose S1 was dropped disappear; their S2/S3 records
    stay, now as unmatched decoys."""
    sub = rec.filter(pl.col('rid').is_in(keep.implode()))
    remap = pl.DataFrame({'rid': sub['rid'], 'new': pl.arange(0, sub.height, eager=True).cast(pl.UInt32)})
    sub = sub.join(remap, on='rid').drop('rid').rename({'new': 'rid'}).sort('rid')
    t = (truth.join(remap.rename({'rid': 'rid1', 'new': 'n1'}), on='rid1')
              .join(remap.rename({'rid': 'rid2', 'new': 'n2'}), on='rid2')
              .select(pl.col('n1').alias('rid1'), pl.col('n2').alias('rid2')))
    return sub, t


def smoke_slice(rec, truth, frac=0.1):
    """Train-only: keep ~frac of S1 entities, their matches, and frac of distractors."""
    s1 = rec.filter((pl.col('src') == 1) & (pl.col('rid').hash(5) % 100 < frac * 100))['rid']
    t = truth.filter(pl.col('rid1').is_in(s1.implode()))
    matched_any = truth['rid2']
    dis = rec.filter((pl.col('src') != 1) & ~pl.col('rid').is_in(matched_any.implode())
                     & (pl.col('rid').hash(5) % 100 < frac * 100))['rid']
    keep = pl.concat([s1, t['rid2'], dis]).unique().sort()
    return reindex(rec, truth.filter(pl.col('rid1').is_in(s1.implode())), keep)


def test_prior(data_dir):
    """Label-free: S2/S3 records per S1 entity in the TEST files, per country."""
    cnt = {}
    for k in (1, 2, 3):
        df = pl.read_csv(os.path.join(data_dir, 'test', f'test_source{k}.tsv'), separator='\t',
                         quote_char=None, infer_schema=False, columns=['country'])
        for c, n in df.with_columns(pl.col('country').fill_null('').str.strip_chars())['country'].value_counts().rows():
            cnt.setdefault(c, [0, 0])[0 if k == 1 else 1] += n
    return {c: v[1] / v[0] for c, v in cnt.items() if v[0] > 0}


def s1_dropout(rec, truth, data_dir, cap=0.5, sibling_weight=2.0):
    """Match the training decoy density to the test set's (prior-shift correction).

    Test decoys are mostly real businesses whose S1 record is absent (siblings, branches). Removing
    a fraction r of S1 entities from training turns their S2/S3 records into exactly such decoys.
    With m true matches and d decoys per S1 in train, and t S2/S3 records per S1 in test (assuming
    the same m in test), keeping (1-r) of S1 gives decoys per S1 of (d + m r)/(1 - r); setting that
    equal to t - m gives r = (t - m - d) / t. Countries absent from train (France) need nothing.
    """
    t_prior = test_prior(data_dir)
    s1 = rec.filter(pl.col('src') == 1).select('rid', 'country')
    n1 = dict(s1['country'].value_counts().rows())
    n23 = dict(rec.filter(pl.col('src') != 1)['country'].value_counts().rows())
    nm = dict(truth.join(s1.rename({'rid': 'rid1'}), on='rid1')['country'].value_counts().rows())
    # Test decoys are disproportionately *siblings* (same brand, nearby address): 44% of test pairs
    # sit in sibling clusters vs 33% in uniformly-dropped training. Preferentially dropping S1
    # entities that belong to a name family (same country + same two leading core-name tokens)
    # turns their records into sibling decoys, at the same overall drop rate r.
    if sibling_weight != 1.0 and 'n_core' in rec.columns:
        fam = rec.filter(pl.col('src') == 1).select(
            'rid', 'country', pl.col('n_core').fill_null('').str.split(' ').list.head(2).list.join(' ').alias('fk'))
        fam = fam.with_columns(pl.len().over('country', 'fk').alias('fsize'))
        s1 = s1.join(fam.select('rid', (pl.when(pl.col('fsize') >= 2).then(sibling_weight).otherwise(1.0))
                                .alias('wt')), on='rid', how='left').with_columns(pl.col('wt').fill_null(1.0))
    else:
        s1 = s1.with_columns(pl.lit(1.0).alias('wt'))
    drop = []
    report = {}
    for c, n in n1.items():
        if c not in t_prior or n == 0:
            continue
        m, d, t = nm.get(c, 0) / n, (n23.get(c, 0) - nm.get(c, 0)) / n, t_prior[c]
        r = min(max((t - m - d) / t, 0.0), cap)
        report[c] = (round(d, 3), round(t - m, 3), round(r, 3))
        if r > 0:
            sc = s1.filter(pl.col('country') == c)
            p = (r * pl.col('wt') / sc['wt'].mean()).clip(0.0, 0.95)
            drop.append(sc.filter(pl.col('rid').hash(17) % 10_000 < p * 10_000)['rid'])
    if not drop:
        return rec, truth, report
    dropped = pl.concat(drop)
    keep = rec.filter(~pl.col('rid').is_in(dropped.implode()))['rid']
    rec2, truth2 = reindex(rec, truth, keep)
    return rec2, truth2, report


def build_split(rec, args, tag, base_maps):
    """Blocking -> equivalence mining -> features. Returns cand, rec, X, maps."""
    cols = ['a_words', 'n_core', 'n_skel']
    # Pass 0 (label-free, 15% of queries, no re-ranking): mine surface-variant maps and apply them
    # BEFORE blocking. Previously maps were applied only after blocking, so transliteration misses
    # such as 'phaundeshan' vs 'foundation' could never become candidates.
    pre = generate_candidates(rec, args.cap, args.topk, args.min_ratio, args.block_chunk, log,
                              rerank=False, query_frac=0.15, workers=args.workers)
    mined0 = equivalence.mine(pre, rec, cols, min_count=8, log=log)
    del pre
    maps0 = equivalence.merge(base_maps, mined0) if base_maps else mined0
    rec = equivalence.apply(rec, maps0)
    gc.collect()
    cand = generate_candidates(rec, args.cap, args.topk, args.min_ratio, args.block_chunk, log,
                               rerank=not args.no_rerank, workers=args.workers)
    log(f'[{tag}] candidates: {cand.height:,} pairs for {cand["rid2"].n_unique():,} S2/S3 records')
    mined = equivalence.mine(cand, rec, cols, log=log)
    maps = equivalence.merge(maps0, mined)
    rec = equivalence.apply(rec, maps)
    rec, idfs = add_idf_weights(rec)
    cand = agreement_features(cand, rec)
    log(f'[{tag}] number-agreement features: {cand["ag_sib_sig"].mean():.3f} of pairs look like a '
        f'sibling cluster (records agreeing on a number that differs from S1\'s)')
    feat_rec = rec.select(REC_COLS)
    X = build_matrix(cand, feat_rec, idfs, os.path.join(args.work_dir, f'{tag}_X.npy'),
                     args.workers, args.feat_chunk, log)
    return cand, rec, X, maps


def decode_best(cand, p, truth, s1_ids):
    ex = exclusive(cand, p)
    results = {}
    results['expected_f'] = macro_f05(select_expected_f(ex), truth, s1_ids)[0]
    for t in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8):
        results[f'thr_{t}'] = macro_f05(select_threshold(ex, t), truth, s1_ids)[0]
    best = max(results, key=results.get)
    return best, results


def apply_decoder(cand, p, choice, edec=None):
    ex = exclusive(cand, p)
    if choice == 'learned':
        return select_learned(ex, edec)
    if choice == 'expected_f':
        return select_expected_f(ex)
    return select_threshold(ex, float(choice.split('_')[1]))


def error_breakdown(per, n):
    """Where the macro-F0.5 loss comes from, by kind of entity-level mistake."""
    cat = (pl.when((pl.col('nt') == 0) & (pl.col('np') > 0)).then(pl.lit('singleton_given_match'))
             .when((pl.col('nt') > 0) & (pl.col('np') == 0)).then(pl.lit('matches_predicted_empty'))
             .when((pl.col('nt') > 0) & (pl.col('tp') == 0)).then(pl.lit('all_predicted_wrong'))
             .when(pl.col('f') < 1).then(pl.lit('partial'))
             .otherwise(pl.lit('perfect')))
    return (per.with_columns(cat.alias('kind'))
               .group_by('kind').agg((pl.len() / n).round(5).alias('share'),
                                     ((1 - pl.col('f')).sum() / n).round(5).alias('F_loss'))
               .sort('F_loss', descending=True).rows())


def write_outputs(rec, cand, pred, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    ids = rec.select('rid', 'entity_id')
    s1 = rec.filter(pl.col('src') == 1).select('rid', 'entity_id').rename({'rid': 'rid1', 'entity_id': 'source1_entity_id'})

    def lists(pairs, colname):
        g = (pairs.join(ids.rename({'rid': 'rid2', 'entity_id': 'e2'}), on='rid2')
                  .group_by('rid1').agg(pl.col('e2').unique().sort().str.join(',').alias(colname)))
        return (s1.join(g, on='rid1', how='left').with_columns(pl.col(colname).fill_null(''))
                  .sort('rid1').select('source1_entity_id', colname))

    m = lists(pred.select('rid1', 'rid2'), 'matched_entity_ids')
    c = lists(cand.select('rid1', 'rid2'), 'candidate_entity_ids')
    mp, cp = os.path.join(out_dir, 'matching_results.tsv'), os.path.join(out_dir, 'candidate_pairs.tsv')
    m.write_csv(mp, separator='\t', quote_style='never')
    c.write_csv(cp, separator='\t', quote_style='never')
    n_nonempty = (m['matched_entity_ids'] != '').sum()
    log(f'wrote {mp} ({m.height:,} rows, {n_nonempty:,} with matches) and {cp}')
    return mp, cp


def run_validator(data_dir, mp, cp):
    cands = glob.glob(os.path.join(data_dir, '..', 'utils', 'validate_submission.py')) + \
        glob.glob('/kaggle/input/**/validate_submission.py', recursive=True)
    if not cands:
        log('validator script not found; skipped (outputs follow the documented format)')
        return
    r = subprocess.run([sys.executable, cands[0], '--matching', mp, '--candidate', cp,
                        '--test-dir', os.path.join(data_dir, 'test')], capture_output=True, text=True)
    log('validator: ' + (r.stdout + r.stderr).strip()[-2000:])


def main():
    log(f'start: cpu_count={os.cpu_count()}, cwd={os.getcwd()}')
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', default=None)
    ap.add_argument('--out_dir', default='/kaggle/working/output' if os.path.isdir('/kaggle/working') else 'output')
    ap.add_argument('--work_dir', default='/tmp/ber_work')
    ap.add_argument('--workers', type=int, default=os.cpu_count())
    ap.add_argument('--cap', type=int, default=400, help='max S1 doc-freq of a blocking key')
    ap.add_argument('--no_rerank', action='store_true', help='disable two-stage (string re-ranked) blocking')
    ap.add_argument('--no_sibling_dropout', action='store_true',
                    help='drop S1 uniformly instead of preferring name families (siblings)')
    ap.add_argument('--topk', type=int, default=10, help='max S1 candidates per S2/S3 record')
    ap.add_argument('--min_ratio', type=float, default=0.3, help='drop candidates below ratio*best score')
    ap.add_argument('--block_chunk', type=int, default=100_000)
    ap.add_argument('--feat_chunk', type=int, default=2_000_000)
    ap.add_argument('--train_frac', type=float, default=None,
                    help='fraction of train queries used to fit (default 1.0 on GPU XGBoost, else 0.35)')
    ap.add_argument('--gbm', choices=['auto', 'lgbm', 'xgb'], default='auto',
                    help='auto = XGBoost on CUDA if a GPU is attached, else LightGBM on CPU')
    ap.add_argument('--rounds', type=int, default=1500)
    ap.add_argument('--smoke', action='store_true', help='train-only run on a 10%% slice')
    ap.add_argument('--artifacts_dir', default=None,
                    help='where trained models/maps/encodings are saved (default: /kaggle/working/artifacts)')
    ap.add_argument('--no_prior_match', dest='match_test_prior', action='store_false',
                    help='disable S1 dropout (training decoy density matched to the test set)')
    ap.add_argument('--test_only', action='store_true',
                    help='skip training; load artifacts from --artifacts_dir (or any /kaggle/input/**/artifacts)')
    args = ap.parse_args()
    args.data_dir = args.data_dir or find_data_dir()
    args.gbm = resolve_backend(args.gbm)
    if args.train_frac is None:
        args.train_frac = 1.0 if (args.gbm == 'xgb' and has_gpu()) else 0.35
    if args.artifacts_dir is None:
        args.artifacts_dir = ('/kaggle/working/artifacts' if os.path.isdir('/kaggle/working') else 'artifacts')
    os.makedirs(args.work_dir, exist_ok=True)
    log(f'config: {vars(args)}')

    if args.test_only:
        test_stage(args, load_artifacts(args))
        return
    state = train_stage(args)
    if state is None:              # smoke run
        return
    # Re-exec as a fresh `--test_only` process: every byte the training stage held (including
    # memory the allocator keeps after `del`) goes back to the OS, so the test stage starts from
    # a clean baseline instead of ~2-5GB above it. Output continues in the same cell/log.
    del state
    log('training done; restarting as a fresh process for the test stage')
    argv = [a for a in sys.argv[1:] if a != '--test_only']
    argv += ['--test_only', '--artifacts_dir', args.artifacts_dir, '--data_dir', args.data_dir]
    sys.stdout.flush()
    sys.stderr.flush()
    os.execv(sys.executable, [sys.executable, os.path.abspath(__file__)] + argv)


def save_artifacts(args, state):
    """Persist everything the test stage needs, under /kaggle/working so a committed notebook
    version keeps it as output: if the test stage ever dies, `--test_only` resumes from here."""
    d = args.artifacts_dir
    os.makedirs(d, exist_ok=True)
    paths = []
    for i, m in enumerate(state['models1'] + state['models2']):
        stem = os.path.join(d, f'model_{i}')
        save(m, stem)
        paths.append(stem + ('.json' if args.gbm == 'xgb' else '.txt'))
    meta = {k: state.get(k) for k in ('maps', 'choice', 'enc', 'res1', 'res2', 'edec')}
    meta['model_files'] = [os.path.basename(p) for p in paths]
    meta['n1'] = len(state['models1'])
    with open(os.path.join(d, 'state.pkl'), 'wb') as f:
        pickle.dump(meta, f)
    log(f'artifacts saved to {d} ({len(paths)} models)')


def load_artifacts(args):
    d = args.artifacts_dir
    if not os.path.exists(os.path.join(d, 'state.pkl')):
        hits = glob.glob('/kaggle/input/**/artifacts/state.pkl', recursive=True)
        if not hits:
            raise FileNotFoundError('--test_only: no artifacts/state.pkl found; attach the previous '
                                    'run\'s output as an input or pass --artifacts_dir')
        d = os.path.dirname(hits[0])
    with open(os.path.join(d, 'state.pkl'), 'rb') as f:
        meta = pickle.load(f)
    models = [load_model(os.path.join(d, p)) for p in meta['model_files']]
    meta['models1'], meta['models2'] = models[:meta['n1']], models[meta['n1']:]
    log(f'loaded artifacts from {d}: decoder={meta["choice"]}, train OOF={meta["res2"][meta["choice"]]:.5f}')
    return meta


def train_stage(args):
    rec = load_split(args.data_dir, 'train', args.work_dir, args.workers)
    truth = load_truth(args.data_dir, rec)
    if args.smoke:
        rec, truth = smoke_slice(rec, truth)
        log(f'smoke slice: {rec.height:,} records, {truth.height:,} true pairs')
    if args.match_test_prior:
        n_before = truth.height
        rec, truth, rep = s1_dropout(rec, truth, args.data_dir,
                                     sibling_weight=1.0 if args.no_sibling_dropout else 2.0)
        log('S1 dropout to match test decoy density {country: (train decoys/S1, test decoys/S1, drop rate)}: '
            f'{rep}; true pairs {n_before:,} -> {truth.height:,}')
    rec = rec.drop('business_name', 'business_address')
    s1_ids = rec.filter(pl.col('src') == 1)['rid']
    cand, rec, X, maps = build_split(rec, args, 'train', None)
    cand = (cand.join(truth.with_columns(pl.lit(1, dtype=pl.UInt8).alias('y')), on=['rid1', 'rid2'], how='left')
                .with_columns(pl.col('y').fill_null(0)).sort('rid2', 'rid1'))
    y = cand['y'].to_numpy().astype(np.float32)
    ceiling = macro_f05(cand.filter(pl.col('y') == 1).select('rid1', 'rid2'), truth, s1_ids)[0]
    log(f'[train] blocking pair recall = {y.sum() / truth.height:.5f}; '
        f'F0.5 ceiling with a perfect matcher = {ceiling:.5f}; pairs/query = {cand.height / max(cand["rid2"].n_unique(), 1):.2f}')

    fold = (cand['rid1'].hash(3) % 2).to_numpy().astype(np.int8)
    qhash = cand['rid2'].hash(9).to_numpy()
    sample = (qhash % 1000) < args.train_frac * 1000
    T, enc = target_encode_train(cand, rec, y, fold, args.feat_chunk, log=log)
    gc.collect()
    get1 = lambda rows: np.hstack([X[rows], T[rows]])  # noqa: E731
    names1 = PAIR_FEATURES + TE_FEATURES
    log(f'stage-1 cross-fit ({len(names1)} features, backend={args.gbm}, train_frac={args.train_frac})')
    oof1, models1 = crossfit(get1, cand.height, y, fold, sample, qhash, names1, args.workers,
                             args.rounds, log, args.gbm)
    best1, res1 = decode_best(cand, oof1, truth, s1_ids)
    log(f'[train OOF] stage-1 macro F0.5: {json.dumps({k: round(v, 5) for k, v in res1.items()})}')

    C = context_features(cand, oof1, rec, args.workers, args.feat_chunk, log)
    get2 = lambda rows: np.hstack([X[rows], T[rows], C[rows]])  # noqa: E731
    names2 = names1 + CONTEXT_FEATURES
    log('stage-2 cross-fit')
    oof2, models2 = crossfit(get2, cand.height, y, fold, sample, qhash, names2, args.workers,
                             args.rounds, log, args.gbm)
    choice, res2 = decode_best(cand, oof2, truth, s1_ids)
    log(f'[train OOF] stage-2 macro F0.5: {json.dumps({k: round(v, 5) for k, v in res2.items()})}')
    # learned cluster-size decoder: always tried, adopted only if its OUT-OF-FOLD score beats the
    # best fixed decoder (it builds on the expected-F sizes)
    edec, pred = None, None
    model_, gain, pred_oof = fit_entity_decoder(exclusive(cand, oof2), truth, len(s1_ids), log, args.workers)
    if model_ is not None:
        res2['learned'] = macro_f05(pred_oof, truth, s1_ids)[0]
        if res2['learned'] > res2[choice]:
            choice, pred, edec = 'learned', pred_oof, model_
    log(f'>>> decoder chosen: {choice}  OOF macro F0.5 = {res2[choice]:.5f}')

    if pred is None:
        pred = apply_decoder(cand, oof2, choice)
    _, per = macro_f05(pred, truth, s1_ids)
    log('OOF loss by entity-level error kind [(kind, share of S1, F0.5 points lost)]: '
        + str(error_breakdown(per, per.height)))
    per = per.join(rec.select(pl.col('rid').alias('rid1'), 'country'), on='rid1')
    log('per-country OOF F0.5: ' + str(per.group_by('country').agg(pl.col('f').mean(), pl.len()).sort('country').rows()))
    tp = pred.join(truth, on=['rid1', 'rid2']).height
    log(f'pair precision={tp / max(pred.height, 1):.5f} recall={tp / truth.height:.5f}')
    imp = sorted(importance(models2[0], names2).items(), key=lambda t: -t[1])[:25]
    log('top stage-2 features (gain): ' + ', '.join(f'{k}:{v:.0f}' for k, v in imp))

    state = {'maps': maps, 'choice': choice, 'enc': enc, 'res1': res1, 'res2': res2,
             'models1': models1, 'models2': models2, 'edec': edec}
    save_artifacts(args, state)
    if args.smoke:
        log('smoke run complete (test skipped)')
        return None
    del X, T, C, cand, rec, oof1, oof2, y, fold, qhash, sample, pred, per, get1, get2
    os.remove(os.path.join(args.work_dir, 'train_X.npy'))
    gc.collect()
    return state


def test_stage(args, state):
    maps, enc, choice = state['maps'], state['enc'], state['choice']
    models1, models2 = state['models1'], state['models2']
    rec_t = load_split(args.data_dir, 'test', args.work_dir, args.workers)
    rec_t = rec_t.drop('business_name', 'business_address')
    cand_t, rec_t, X_t, _ = build_split(rec_t, args, 'test', maps)
    gc.collect()
    T_t = target_encode_apply(cand_t, rec_t, enc, args.feat_chunk)
    log('[test] stage-1 predict')
    p1 = predict_avg(models1, lambda rows: np.hstack([X_t[rows], T_t[rows]]), cand_t.height, log=log)
    C_t = context_features(cand_t, p1, rec_t, args.workers, args.feat_chunk, log)
    log('[test] stage-2 predict')
    p2 = predict_avg(models2, lambda rows: np.hstack([X_t[rows], T_t[rows], C_t[rows]]),
                     cand_t.height, log=log)
    pred_t = apply_decoder(cand_t, p2, choice, state.get('edec'))
    log(f'[test] predicted {pred_t.height:,} matches for {pred_t["rid1"].n_unique():,} S1 entities')
    mp, cp = write_outputs(rec_t, cand_t, pred_t, args.out_dir)
    run_validator(args.data_dir, mp, cp)
    log('done')


if __name__ == '__main__':
    main()
