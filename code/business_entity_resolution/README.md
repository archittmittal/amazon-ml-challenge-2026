# Business Entity Resolution: reproducible pipeline

This pipeline runs from raw data through blocking and matching to `output/matching_results.tsv` and `output/candidate_pairs.tsv`.

- It uses only the provided training/test TSVs: no external data, APIs, geocoders or pretrained models.
- The models are gradient-boosted trees: LightGBM (MIT) on CPU, or XGBoost (Apache-2.0) on a GPU.
- It runs inside a standard 4-core / 30 GB Kaggle notebook (lowest free RAM in the final run's test stage: ~13 GB).

## Reproduce the final submission

The submitted files came from **one `run.py` run (training + test inference) followed by `postprocess.py`**.
That run was on a Kaggle CPU notebook (4 cores, 30 GB) with all default flags: `cap=400`, two-stage re-ranked blocking, LightGBM, `train_frac=0.35`. Train and test ran in one notebook session (`run.py` re-execs itself for the test stage). The test stage alone took 303 min (`logs/run98_final_test_stage.log`).

```bash
pip install -r requirements.txt

# 1. train (2-fold cross-fit, stage 1 + stage 2) and predict the test set
python src/run.py --data_dir <student_resource>/dataset --out_dir output_raw --gbm lgbm

# 2. attach address-less exact-name twins (last candidate-generation step; see below)
python src/postprocess.py --test_dir <student_resource>/dataset/test --in_dir output_raw --out_dir output

# 3. official format check
python <student_resource>/utils/validate_submission.py \
    --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv \
    --test-dir <student_resource>/dataset/test
```

`reproduce.sh` runs the same three steps: `bash reproduce.sh <student_resource>`.

### On Kaggle

1. Upload `student_resource/` as a Kaggle Dataset. `run.py` finds `*/train/train_source1.tsv` anywhere under `/kaggle/input`.
2. Upload this folder as a second Dataset (for example `ber-code`).
3. In a notebook (Internet **on** so `_bootstrap()` can upgrade `polars`/`rapidfuzz` if the image is old), run:
   ```
   !python /kaggle/input/ber-code/src/run.py --smoke   # optional: 10% train slice, ~20 min, prints OOF F0.5
   !python /kaggle/input/ber-code/src/run.py           # full run -> /kaggle/working/output
   !python /kaggle/input/ber-code/src/postprocess.py --test_dir <...>/dataset/test \
           --in_dir /kaggle/working/output --out_dir /kaggle/working/final
   ```
   Trained models and state are saved to `/kaggle/working/artifacts`. If the test stage dies, you can resume it with `--test_only`.
   With a GPU attached, `--gbm auto` switches to XGBoost/CUDA and trains on 100% of pairs.

## What the log prints (train stage)

- blocking pair recall, and the **F0.5 ceiling** (the score a perfect matcher would get on these candidates)
- stage-1 and stage-2 out-of-fold macro F0.5, for expected-F decoding and for fixed thresholds 0.3–0.8
- the chosen decoder, an error breakdown, per-country OOF F0.5, pair precision/recall, and the top stage-2 features

`logs/` holds the actual Kaggle logs behind the submitted numbers.

## Flags

| flag | default | effect |
|---|---|---|
| `--cap` | 400 | Max Source-1 document frequency of a blocking key. Higher gives more recall but more pairs, time and memory. |
| `--no_rerank` | off | Disable two-stage blocking (string re-ranking of the top-50 key-score pool). |
| `--topk` | 10 | Max S1 candidates per S2/S3 record. |
| `--min_ratio` | 0.3 | Drop candidates below this fraction of the query's best key score. |
| `--gbm` | auto | `lgbm`, `xgb`, or `auto` (XGBoost on CUDA if a GPU is present, else LightGBM). |
| `--train_frac` | 1.0 on GPU, else 0.35 | Fraction of training queries used to fit each fold model. |
| `--rounds` | 1500 | Max boosting rounds (early stopping on a 4% query hold-out). |
| `--no_prior_match` | off | Disable S1 dropout (training decoy density matched to the test set). |
| `--no_sibling_dropout` | off | Make that dropout uniform instead of name-family weighted. |
| `--smoke` | off | Train-only run on a 10% entity slice. |
| `--test_only` | off | Skip training and load `--artifacts_dir` (or any `/kaggle/input/**/artifacts`). |
| `--block_chunk` / `--feat_chunk` | 100000 / 2000000 | Batch sizes; `memguard.py` also halves them automatically when free RAM drops below 6 GB. |
| `--work_dir` | /tmp/ber_work | Parquet caches and feature memmaps. |

## Layout

```
src/
  run.py           entry point: train stage (cross-fit) -> re-exec -> test stage -> outputs -> validator
  text.py          Brahmic transliteration (9 Indic scripts), Latin folding, de-obfuscation, phonetic skeleton
  prepare.py       parallel canonicalisation, parquet cache, ground-truth loading
  blocking.py      conjunctive-key IDF inverted index (S2/S3 -> S1) + string re-ranking
  equivalence.py   self-supervised token-equivalence mining (label-free; adapts to France)
  features.py      vectorised pairwise features (rapidfuzz cpdist + polars set ops)
  sibling.py       sibling discrimination: number agreement, out-of-fold residue-word target encoding
  context.py       stage-2 collective features (competition, anchors, entity coherence)
  model.py         LightGBM / XGBoost 2-fold cross-fitting grouped by Source-1 entity
  decode.py        exclusivity + expected-F0.5-optimal subset decoding (+ optional learned size decoder)
  metrics.py       challenge metric (macro F0.5 per S1 entity, singletons included)
  memguard.py      adaptive chunk sizing for 30 GB machines
  postprocess.py   final step: address-less records joined to their exact-name twin's entity
tools (not needed to reproduce the submission):
  diagnose.py      label-free per-country sanity checks of a submission against train statistics
  blocking_lab.py  blocking recall experiments on labelled train data
  ensemble.py      gap-fill merge of two submissions (evaluated; not used in the final submission)
logs/              Kaggle logs of the runs quoted in the documentation
```
