# Amazon ML Challenge 2026: Business Entity Resolution

Team: Archit Mittal, Purvansh Joshi, Aviral Mittal

For each Source-1 business, the task is to find every Source-2/3 record that refers to the same real-world entity (US, India, and an unseen France). Submissions are scored by macro F0.5.

**Result:** train out-of-fold macro F0.5 = **0.9853**. The method is conjunctive-key blocking with string re-ranking, then a two-stage cross-fitted LightGBM (pairwise features, then collective features), then expected-F0.5 decoding for each entity.

The full methodology is in **[Documentation_template.md](Documentation_template.md)**.

## Repository layout (= submission zip layout)

```
output/
  matching_results.tsv        final matches (the leaderboard file)
  candidate_pairs.tsv         blocking candidate set (661 MB, attached to the GitHub Release)
code/business_entity_resolution/
  src/                        full pipeline source
  README.md                   exact reproduction steps and flags
  requirements.txt
  reproduce.sh                run.py -> postprocess.py -> validator
  logs/                       Kaggle logs behind the reported numbers
Documentation_template.md     methodology write-up
make_submission.sh            builds <team_name>_submission.zip
```

## Reproduce

```bash
cd code/business_entity_resolution
pip install -r requirements.txt
bash reproduce.sh /path/to/student_resource
```

The challenge dataset is not included in this repository. Download it from the challenge portal.

## Build the submission zip

```bash
bash make_submission.sh <team_name>    # -> dist/<team_name>_submission.zip
```

### Submission Verification Checklist
- [x] **`output/matching_results.tsv`**: Final matched entity predictions.
- [x] **`output/candidate_pairs.tsv`**: Candidate pair blocking set (~661 MB, attached to GitHub Release `v1.0-submission`).
- [x] **`code/business_entity_resolution/`**: Self-contained pipeline code with `src/`, `README.md`, and `requirements.txt`.
- [x] **`Documentation_template.md`**: Completed methodology write-up.

