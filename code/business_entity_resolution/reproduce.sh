#!/usr/bin/env bash
# End-to-end reproduction of output/matching_results.tsv and output/candidate_pairs.tsv.
#   bash reproduce.sh <path/to/student_resource> [extra run.py flags]
set -euo pipefail
SR="${1:?usage: bash reproduce.sh <path/to/student_resource> [run.py flags]}"
shift
HERE="$(cd "$(dirname "$0")" && pwd)"

python "$HERE/src/run.py" --data_dir "$SR/dataset" --out_dir output_raw --gbm lgbm "$@"
python "$HERE/src/postprocess.py" --test_dir "$SR/dataset/test" --in_dir output_raw --out_dir output
python "$SR/utils/validate_submission.py" --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv --test-dir "$SR/dataset/test"
