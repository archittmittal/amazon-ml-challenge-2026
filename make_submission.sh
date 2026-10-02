#!/usr/bin/env bash
# Build <team_name>_submission.zip in the layout the challenge requires.
#   bash make_submission.sh <team_name>
set -euo pipefail
TEAM="${1:?usage: bash make_submission.sh <team_name>}"
cd "$(dirname "$0")"
for f in output/matching_results.tsv output/candidate_pairs.tsv Documentation_template.md \
         code/business_entity_resolution/README.md code/business_entity_resolution/requirements.txt; do
  [[ -f "$f" ]] || { echo "missing $f" >&2; exit 1; }
done
mkdir -p dist
ZIP="dist/${TEAM}_submission.zip"
rm -f "$ZIP"
zip -r -9 "$ZIP" output/matching_results.tsv output/candidate_pairs.tsv Documentation_template.md \
    code/business_entity_resolution -x '*/__pycache__/*' '*.pyc' '*.DS_Store'
unzip -l "$ZIP"
