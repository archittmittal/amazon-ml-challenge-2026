#!/usr/bin/env bash
# Build <team_name>_submission.zip in the layout the challenge requires.
# Usage: bash make_submission.sh <team_name>
set -euo pipefail
TEAM="${1:?usage: bash make_submission.sh <team_name>}"
cd "$(dirname "$0")"

echo "==> Validating required files for submission package..."
for f in output/matching_results.tsv output/candidate_pairs.tsv Documentation_template.md \
         code/business_entity_resolution/README.md code/business_entity_resolution/requirements.txt; do
  if [[ ! -f "$f" ]]; then
    echo "ERROR: Missing required file '$f'" >&2
    exit 1
  fi
done

echo "==> All required files present. Building submission package..."
mkdir -p dist
ZIP="dist/${TEAM}_submission.zip"
rm -f "$ZIP"
zip -r -9 "$ZIP" output/matching_results.tsv output/candidate_pairs.tsv Documentation_template.md \
    code/business_entity_resolution -x '*/__pycache__/*' '*.pyc' '*.DS_Store'

echo "==> Submission package successfully built at: $ZIP"
ls -lh "$ZIP"
unzip -l "$ZIP"
