#!/usr/bin/env bash
set -euo pipefail

BLOG_ROOT="/home/fxlin/projects/fxlin1933.github.io"
cd "$BLOG_ROOT"

./scripts/update_daily_discovery.py

if ! git diff --quiet -- data/daily-discovery.json data/daily-discovery-last-run.json; then
  git add data/daily-discovery.json data/daily-discovery-last-run.json
  git commit -m "Update daily discovery"
  git push
fi
