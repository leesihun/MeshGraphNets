#!/usr/bin/env bash
set -euo pipefail

REMOTE_URL=$(git remote get-url origin 2>/dev/null) || { echo "Error: no git remote 'origin' found."; exit 1; }
BRANCH=$(git rev-parse --abbrev-ref HEAD 2>/dev/null) || { echo "Error: not inside a git repository."; exit 1; }

echo "Remote: $REMOTE_URL"
echo "Branch: $BRANCH"

git clone --branch "$BRANCH" --single-branch "$REMOTE_URL" temp_repo
rsync -a --exclude='.git' temp_repo/ .
rm -rf temp_repo

echo "Pulled latest changes from $REMOTE_URL ($BRANCH)."
