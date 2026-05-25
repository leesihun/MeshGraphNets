#!/usr/bin/env bash
set -u

REMOTE="${REMOTE:-origin}"
BRANCH="$(git rev-parse --abbrev-ref HEAD)"

if [ "$BRANCH" = "HEAD" ]; then
    echo "Error: detached HEAD. Check out a branch before auto-pushing."
    exit 1
fi

COMMIT_MESSAGE="${*:-$(date '+%Y-%m-%d %H:%M:%S')}"

echo "Staging changes..."
git add -A

if ! git diff --cached --quiet; then
    echo "Creating commit..."
    git commit -m "$COMMIT_MESSAGE"
else
    echo "No file changes to commit."
fi

if ! git rev-parse --abbrev-ref --symbolic-full-name '@{u}' >/dev/null 2>&1; then
    echo "No upstream configured. Setting upstream to $REMOTE/$BRANCH on push."
    PUSH_ARGS=(-u "$REMOTE" "$BRANCH")
else
    PUSH_ARGS=()
fi

echo "Fetching $REMOTE..."
git fetch "$REMOTE"

UPSTREAM="$(git rev-parse --abbrev-ref --symbolic-full-name '@{u}' 2>/dev/null || true)"
if [ -n "$UPSTREAM" ]; then
    BEHIND="$(git rev-list --count "HEAD..$UPSTREAM")"
    if [ "$BEHIND" -gt 0 ]; then
        echo "Rebasing onto $UPSTREAM..."
        git rebase "$UPSTREAM"
    else
        echo "Already up to date with $UPSTREAM."
    fi
fi

echo "Pushing to remote..."
if timeout 60 git push "${PUSH_ARGS[@]}" 2>&1; then
    echo "Successfully pushed to remote!"
else
    echo "Error: failed to push. Changes remain committed locally."
    exit 1
fi
