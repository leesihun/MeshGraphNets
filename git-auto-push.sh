#!/usr/bin/env bash
set -euo pipefail

REMOTE="${REMOTE:-origin}"
BRANCH="$(git rev-parse --abbrev-ref HEAD)"

if [ "$BRANCH" = "HEAD" ]; then
    echo "Error: detached HEAD. Check out a branch before auto-pushing."
    exit 1
fi

if [ "$#" -gt 0 ]; then
    COMMIT_MESSAGE="$*"
else
    printf -v COMMIT_MESSAGE '%(%Y-%m-%d %H:%M:%S)T' -1
fi

echo "Staging changes..."
git add -A

if ! git diff --cached --quiet; then
    echo "Creating commit..."
    git commit -m "$COMMIT_MESSAGE"
else
    echo "No file changes to commit."
fi

echo "Fetching $REMOTE..."
git fetch "$REMOTE"

UPSTREAM="$(git rev-parse --abbrev-ref --symbolic-full-name '@{u}' 2>/dev/null || true)"
if [ -z "$UPSTREAM" ]; then
    if git show-ref --verify --quiet "refs/remotes/$REMOTE/$BRANCH"; then
        UPSTREAM="$REMOTE/$BRANCH"
    fi
    echo "No upstream configured. Setting upstream to $REMOTE/$BRANCH on push."
    PUSH_ARGS=(-u "$REMOTE" "$BRANCH")
else
    PUSH_ARGS=()
fi

if [ -n "$UPSTREAM" ]; then
    BEHIND="$(git rev-list --count "HEAD..$UPSTREAM")"
    if [ "$BEHIND" -gt 0 ]; then
        echo "Rebasing onto $UPSTREAM..."
        git rebase "$UPSTREAM"
    else
        echo "Already up to date with $UPSTREAM."
    fi
fi

push_with_timeout() {
    if [ -x /usr/bin/timeout ]; then
        /usr/bin/timeout 60 "$@"
    else
        "$@"
    fi
}

echo "Pushing to remote..."
if push_with_timeout git push "${PUSH_ARGS[@]}"; then
    echo "Successfully pushed to remote!"
else
    echo "Error: failed to push. Changes remain committed locally."
    exit 1
fi
