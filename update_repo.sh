#!/bin/bash
set -e

# Simple helper to update this repo efficiently.
# Usage:
#   ./update_repo.sh "your commit message"

REPO_DIR="/home/zheyaogao/Experiments/ESD_seg"
cd "$REPO_DIR"

# Show current status
echo "===== Git status before update ====="
git status

# Stage all changes (respects .gitignore)
git add .

echo
if [ -z "$1" ]; then
  read -rp "Enter commit message: " MSG
else
  MSG="$1"
fi

if [ -z "$MSG" ]; then
  echo "No commit message provided, aborting."
  exit 1
fi

# Commit changes
git commit -m "$MSG" || echo "Nothing to commit (working tree clean)."

# Make sure we’re up to date with remote
echo "\n===== Pulling latest changes (rebase) ====="
git pull --rebase origin main || {
  echo "git pull --rebase failed. Resolve conflicts, then run the script again."
  exit 1
}

# Push to remote
echo "\n===== Pushing to origin/main ====="
git push origin main

echo "\nDone: repository is up to date."
