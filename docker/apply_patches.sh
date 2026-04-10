#!/bin/bash
set -e

cd /build/vllm

echo "=== Fetching community PRs ==="

# PR #38891 - Per-layer attention backend for Gemma4 heterogeneous head dims
echo "Fetching PR #38891 (per-layer attention backend)..."
git fetch origin pull/38891/head:pr-38891 && git cherry-pick --no-commit pr-38891 || {
    echo "PR #38891: cherry-pick failed, trying merge..."
    git reset --hard HEAD
    git fetch origin pull/38891/head:pr-38891
    git merge --no-commit --no-ff pr-38891 || true
}

# PR #39084 - Fix Gemma4 NVFP4 expert scale suffix mapping
echo "Fetching PR #39084 (NVFP4 scale suffix fix)..."
git fetch origin pull/39084/head:pr-39084 && git cherry-pick --no-commit pr-39084 || {
    echo "PR #39084: cherry-pick failed, trying merge..."
    git reset --hard HEAD
    git fetch origin pull/39084/head:pr-39084
    git merge --no-commit --no-ff pr-39084 || true
}

# PR #39406 - Robust quantized MoE expert weight loading
echo "Fetching PR #39406 (robust MoE weight loading)..."
git fetch origin pull/39406/head:pr-39406 && git cherry-pick --no-commit pr-39406 || {
    echo "PR #39406: cherry-pick failed, trying merge..."
    git reset --hard HEAD
    git fetch origin pull/39406/head:pr-39406
    git merge --no-commit --no-ff pr-39406 || true
}

echo "=== PRs applied ==="
git diff --stat HEAD
