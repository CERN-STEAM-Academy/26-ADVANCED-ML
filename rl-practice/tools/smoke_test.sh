#!/usr/bin/env bash
# End-to-end rehearsal of the student experience, for a CERN Kubeflow notebook server.
#
# Run it from a JupyterLab terminal inside an already-cloned repository:
#
#     bash tools/smoke_test.sh            # checks + notebook 1     (~10 min)
#     bash tools/smoke_test.sh --full     # the above + notebook 2  (~30 min)
#
# It does what a student does, in order, and says PASS or FAIL for each step. Nothing is
# skipped silently: every step that cannot run says so and the summary at the end counts
# it as a failure.
#
# Overrides, mostly for testing this script itself:
#   PYTHON=...    interpreter to use              (default: python)
#   SKIP_PIP=1    assume dependencies are present
#   SKIP_DATA=1   assume assets/ is already there
#   ASSETS_URL=   where to fetch assets.tar.gz

set -uo pipefail

PYTHON="${PYTHON:-python}"
ASSETS_URL="${ASSETS_URL:-https://cernbox.cern.ch/s/QbQHtpgOSgkpCho/download}"
FULL=0
[ "${1:-}" = "--full" ] && FULL=1

cd "$(dirname "$0")/.." || exit 1
ROOT="$(pwd)"
export PYTHONPATH="$ROOT"
export TOKENIZERS_PARALLELISM=false

PASSED=0
FAILED=0
declare -a RESULTS

step() {
  local name="$1"; shift
  echo
  echo "──────────────────────────────────────────────────────────────────────"
  echo "▶ $name"
  echo "──────────────────────────────────────────────────────────────────────"
  local start; start=$(date +%s)
  if "$@"; then
    local secs=$(( $(date +%s) - start ))
    echo "  PASS  ($name, ${secs}s)"
    RESULTS+=("PASS  ${secs}s  $name")
    PASSED=$((PASSED + 1))
  else
    local secs=$(( $(date +%s) - start ))
    echo "  FAIL  ($name, ${secs}s)"
    RESULTS+=("FAIL  ${secs}s  $name")
    FAILED=$((FAILED + 1))
  fi
}

# --- 1. dependencies -----------------------------------------------------------------
install_deps() {
  [ "${SKIP_PIP:-0}" = "1" ] && { echo "  (skipped)"; return 0; }
  # Already satisfied? An image built for this session has the whole stack baked in, and
  # reinstalling it over a network filesystem is the slowest thing in the script.
  if $PYTHON -c "import torch, transformers, trl, peft, datasets, gymnasium" 2>/dev/null; then
    echo "  dependencies already present, nothing to install"
    return 0
  fi
  # -c constraints.txt is not optional: it stops pip replacing the image's CUDA build of
  # torch, which would fail much later and far less obviously.
  $PYTHON -m pip install --user -q -c constraints.txt -r requirements.txt
}

# --- 2. data -------------------------------------------------------------------------
fetch_assets() {
  [ "${SKIP_DATA:-0}" = "1" ] && { echo "  (skipped)"; return 0; }
  # Ask the resolver rather than looking in one hardcoded place. It searches the
  # repository, then RLPRACTICE_SHARED_DIR - which covers an unpacked tarball and an
  # image that baked the assets in and announced them. Only download if none of that
  # turned anything up.
  if $PYTHON -c "
import sys
sys.path.insert(0, '"'"'$ROOT'"'"')
from rlpractice import paths
found = paths.model_dir()
print('  assets already available:', found) if found else sys.exit(1)
" 2>/dev/null; then
    return 0
  fi
  echo "  downloading 782 MB from CERNBox..."
  # NOTE the URL form. The /index.php/s/<token>/download form redirects to a malformed
  # hostname (cernbox.cern.chs) and cannot be fetched by wget or curl.
  curl -sSL -o /tmp/assets.tar.gz "$ASSETS_URL" || return 1
  echo "  unpacking into $ROOT (so SHARED_DIR can stay None)..."
  tar xzf /tmp/assets.tar.gz -C "$ROOT" || return 1
  rm -f /tmp/assets.tar.gz
  [ -f assets/base_model/config.json ] || { echo "  base_model/config.json missing after unpack"; return 1; }
  echo "  assets: $(find assets -type f | wc -l) files, $(du -sh assets | cut -f1)"
}

# --- 3. environment ------------------------------------------------------------------
check_env() { $PYTHON tools/check_env.py; }

# --- 4. path resolution --------------------------------------------------------------
check_paths() {
  $PYTHON - <<'PY'
from rlpractice import paths
print(paths.describe())
assert paths.model_dir(), "base model not found - check assets/ or SHARED_DIR"
for name in ("reference_adapters", "reference_logs", "snapshots", "dqn"):
    assert paths.asset(name), f"{name} not found"
print("\nall artefacts resolved")
PY
}

# --- 5. tests ------------------------------------------------------------------------
run_tests() { $PYTHON -m pytest tests/ -q; }

# --- 6/7. notebooks ------------------------------------------------------------------
run_notebook() {
  local nb="$1" timeout_s="$2"
  ( cd notebooks && $PYTHON -m nbconvert --to notebook --execute \
      --ExecutePreprocessor.timeout="$timeout_s" \
      --output "/tmp/smoke_${nb}" "${nb}.ipynb" >/dev/null )
}

echo "======================================================================"
echo " RL practice - end-to-end smoke test"
echo " repository : $ROOT"
echo " python     : $($PYTHON -c 'import sys; print(sys.executable, sys.version.split()[0])' 2>/dev/null || echo "$PYTHON (not runnable)")"
echo " mode       : $([ $FULL = 1 ] && echo 'full (includes notebook 2)' || echo 'fast (skip notebook 2, pass --full to include it)')"
echo "======================================================================"

step "install dependencies"        install_deps
step "download and unpack assets"  fetch_assets
step "environment check"           check_env
step "artefact resolution"         check_paths
step "test suite"                  run_tests
step "notebook 1 (classics)"       run_notebook 01_classics_solutions 2400
if [ $FULL = 1 ]; then
  step "notebook 2 (GRPO)"         run_notebook 02_grpo_solutions 3600
else
  echo
  echo "  notebook 2 skipped - re-run with --full to include it (~17 min on a T4)"
fi

echo
echo "======================================================================"
echo " SUMMARY"
echo "======================================================================"
for line in "${RESULTS[@]}"; do echo "  $line"; done
echo
if [ $FAILED -eq 0 ]; then
  echo "  ALL $PASSED STEPS PASSED - the session is ready to run."
  exit 0
fi
echo "  $FAILED of $((PASSED + FAILED)) steps FAILED. See the output above."
exit 1
