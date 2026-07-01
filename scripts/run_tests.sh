#!/usr/bin/env bash
# Run the NumPy-only test suite (no torch / no downloads required).
set -e
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD:$PYTHONPATH"
for f in tests/test_*.py; do
  echo "== $f =="
  python3 "$f"
done
echo "ALL TESTS PASSED"
