#!/bin/bash
set -e
cd "$(dirname "$0")/.."
python3 -m pytest test_tool/e2e_tests/test_engine.py -v 2>&1
