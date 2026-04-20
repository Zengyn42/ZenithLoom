#!/bin/bash
set -e
cd "$(dirname "$0")/.."
python3 -m pytest test_tool/unit_tests/ -v 2>&1 || python3 test_tool/unit_tests/test_*.py
