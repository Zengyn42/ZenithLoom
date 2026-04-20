#!/bin/bash
set -e
export PYTHONPATH=$PYTHONPATH:.
python3 -m pytest test_tool/unit_tests/ -v 2>&1
