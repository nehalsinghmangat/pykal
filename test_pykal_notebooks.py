#!/usr/bin/env python3
import sys
from pathlib import Path
from test_notebooks import execute_notebook

PYKAL_NOTEBOOKS = [
    "pf_pykal.ipynb",
    "mpc_pykal.ipynb",
    "kf_pykal.ipynb",
    "ukf_pykal.ipynb",
    "ai_ukf_pykal.ipynb",
    "lqr_pykal.ipynb",
    "pid_pykal.ipynb",
]

NOTEBOOK_DIR = Path("docs/source/notebooks/algorithm_library")

results = {}
for nb_name in PYKAL_NOTEBOOKS:
    nb_path = NOTEBOOK_DIR / nb_name
    print(f"\nTesting {nb_name}...", flush=True)
    result, msg = execute_notebook(nb_path, timeout=180)
    results[nb_name] = result
    print(f"  {'✓ PASS' if result else '✗ FAIL'}")

print("\n" + "="*50)
print("PYKAL NOTEBOOKS SUMMARY")
print("="*50)
passed = sum(results.values())
total = len(results)
for nb, result in results.items():
    print(f"{'✓' if result else '✗'} {nb}")
print(f"\n{passed}/{total} passed ({100*passed/total:.1f}%)")
sys.exit(0 if passed == total else 1)
