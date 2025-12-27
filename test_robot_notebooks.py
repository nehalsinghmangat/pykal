#!/usr/bin/env python3
import sys
from pathlib import Path
from test_notebooks import execute_notebook

TURTLEBOT_NOTEBOOKS = [
    "kf_turtlebot.ipynb",
    "ukf_turtlebot.ipynb",
    "pf_turtlebot.ipynb",
    "pid_turtlebot.ipynb",
    "lqr_turtlebot.ipynb",
    "mpc_turtlebot.ipynb",
]

CRAZYFLIE_NOTEBOOKS = [
    "kf_crazyflie.ipynb",
    "ukf_crazyflie.ipynb",
    "ai_ukf_crazyflie.ipynb",
    "pf_crazyflie.ipynb",
    "pid_crazyflie.ipynb",
    "lqr_crazyflie.ipynb",
    "mpc_crazyflie.ipynb",
]

NOTEBOOK_DIR = Path("docs/source/notebooks/algorithm_library")

def test_notebooks(notebooks, category):
    print(f"\n{'='*60}")
    print(f"TESTING {category.upper()} NOTEBOOKS")
    print('='*60)
    
    results = {}
    for nb_name in notebooks:
        nb_path = NOTEBOOK_DIR / nb_name
        print(f"\nTesting {nb_name}...", flush=True)
        result, msg = execute_notebook(nb_path, timeout=180)
        results[nb_name] = (result, msg)
        print(f"  {'✓ PASS' if result else '✗ FAIL'}")
        if not result and len(msg) < 500:
            print(f"  Error: {msg[:200]}")
    
    return results

# Test TurtleBot notebooks
tb_results = test_notebooks(TURTLEBOT_NOTEBOOKS, "TurtleBot")

# Test Crazyflie notebooks
cf_results = test_notebooks(CRAZYFLIE_NOTEBOOKS, "Crazyflie")

# Summary
print("\n" + "="*60)
print("SUMMARY")
print("="*60)

print("\nTurtleBot:")
tb_passed = sum(1 for r, _ in tb_results.values() if r)
for nb, (result, _) in tb_results.items():
    print(f"  {'✓' if result else '✗'} {nb}")
print(f"  {tb_passed}/{len(TURTLEBOT_NOTEBOOKS)} passed")

print("\nCrazyflie:")
cf_passed = sum(1 for r, _ in cf_results.values() if r)
for nb, (result, _) in cf_results.items():
    print(f"  {'✓' if result else '✗'} {nb}")
print(f"  {cf_passed}/{len(CRAZYFLIE_NOTEBOOKS)} passed")

total_passed = tb_passed + cf_passed
total = len(TURTLEBOT_NOTEBOOKS) + len(CRAZYFLIE_NOTEBOOKS)
print(f"\nOverall: {total_passed}/{total} passed ({100*total_passed/total:.1f}%)")

sys.exit(0 if total_passed == total else 1)
