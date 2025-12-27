#!/usr/bin/env python3
"""
Test script for executing pykal algorithm library notebooks.

This script tests all notebooks can execute without errors, validating:
1. Algorithm implementations work correctly
2. Visualizations can be generated
3. Integration examples are syntactically correct
"""

import os
import sys
from pathlib import Path
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
from nbconvert.preprocessors import CellExecutionError
import time

# Notebook directory
NOTEBOOK_DIR = Path("docs/source/notebooks/algorithm_library")

# List of notebooks to test
NOTEBOOKS = [
    "pf_pykal.ipynb",
    "pf_turtlebot.ipynb",
    "pf_crazyflie.ipynb",
    "mpc_pykal.ipynb",
    "mpc_turtlebot.ipynb",
    "mpc_crazyflie.ipynb",
    "lqr_turtlebot.ipynb",
    "lqr_crazyflie.ipynb",
]

# Execution timeout per cell (in seconds)
CELL_TIMEOUT = 300


def execute_notebook(notebook_path: Path, timeout: int = CELL_TIMEOUT) -> tuple[bool, str]:
    """
    Execute a Jupyter notebook and return success status.

    Args:
        notebook_path: Path to the notebook file
        timeout: Maximum time per cell in seconds

    Returns:
        (success, message) tuple
    """
    print(f"\n{'='*80}")
    print(f"Testing: {notebook_path.name}")
    print(f"{'='*80}")

    try:
        # Read notebook
        with open(notebook_path) as f:
            nb = nbformat.read(f, as_version=4)

        # Insert matplotlib backend setup at the beginning
        # This ensures plots work in headless environment
        setup_cell = nbformat.v4.new_code_cell(
            source="import matplotlib\nmatplotlib.use('Agg')  # Use non-interactive backend for testing"
        )
        nb.cells.insert(0, setup_cell)

        # Create executor
        ep = ExecutePreprocessor(
            timeout=timeout,
            kernel_name='python3',
            allow_errors=False  # Stop on first error
        )

        # Execute notebook
        start_time = time.time()
        ep.preprocess(nb, {'metadata': {'path': notebook_path.parent}})
        elapsed_time = time.time() - start_time

        print(f"✓ SUCCESS: Executed in {elapsed_time:.2f}s")
        return True, f"Success ({elapsed_time:.2f}s)"

    except CellExecutionError as e:
        error_msg = str(e)
        print(f"✗ FAILED: {error_msg}")
        return False, error_msg

    except Exception as e:
        error_msg = f"Unexpected error: {str(e)}"
        print(f"✗ FAILED: {error_msg}")
        return False, error_msg


def main():
    """Run all notebook tests."""
    print("\n" + "="*80)
    print("PYKAL NOTEBOOK TEST SUITE")
    print("="*80)
    print(f"Testing {len(NOTEBOOKS)} notebooks from {NOTEBOOK_DIR}")
    print(f"Timeout per cell: {CELL_TIMEOUT}s")

    results = {}
    failed = []

    for notebook_name in NOTEBOOKS:
        notebook_path = NOTEBOOK_DIR / notebook_name

        if not notebook_path.exists():
            print(f"\n✗ SKIPPED: {notebook_name} (file not found)")
            results[notebook_name] = (False, "File not found")
            failed.append(notebook_name)
            continue

        success, message = execute_notebook(notebook_path)
        results[notebook_name] = (success, message)

        if not success:
            failed.append(notebook_name)

    # Print summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)

    for notebook_name, (success, message) in results.items():
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"{status}: {notebook_name}")
        if not success:
            print(f"       {message[:100]}...")

    # Print statistics
    total = len(NOTEBOOKS)
    passed = total - len(failed)
    pass_rate = (passed / total) * 100 if total > 0 else 0

    print(f"\n{passed}/{total} notebooks passed ({pass_rate:.1f}%)")

    if failed:
        print(f"\nFailed notebooks:")
        for notebook in failed:
            print(f"  - {notebook}")
        sys.exit(1)
    else:
        print("\n✓ All notebooks passed!")
        sys.exit(0)


if __name__ == "__main__":
    main()
