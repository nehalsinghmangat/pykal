#!/usr/bin/env python3
"""Fix double-escaped LaTeX in turtlebot notebook."""

import json

nb_path = "docs/source/notebooks/tutorial/theory_to_python/turtlebot_state_estimation.ipynb"

with open(nb_path, 'r') as f:
    nb = json.load(f)

def fix_double_escape(source):
    """Fix double-escaped backslashes in LaTeX."""
    if not source:
        return source

    result = []
    for line in source:
        # Replace double backslashes with single backslashes
        # But be careful: in Python strings, \\ is one backslash
        # In JSON, \\ is one backslash when loaded
        # So \\\\ in the loaded string means it was \\\\\\\\ in JSON (4 backslashes -> 2 in Python)
        # We want to convert that to \\ in Python (which is \\\\ in JSON)

        # The issue is that we have double-escaped LaTeX commands
        # For example: \\\\mod should be \\mod
        # In the Python string after json.load(), \\\\mod is actually 2 backslashes + mod
        # We want it to be 1 backslash + mod (which would be represented as \\mod in JSON)

        # Replace \\\\ with \\
        fixed_line = line.replace('\\\\', '\\')
        result.append(fixed_line)

    return result

# Process all markdown cells
for cell in nb['cells']:
    if cell['cell_type'] == 'markdown':
        source = cell.get('source', [])
        if source:
            # Check if any line has LaTeX
            has_latex = any('\\' in line for line in source)
            if has_latex:
                cell['source'] = fix_double_escape(source)

# Write back
with open(nb_path, 'w') as f:
    json.dump(nb, f, indent=1)

print("Fixed double-escaped LaTeX backslashes")
