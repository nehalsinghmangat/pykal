#!/usr/bin/env python3
"""Fix LaTeX formatting for Jupyter notebooks - source as list of strings."""

import json
import re

nb_path = "docs/source/notebooks/tutorial/theory_to_python/turtlebot_state_estimation.ipynb"

with open(nb_path, 'r') as f:
    nb = json.load(f)

def fix_latex_formatting(source_lines):
    """Fix LaTeX display math formatting for Jupyter notebooks."""
    if not source_lines:
        return source_lines

    # If it's a single string, split it into lines
    if len(source_lines) == 1 and '\n' in source_lines[0]:
        text = source_lines[0]
        lines = text.split('\n')
        # Reconstruct as list of strings with \n at end of each line (except last)
        source_lines = [line + '\n' for line in lines[:-1]]
        if lines[-1]:  # Add last line if it's not empty
            source_lines.append(lines[-1])

    result = []
    i = 0
    while i < len(source_lines):
        line = source_lines[i]

        # Check if this line contains opening $$
        if '$$' in line and line.strip().startswith('$$'):
            # Check if previous line is blank
            if result and result[-1].strip() != '':
                result.append('\n')

            # Add the $$ line
            result.append(line)
            i += 1

            # Add content until closing $$
            while i < len(source_lines):
                line = source_lines[i]
                result.append(line)

                # Check if this line contains closing $$
                if '$$' in line and line.strip().endswith('$$\n') or line.strip() == '$$':
                    i += 1
                    # Ensure blank line after $$
                    if i < len(source_lines) and source_lines[i].strip() != '':
                        result.append('\n')
                    break
                i += 1
        else:
            result.append(line)
            i += 1

    return result

# Process all markdown cells
for cell in nb['cells']:
    if cell['cell_type'] == 'markdown':
        source = cell.get('source', [])
        if source and any('$$' in ''.join(source)):
            cell['source'] = fix_latex_formatting(source)

# Write back
with open(nb_path, 'w') as f:
    json.dump(nb, f, indent=1)

print("Fixed LaTeX formatting for Jupyter notebooks")
