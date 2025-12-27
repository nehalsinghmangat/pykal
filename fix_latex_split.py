#!/usr/bin/env python3
"""Fix LaTeX formatting - split $$ onto separate lines."""

import json
import re

nb_path = "docs/source/notebooks/tutorial/theory_to_python/turtlebot_state_estimation.ipynb"

with open(nb_path, 'r') as f:
    nb = json.load(f)

def fix_latex_formatting(source):
    """
    Fix LaTeX display math formatting.

    Convert: '$$math$$\n'
    To:      '$$\n', 'math\n', '$$\n'
    """
    if not source:
        return source

    result = []
    i = 0

    while i < len(source):
        line = source[i]

        # Check if line contains $$ with content (not just $$)
        if '$$' in line:
            stripped = line.strip()

            # Case 1: Line is just $$ (already correct)
            if stripped == '$$':
                # Ensure blank line before if needed
                if result and result[-1] != '\n':
                    result.append('\n')
                result.append('$$\n')
                i += 1
                continue

            # Case 2: Line has $$content$$ (needs splitting)
            if stripped.startswith('$$') and stripped.endswith('$$'):
                # Ensure blank line before
                if result and result[-1] != '\n':
                    result.append('\n')

                # Extract content between $$
                content = stripped[2:-2].strip()

                # Add as separate lines
                result.append('$$\n')
                result.append(content + '\n')
                result.append('$$\n')

                # Ensure blank line after
                if i + 1 < len(source) and source[i + 1] != '\n':
                    result.append('\n')

                i += 1
                continue

            # Case 3: Line starts with $$ but doesn't end with it
            if stripped.startswith('$$'):
                # Ensure blank line before
                if result and result[-1] != '\n':
                    result.append('\n')

                # Extract content after $$
                content = stripped[2:].strip()

                result.append('$$\n')
                if content:
                    result.append(content + '\n')

                i += 1

                # Continue until we find closing $$
                while i < len(source):
                    line = source[i]
                    if '$$' in line:
                        # This line has the closing $$
                        stripped = line.strip()
                        if stripped.endswith('$$'):
                            content = stripped[:-2].strip()
                            if content:
                                result.append(content + '\n')
                            result.append('$$\n')

                            # Ensure blank line after
                            if i + 1 < len(source) and source[i + 1] != '\n':
                                result.append('\n')
                        i += 1
                        break
                    else:
                        result.append(line)
                        i += 1
                continue

        result.append(line)
        i += 1

    return result

# Process all markdown cells
for cell in nb['cells']:
    if cell['cell_type'] == 'markdown':
        source = cell.get('source', [])
        if source and any('$$' in str(line) for line in source):
            cell['source'] = fix_latex_formatting(source)

# Write back
with open(nb_path, 'w') as f:
    json.dump(nb, f, indent=1)

print("Fixed LaTeX formatting - split $$ onto separate lines")
