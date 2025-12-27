#!/usr/bin/env python3
"""Fix LaTeX formatting in turtlebot notebook to have proper newlines around display math."""

import json
import re

nb_path = "docs/source/notebooks/tutorial/theory_to_python/turtlebot_state_estimation.ipynb"

with open(nb_path, 'r') as f:
    nb = json.load(f)

def fix_latex_formatting(source_lines):
    """Fix LaTeX display math formatting to have blank lines before and after $$."""
    if not source_lines:
        return source_lines

    # Join lines to work with the full text
    text = ''.join(source_lines)

    # Pattern to match display math blocks that don't have proper spacing
    # Match: (non-newline)($$)(content)($$)(non-newline)
    # Replace with: (non-newline)\n\n$$(content)$$\n\n(non-newline)

    # First, ensure there's a blank line before $$
    text = re.sub(r'([^\n])\n\$\$', r'\1\n\n$$', text)

    # Then, ensure there's a blank line after $$
    text = re.sub(r'\$\$\n([^\n])', r'$$\n\n\1', text)

    # Convert back to list of lines (preserve original line structure as much as possible)
    return [text]

# Process all markdown cells
for cell in nb['cells']:
    if cell['cell_type'] == 'markdown':
        source = cell.get('source', [])
        if source and any('$$' in line for line in source):
            cell['source'] = fix_latex_formatting(source)

# Write back
with open(nb_path, 'w') as f:
    json.dump(nb, f, indent=1)

print("Fixed LaTeX formatting in all markdown cells")
