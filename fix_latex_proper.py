#!/usr/bin/env python3
"""Fix LaTeX formatting to match car_cruise_control.ipynb format."""

import json

nb_path = "docs/source/notebooks/tutorial/theory_to_python/turtlebot_state_estimation.ipynb"

with open(nb_path, 'r') as f:
    nb = json.load(f)

def fix_latex_formatting(source):
    """
    Fix LaTeX display math formatting for Jupyter notebooks.

    Proper format:
    - Text line ending with :\n
    - Blank line \n
    - Opening $$\n
    - Math content
    - Closing $$\n
    - Blank line \n
    """
    if not source:
        return source

    # If source is a single string, split it properly
    if len(source) == 1 and '\n' in source[0]:
        text = source[0]
        source = [line + '\n' for line in text.split('\n')]
        # Remove trailing empty strings
        while source and source[-1] == '\n':
            source = source[:-1]
        if source and not source[-1].endswith('\n'):
            source[-1] = source[-1] + '\n'

    result = []
    i = 0

    while i < len(source):
        line = source[i]

        # Check if this line is opening $$
        if line.strip() == '$$':
            # Check if we need to add blank line before $$
            if result and result[-1] != '\n':
                result.append('\n')

            # Add opening $$
            result.append('$$\n')
            i += 1

            # Collect lines until closing $$
            math_lines = []
            while i < len(source):
                line = source[i]
                if line.strip() == '$$':
                    # Found closing $$
                    # Add the math content
                    result.extend(math_lines)
                    # Add closing $$
                    result.append('$$\n')
                    i += 1
                    # Ensure blank line after $$
                    if i < len(source) and source[i] != '\n':
                        result.append('\n')
                    break
                else:
                    math_lines.append(line)
                    i += 1
        else:
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

print("Fixed LaTeX formatting to match car_cruise_control.ipynb")
