#!/usr/bin/env python3
"""Remove duplicate Experimentation section from turtlebot notebook."""

import json

nb_path = "docs/source/notebooks/tutorial/theory_to_python/turtlebot_state_estimation.ipynb"

with open(nb_path, 'r') as f:
    nb = json.load(f)

# Find and remove the duplicate experimentation cell (keep the first one)
cells = nb['cells']
new_cells = []
experimentation_count = 0

for cell in cells:
    source = ''.join(cell.get('source', []))

    # Check if this is an Experimentation section
    if cell['cell_type'] == 'markdown' and '## Experimentation' in source:
        experimentation_count += 1
        # Only keep the first one (which comes after callback wrapper at cell 723ac54b)
        if experimentation_count == 1:
            new_cells.append(cell)
        # Skip the duplicate
        continue
    else:
        new_cells.append(cell)

nb['cells'] = new_cells

with open(nb_path, 'w') as f:
    json.dump(nb, f, indent=1)

print(f"Removed duplicate Experimentation section")
print(f"Total cells: {len(new_cells)}")
