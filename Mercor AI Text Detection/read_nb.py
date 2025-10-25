import nbformat
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8')

nb = nbformat.read(Path('0.9782 percent.ipynb'), as_version=4)

for idx, cell in enumerate(nb.cells):
    if cell.cell_type == 'code':
        print(f"\n### Cell {idx}")
        print(cell.source)
