#!/usr/bin/env python3
"""Add multi-CGH visualization cells to phase-generation notebook."""
import json

path = "d:/6.2谈话资料/约束分离/SLMTools-main - 副本/examples/phase-generation-methods-comparison-2024-8-22.ipynb"
with open(path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

cells = nb['cells']

# Find d8ea7362 index
insert_idx = None
for i, c in enumerate(cells):
    if 'd8ea7362' in c.get('id', ''):
        insert_idx = i + 1
        break

if insert_idx is None:
    print("ERROR: d8ea7362 not found")
    exit(1)

# Cells to insert
new_cells = [
    {
        "id": "fig_multi_out",
        "cell_type": "code",
        "metadata": {},
        "source": [
            "# csotMultiCGH 重建像\n",
            "fig_csot_multi_out = cl1(out_multi)\n"
        ],
        "outputs": [],
        "execution_count": None
    },
    {
        "id": "fig_multi_phase",
        "cell_type": "code",
        "metadata": {},
        "source": [
            "# csotMultiCGH 第1幅CGH相位\n",
            "fig_csot_multi_phase = cl2(CGHs_multi[1])\n"
        ],
        "outputs": [],
        "execution_count": None
    },
    {
        "id": "fig_multi_mraf_out",
        "cell_type": "code",
        "metadata": {},
        "source": [
            "# csotMultiCGH_MRAF 重建像\n",
            "fig_csot_mraf_out = cl1(out_multi_mraf)\n"
        ],
        "outputs": [],
        "execution_count": None
    },
    {
        "id": "fig_multi_mraf_phase",
        "cell_type": "code",
        "metadata": {},
        "source": [
            "# csotMultiCGH_MRAF 第1幅CGH相位\n",
            "fig_csot_mraf_phase = cl2(CGHs_mraf[1])\n"
        ],
        "outputs": [],
        "execution_count": None
    },
]

# Insert in reverse order
for c in reversed(new_cells):
    cells.insert(insert_idx, c)

print(f"Inserted {len(new_cells)} visualization cells after d8ea7362 (index {insert_idx})")

# Also clear outputs of cells that reference old single-CGH variables
for c in cells:
    # Clear RMS cell output
    if '0d1b883b' in c.get('id', ''):
        c['outputs'] = []
        c['execution_count'] = None
    # Clear efficiency cell output
    if '059b73a0' in c.get('id', ''):
        c['outputs'] = []
        c['execution_count'] = None
    # Clear visualization cell output
    if '388c6bb6' in c.get('id', ''):
        c['outputs'] = []
        c['execution_count'] = None
    # Clear multi-CGH cells output
    if 'csot_multi_001' in c.get('id', ''):
        c['outputs'] = []
        c['execution_count'] = None
    if 'csot_multi_mraf_001' in c.get('id', ''):
        c['outputs'] = []
        c['execution_count'] = None

with open(path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

# Verify
with open(path, 'r', encoding='utf-8') as f:
    v = json.load(f)
viz_cells = sum(1 for c in v['cells'] if 'fig_multi' in c.get('id', ''))
print(f"Verification: {viz_cells} multi-CGH viz cells, {len(v['cells'])} total cells")
print("DONE")
