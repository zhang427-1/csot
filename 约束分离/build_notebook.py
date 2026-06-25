#!/usr/bin/env python3
"""从原始 phase-generation notebook 开始, 一次性加入所有 CSOT 方法"""
import json

path = "d:/6.2谈话资料/约束分离/SLMTools-main - 副本/examples/phase-generation-methods-comparison-2024-8-22.ipynb"
with open(path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

cells = nb['cells']

# --- Step 1: Fix cell 0 — add Pkg.activate ---
cells[0]['source'] = [
    "# ★ 指定使用包含CSOT的SLMTools副本\n",
    "using Pkg\n",
    'Pkg.activate("d:/6.2谈话资料/约束分离/SLMTools-main - 副本")\n',
    "\n",
    "using SLMTools, Images, FreeTypeAbstraction, FileIO, Plots\n",
    "using SLMTools: wrap\n"
]

# --- Step 2: Insert CSOT cells after OT+MRAF (cell index 8, id 43228aa5) ---
insert_at = None
for i, c in enumerate(cells):
    if c.get('id') == '43228aa5-b4bd-4f52-a663-c2ae3e97d09a':
        insert_at = i + 1
        break

new_cells = []

# csotGS cell
new_cells.append({
    "id": "csot_gs_001",
    "cell_type": "code", "metadata": {},
    "source": [
        "# =========== CSOT: 约束分离单CGH GS (N=2→4组) ===========\n",
        "Φotcsotgs, csot_groups, csot_errors = csotGS(abs(I), sqrt(C), 10000, Φot; n_groups=4, verbose=true)\n",
        "look(C, square(sft(I * Φotcsotgs)))\n"
    ],
    "outputs": [], "execution_count": None
})

# csotMRAF cell
new_cells.append({
    "id": "csot_mraf_001",
    "cell_type": "code", "metadata": {},
    "source": [
        "# =========== CSOT: 约束分离单CGH MRAF (N=2→4组) ===========\n",
        "Φotcsotmraf, csotmraf_groups, csotmraf_errors = csotMRAF(abs(I), sqrt(C), 10000, Φot, CartesianIndices((17:112,17:112)), 0.48; n_groups=4, verbose=true)\n",
        "look(C, square(sft(I * Φotcsotmraf)))\n"
    ],
    "outputs": [], "execution_count": None
})

# csotMultiCGH cell
new_cells.append({
    "id": "csot_multi_001",
    "cell_type": "code", "metadata": {},
    "source": [
        "# =========== CSOT: 约束分离多CGH GS 时间复用 (N=3→9组) ===========\n",
        "CGHs_multi, multi_groups, multi_errs = csotMultiCGH(abs(I), sqrt(C), 1111, Φot; n_groups=9, verbose=true)\n",
        "out_multi = reconstructMultiCGH(abs(I), CGHs_multi; groups=multi_groups)\n",
        "look(C, out_multi)\n"
    ],
    "outputs": [], "execution_count": None
})

# csotMultiCGH_MRAF cell
new_cells.append({
    "id": "csot_multi_mraf_001",
    "cell_type": "code", "metadata": {},
    "source": [
        "# =========== CSOT: 约束分离多CGH MRAF 时间复用 (N=3→9组) ===========\n",
        "CGHs_mraf, mraf_groups, mraf_errs = csotMultiCGH_MRAF(abs(I), sqrt(C), 1111, Φot,\n",
        "    CartesianIndices((17:112,17:112)), 0.48; n_groups=9, verbose=true)\n",
        "out_multi_mraf = reconstructMultiCGH(abs(I), CGHs_mraf; groups=mraf_groups)\n",
        "look(C, out_multi_mraf)\n"
    ],
    "outputs": [], "execution_count": None
})

for c in reversed(new_cells):
    cells.insert(insert_at, c)

# --- Step 3: Update RMS evaluation (original cell 9, id 0d1b883b) ---
for c in cells:
    if c.get('id') == '0d1b883b-5591-4eb5-b03d-dbb351be6f9c':
        c['source'] = [
            "# Evaluate RMS error for each phase — CSOT 多CGH 返回强度, 其他返回相位\n",
            "for (name, Φ) in [(\"GS random\",Φgs), (\"OT\",Φot), (\"OT+GS\",Φotgs), (\"OT+MRAF\",Φotmraf),\n",
            '                    ("OT+csotGS",Φotcsotgs), ("OT+csotMRAF",Φotcsotmraf),\n',
            '                    ("OT+csotMultiCGH",out_multi), ("OT+csotMultiMRAF",out_multi_mraf)]\n',
            '    err = Φ isa LF{<:Phase} ? SchroffError(C, square(sft(abs(I) * Φ))) : SchroffError(C, Φ)\n',
            '    println(name, ": ", err)\n',
            "end\n"
        ]

# --- Step 4: Update visualization (original cell 10, id 388c6bb6) ---
for c in cells:
    if c.get('id') == '388c6bb6-18d3-4212-845a-a8d0aea0cb3f':
        c['source'] = [
            "# Merge images — 8 methods\n",
            "phases = [Φgs, Φot, Φotgs, Φotmraf, Φotcsotgs, Φotcsotmraf]\n",
            "outbeams = [square(sft(abs(I) * Φ)) for Φ in phases]\n",
            "push!(outbeams, out_multi)       # csotMultiCGH 重建\n",
            "push!(outbeams, out_multi_mraf)   # csotMultiCGH_MRAF 重建\n",
            "anns = handAnnotate(look.([C outbeams... ; square(I) phases..., Φot]),\n",
            '    ("(a)","(j)","(b)","(k)","(c)","(l)","(d)","(m)","(e)","(n)","(f)","(o)","(g)","(p)","(h)","(q)","(i)","(r)"),12,(20,20))\n',
            "anns[:,1] = padadd.(anns[:,1],15,:r,1)\n",
            "fig = mergeStrict(anns; padright = 2, padbottom=2, fillval=1)\n"
        ]

# --- Step 5: Update efficiency (original cell 11, id 059b73a0) ---
for c in cells:
    if c.get('id') == '059b73a0-64a1-439e-9b8b-c87be2fa0655':
        c['source'] = [
            "# Get efficiencies for all methods\n",
            "function boxEfficiency(F::LF{Intensity},roi::CartesianIndices)\n",
            "    sum(F[roi].data)/sum(F.data)\n",
            "end\n",
            "function boxEfficiency(F::LF{<:Amplitude},roi::CartesianIndices)\n",
            "    boxEfficiency(square(F),roi)\n",
            "end\n",
            "for (name, Φ) in [(\"GS random\",Φgs), (\"OT\",Φot), (\"OT+GS\",Φotgs), (\"OT+MRAF\",Φotmraf),\n",
            '                    ("OT+csotGS",Φotcsotgs), ("OT+csotMRAF",Φotcsotmraf),\n',
            '                    ("OT+csotMultiCGH",out_multi), ("OT+csotMultiMRAF",out_multi_mraf)]\n',
            '    η = Φ isa LF{<:Phase} ? boxEfficiency(square(sft(abs(I) * Φ)), CartesianIndices((17:112,17:112))) :\n',
            '        boxEfficiency(Φ, CartesianIndices((17:112,17:112)))\n',
            '    println(name, ": ", η)\n',
            "end\n"
        ]

# Write
with open(path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

# Verify
with open(path, 'r', encoding='utf-8') as f:
    verify = json.load(f)
count = sum(1 for c in verify['cells'] for l in c['source'] if 'csot' in l.lower())
print(f"Done! {count} lines contain CSOT references across {len(verify['cells'])} cells")
