import json, sys

path = "d:/6.2谈话资料/约束分离/SLMTools-main - 副本/examples/phase-generation-methods-comparison-2024-8-22.ipynb"
with open(path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

cells = nb['cells']

# Fix: Cell 15 (388c6bb6) — merge images 2行8列
for c in cells:
    if '388c6bb6' in c.get('id',''):
        c['source'] = [
            "# Merge images — 8 methods (6 single-CGH + 2 multi-CGH)\n",
            "phases = [Φgs, Φot, Φotgs, Φotmraf, Φotcsotgs, Φotcsotmraf]\n",
            "outbeams = [square(sft(abs(I) * Φ)) for Φ in phases]\n",
            "push!(outbeams, out_multi)       # csotMultiCGH 重建\n",
            "push!(outbeams, out_multi_mraf)   # csotMultiCGH_MRAF 重建\n",
            "# 1(C) + 8(outbeams) + 1(input) + 6(phases) = 16 images, 8 per row\n",
            "imgs = [C, outbeams..., square(I), phases...]\n",
            "anns = handAnnotate(look.(imgs),\n",
            '    ("(a)","(i)","(b)","(j)","(c)","(k)","(d)","(l)",\n',
            '     "(e)","(m)","(f)","(n)","(g)","(o)","(h)","(p)"),8,(20,20))\n',
            "anns[:,1] = padadd.(anns[:,1],15,:r,1)\n",
            "fig = mergeStrict(anns; padright = 2, padbottom=2, fillval=1)\n"
        ]
        print("Fixed cell 15\n")

# Fix: Cell 20 (f2348448) — colorized output, match outbeams[1:6] to phases
for c in cells:
    if 'f2348448' in c.get('id',''):
        c['source'] = [
            "cl1 = x -> colorize(look(x);cmap = cgrad([RGB(0,0,0),RGB(0.9,0,0),RGB(1,1,1)],256))\n",
            "cl2 = x -> colorize(look(x);cmap = cgrad([RGB(0,0,0),RGB(1,1,1)],256))\n",
            "t = (x->reshape(x,(1,length(x))))\n",
            "# First 6 outbeams match 6 phases for 2×6 layout\n",
            "subimgs = vcat( cl1.(outbeams[1:6]) |> t , cl2.(phases) |> t)\n",
            'println("Output beams + phases: ", size(subimgs))\n',
            "subimgs\n"
        ]
        print("Fixed cell 20\n")

# Insert after d8ea7362: fig7/fig8 for csotMultiCGH
fig78_cell = {
    "id": "fig78_multi",
    "cell_type": "code", "metadata": {},
    "source": [
        "# csotMultiCGH 重建像 + CGH相位\n",
        "fig7 = cl1(out_multi)\n",
        "fig8 = cl2(CGHs_multi[1])\n"
    ],
    "outputs": [], "execution_count": None
}

# Insert after fig78: fig9/fig10 for csotMultiCGH_MRAF
fig910_cell = {
    "id": "fig910_mraf",
    "cell_type": "code", "metadata": {},
    "source": [
        "# csotMultiCGH_MRAF 重建像 + CGH相位\n",
        "fig9 = cl1(out_multi_mraf)\n",
        "fig10 = cl2(CGHs_mraf[1])\n"
    ],
    "outputs": [], "execution_count": None
}

insert_idx = None
for i, c in enumerate(cells):
    if 'd8ea7362' in c.get('id',''):
        insert_idx = i + 1
        break

if insert_idx:
    cells.insert(insert_idx, fig78_cell)
    cells.insert(insert_idx + 1, fig910_cell)
    print(f"Inserted fig7-fig10 after cell index {insert_idx}")

# Write back
with open(path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

# Verify
with open(path, 'r', encoding='utf-8') as f:
    v = json.load(f)
counts = {'imgs': 0, 'outbeams[1:6]': 0, 'fig7': 0, 'fig9': 0}
for c in v['cells']:
    for l in c['source']:
        for key in counts:
            if key in l: counts[key] += 1
print(f"Verify: imgs={counts['imgs']}, outbeams[1:6]={counts['outbeams[1:6]']}, fig7={counts['fig7']}, fig9={counts['fig9']}")
print("DONE - Reopen notebook now")
