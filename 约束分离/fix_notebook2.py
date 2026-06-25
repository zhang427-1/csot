import json, sys

with open(sys.argv[1], 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Find the OT cell (a2a6e1b6) and the dsTargets cell (fdb43905)
# Insert JLD2 cell before OT cell
jld2_cell = {
    "id": "91fa7ef9",
    "cell_type": "code",
    "metadata": {},
    "source": [
        "# ★ 从 JLD2 加载预计算结果 (绕过 OT marginals 兼容性问题)\n",
        "using JLD2\n",
        'jld_path = "d:/6.2谈话资料/约束分离/SLMTools-main - 副本/examples/target_comparisons.jld2"\n',
        "if isfile(jld_path)\n",
        '    jldopen(jld_path, "r") do f\n',
        '        _ots = f["Φots"];           global Φots   = Tuple(LF{ComplexPhase}(p.data,p.L,p.flambda) for p in _ots)\n',
        '        _m100 = f["Φmrafs100"];     global Φmrafs100  = Tuple(LF{ComplexPhase}(p.data,p.L,p.flambda) for p in _m100)\n',
        '        _m1000 = f["Φmrafs1000"];   global Φmrafs1000 = Tuple(LF{ComplexPhase}(p.data,p.L,p.flambda) for p in _m1000)\n',
        '        _g100 = f["Φgs100"];        global Φgs100   = Tuple(LF{ComplexPhase}(p.data,p.L,p.flambda) for p in _g100)\n',
        '        _g1000 = f["Φgs1000"];      global Φgs1000  = Tuple(LF{ComplexPhase}(p.data,p.L,p.flambda) for p in _g1000)\n',
        '        if haskey(f, "Φmrafs1000_loose")\n',
        '            _l = f["Φmrafs1000_loose"]; global Φmrafs1000_loose = Tuple(LF{ComplexPhase}(p.data,p.L,p.flambda) for p in _l)\n',
        "        end\n",
        "    end\n",
        '    println("✓ Loaded Φots, Φgs100/1000, Φmrafs100/1000 from JLD2")\n',
        "else\n",
        '    @warn "JLD2 file not found at $jld_path, will compute OT from scratch"\n',
        "end\n"
    ],
    "outputs": [],
    "execution_count": None
}

# Modify OT cell to be conditional
ot_source = [
    "# OT phases — 仅当 JLD2 未加载时才计算\n",
    "if !@isdefined(Φots)\n",
    "    Φots = ((upsample(otPhase(normalizeLF(downsample(square(I2)[roiIn2],8)), normalizeLF(dsTargets[i]), 0.001), I2.L, bc=Linear()) |> SLMTools.wrap for i=1:6)...,);\n",
    "else\n",
    '    println("Φots already loaded from JLD2, skipping OT computation.")\n',
    "end\n"
]

# Find indices
ot_idx = None
ds_idx = None
for i, cell in enumerate(nb['cells']):
    cid = cell.get('id', '')
    if cid == 'a2a6e1b6-d54d-4cea-9857-8be9c8797ef0':
        ot_idx = i
    if cid == 'fdb43905-3e85-44f1-855b-d33b63b016ef':
        ds_idx = i

# Insert JLD2 cell after dsTargets (before OT)
if ds_idx is not None:
    nb['cells'].insert(ds_idx + 1, jld2_cell)
    print(f"Inserted JLD2 cell after cell index {ds_idx}")

# Fix OT cell (index may have shifted)
for i, cell in enumerate(nb['cells']):
    if cell.get('id', '') == 'a2a6e1b6-d54d-4cea-9857-8be9c8797ef0':
        cell['source'] = ot_source
        print(f"Fixed OT cell at index {i}")
        break

# Also remove/comment out the old JLD2 cell if it exists
for cell in nb['cells']:
    if cell.get('id', '') == '58d8eba9':
        cell['source'] = [
            "# JLD2 已在前面加载, 此处跳过\n",
            '# println("Variables already loaded, skipping duplicate JLD2 load.")\n'
        ]
        print("Commented out old JLD2 cell")
        break

with open(sys.argv[1], 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("Done!")
