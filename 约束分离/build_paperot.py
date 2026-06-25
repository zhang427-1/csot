#!/usr/bin/env python3
"""从原始 paper-OT notebook 开始, 一次性加入所有 CSOT 方法"""
import json, copy

path = "d:/6.2谈话资料/约束分离/SLMTools-main - 副本/examples/paper-OT-initialization-comparison-final.ipynb"
with open(path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

cells = nb['cells']

def find_idx(cid_pattern):
    for i, c in enumerate(cells):
        if cid_pattern in c.get('id', ''):
            return i
    return None

def insert_after(cid_pattern, new_cell):
    idx = find_idx(cid_pattern)
    if idx is not None:
        cells.insert(idx + 1, new_cell)
        return True
    return False

def replace_cell(cid_pattern, new_source):
    for c in cells:
        if cid_pattern in c.get('id', ''):
            c['source'] = new_source
            return True
    return False

def mkcell(cid, source_lines):
    return {"id": cid, "cell_type": "code", "metadata": {},
            "source": [l + "\n" for l in source_lines if not l.endswith("\n")] +
                      [l for l in source_lines if l.endswith("\n")],
            "outputs": [], "execution_count": None}

# ===== 1. Fix cell 1: add Pkg.activate =====
replace_cell('2a29b041', [
    "# ★ 指定使用包含CSOT的SLMTools副本\n",
    "using Pkg\n",
    'Pkg.activate("d:/6.2谈话资料/约束分离/SLMTools-main - 副本")\n',
    "\n",
    "using SLMTools\n",
    "using Images, FFTW, JLD2, LinearAlgebra, IJuliaBell, FileIO, DSP, Plots, FreeTypeAbstraction\n",
    "using SLMTools: wrap\n",
    "import Interpolations: Linear\n"
])

# ===== 2. Insert JLD2 loading BEFORE OT (after cell 5 fdb43905) =====
insert_after('fdb43905', mkcell('jld2_load_001', [
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
]))

# ===== 3. Make OT cell conditional =====
replace_cell('a2a6e1b6', [
    "# OT phases — 仅当 JLD2 未加载时才计算\n",
    "if !@isdefined(Φots)\n",
    "    Φots = ((upsample(otPhase(normalizeLF(downsample(square(I2)[roiIn2],8)), normalizeLF(dsTargets[i]), 0.001), I2.L, bc=Linear()) |> SLMTools.wrap for i=1:6)...,);\n",
    "else\n",
    '    println("Φots already loaded from JLD2, skipping OT computation.")\n',
    "end\n"
])

# ===== 4. After Φgs1000 (2ec49b25), add CSOT cells =====
csot_cells = [
    mkcell('csot_gs100_002', [
        "# =========== CSOT: 约束分离单CGH GS 100 ===========\n",
        "# 统一类型为 Float64\n",
        "I2_f64 = LF{Modulus}(Float64.(I2.data), I2.L, I2.flambda)\n",
        "tgts_f64 = ntuple(6) do i\n",
        "    LF{Modulus}(Float64.(sqrt(targets[i]).data), targets[i].L, targets[i].flambda)\n",
        "end\n",
        "\n",
        "# 100 迭代\n",
        "Φcsotgs100 = ntuple(6) do i\n",
        "    result = csotGS(I2_f64, tgts_f64[i], 100, Φots[i]; n_groups=4, verbose=false)\n",
        "    result[1]  # 提取相位 LF{ComplexPhase}\n",
        "end\n",
        'println("csotGS 100 done")\n'
    ]),
    mkcell('csot_gs1000_002', [
        "# 900 迭代 (总计 1000)\n",
        "Φcsotgs1000 = ntuple(6) do i\n",
        "    result = csotGS(I2_f64, tgts_f64[i], 900, Φcsotgs100[i]; n_groups=4, verbose=false)\n",
        "    result[1]\n",
        "end\n",
        'println("csotGS 1000 done")\n'
    ]),
    mkcell('csot_mraf100_002', [
        "# =========== CSOT: 约束分离单CGH MRAF 100 ===========\n",
        "Φcsotmraf100 = ntuple(6) do i\n",
        "    result = csotMRAF(I2_f64, tgts_f64[i], 100, Φots[i], rois2[i], ms[i]; n_groups=4, verbose=false)\n",
        "    result[1]\n",
        "end\n",
        'println("csotMRAF 100 done")\n'
    ]),
    mkcell('csot_mraf1000_002', [
        "# 900 迭代 (总计 1000)\n",
        "Φcsotmraf1000 = ntuple(6) do i\n",
        "    result = csotMRAF(I2_f64, tgts_f64[i], 900, Φcsotmraf100[i], rois2[i], ms[i]; n_groups=4, verbose=false)\n",
        "    result[1]\n",
        "end\n",
        'println("csotMRAF 1000 done")\n'
    ]),
    mkcell('csot_multi_002', [
        "# =========== CSOT: 约束分离多CGH GS 时间复用 ===========\n",
        "# 9组 × 100 迭代\n",
        "Φcsotmulti = ntuple(6) do i\n",
        "    CGHs_i, groups_i, _ = csotMultiCGH(I2_f64, tgts_f64[i], 100, Φots[i]; n_groups=9, verbose=false)\n",
        "    reconstructMultiCGH(I2_f64, CGHs_i; groups=groups_i)\n",
        "end\n",
        'println("csotMultiCGH done")\n'
    ]),
    mkcell('csot_multi_mraf_002', [
        "# =========== CSOT: 约束分离多CGH MRAF 时间复用 ===========\n",
        "# 9组 × 100 迭代\n",
        "Φcsotmulti_mraf = ntuple(6) do i\n",
        "    CGHs_i, groups_i, _ = csotMultiCGH_MRAF(I2_f64, tgts_f64[i], 100, Φots[i],\n",
        "        rois2[i], ms[i]; n_groups=9, verbose=false)\n",
        "    reconstructMultiCGH(I2_f64, CGHs_i; groups=groups_i)\n",
        "end\n",
        'println("csotMultiCGH MRAF done")\n'
    ]),
]

# Insert after Φgs1000 (2ec49b25)
for c in reversed(csot_cells):
    insert_after('2ec49b25', c)

# ===== 5. Add CSOT visualization AFTER Φmrafs1000_loose viz (c06dd972) =====
csot_viz_cells = [
    mkcell('csot_viz_002', [
        "# =========== CSOT 单CGH 可视化 ===========\n",
        'println("\\n=== CSOT GS 100 ===")\n',
        "fig_csotgs100 = lookSix(Tuple(getStats(I2_f64, targets[i], Φcsotgs100[i], trois[i], rois2[i]) for i=1:6), ds)\n",
        "\n",
        'println("\\n=== CSOT GS 1000 ===")\n',
        "fig_csotgs1000 = lookSix(Tuple(getStats(I2_f64, targets[i], Φcsotgs1000[i], trois[i], rois2[i]) for i=1:6), ds)\n",
        "\n",
        'println("\\n=== CSOT MRAF 1000 ===")\n',
        "fig_csotmraf1000 = lookSix(Tuple(getStats(I2_f64, targets[i], Φcsotmraf1000[i], trois[i], rois2[i]) for i=1:6), ds)\n"
    ]),
    mkcell('csot_multiviz_002', [
        "# =========== CSOT 多CGH 可视化 ===========\n",
        'println("\\n=== CSOT MultiCGH GS ===")\n',
        "fig_csotmulti = lookSix(Tuple(getStats(I2_f64, targets[i], Φcsotmulti[i], trois[i], rois2[i]) for i=1:6), ds)\n",
        "\n",
        'println("\\n=== CSOT MultiCGH MRAF ===")\n',
        "fig_csotmulti_mraf = lookSix(Tuple(getStats(I2_f64, targets[i], Φcsotmulti_mraf[i], trois[i], rois2[i]) for i=1:6), ds)\n"
    ]),
]

for c in reversed(csot_viz_cells):
    insert_after('c06dd972', c)

# ===== 6. Add summary table BEFORE jldsave =====
summary_cell = mkcell('summary_002', [
    "# ================================================================\n",
    "# 论文风格对比汇总: 所有方法的 SchroffError 和 Efficiency\n",
    "# ================================================================\n",
    'target_names = ["Diadem", "Shuriken", "Fourgon", "OR Gate", "Squid", "Q-tip"]\n',
    'method_names = ["OT", "GS100", "GS1000", "MRAF100", "MRAF1000", "MRAF1000L",\n',
    '                "csotGS100", "csotGS1000", "csotMRAF1000",\n',
    '                "csotMultiCGH", "csotMultiMRAF"]\n',
    'multi_methods = Set(["csotMultiCGH", "csotMultiMRAF"])  # 已为重建强度\n',
    "all_phases = [Φots, Φgs100, Φgs1000, Φmrafs100, Φmrafs1000, Φmrafs1000_loose,\n",
    "              Φcsotgs100, Φcsotgs1000, Φcsotmraf1000,\n",
    "              Φcsotmulti, Φcsotmulti_mraf]\n",
    "\n",
    'println("="^90)\n',
    'println("  Full Comparison: SchroffError (ε) and Efficiency (η)")\n',
    'println("="^90)\n',
    "\n",
    "for (ti, tname) in enumerate(target_names)\n",
    '    println("\\n--- $tname ---")\n',
    '    println(rpad("Method", 16), " SchroffErr ε  ", " Efficiency η")\n',
    '    println("-"^50)\n',
    "    for (mi, mname) in enumerate(method_names)\n",
    "        Φ = all_phases[mi][ti]\n",
    "        in_beam = mi <= 6 ? I2 : I2_f64\n",
    "        out_intensity = mname in multi_methods ? Φ : square(sft(in_beam * Φ))\n",
    "        ε = SchroffError(targets[ti], out_intensity, 0.1)\n",
    "        η = boxEfficiency(out_intensity, rois2[ti])\n",
    '        println(rpad(mname, 16), " ", lpad(round(ε, digits=6), 12), "  ", round(η*100, digits=2), "%")\n',
    "    end\n",
    "end\n"
])

insert_after('f4e5df2d', summary_cell)

# ===== 7. Update jldsave =====
replace_cell('f4e5df2d', [
    'jldsave("target_comparisons.jld2"; I2, targets, trois, rois2,\n',
    "    Φots, Φmrafs100, Φmrafs1000, Φgs100, Φgs1000, Φmrafs1000_loose,\n",
    "    Φcsotgs100, Φcsotgs1000, Φcsotmraf100, Φcsotmraf1000,\n",
    "    Φcsotmulti, Φcsotmulti_mraf);\n"
])

# Write
with open(path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

# Verify
with open(path, 'r', encoding='utf-8') as f:
    v = json.load(f)
csot_lines = sum(1 for c in v['cells'] for l in c['source'] if 'csot' in l.lower())
print(f"Done! {csot_lines} CSOT lines, {len(v['cells'])} total cells")
