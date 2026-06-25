#!/usr/bin/env python3
"""Remove single-CGH CSOT methods from both notebooks, keep only multi-CGH."""
import json

# ============================================================
# Phase-generation notebook
# ============================================================
pg_path = "d:/6.2谈话资料/约束分离/SLMTools-main - 副本/examples/phase-generation-methods-comparison-2024-8-22.ipynb"
with open(pg_path, 'r', encoding='utf-8') as f:
    pg = json.load(f)

# Remove single-CGH cells
pg['cells'] = [c for c in pg['cells'] if c.get('id') not in ('csot_gs_001', 'csot_mraf_001')]

# Update RMS evaluation cell (0d1b883b)
for c in pg['cells']:
    if '0d1b883b' in c.get('id',''):
        c['source'] = [
            "# Evaluate RMS error for each phase — CSOT 多CGH 返回强度, 其他返回相位\n",
            "for (name, Φ) in [(\"GS random\",Φgs), (\"OT\",Φot), (\"OT+GS\",Φotgs), (\"OT+MRAF\",Φotmraf),\n",
            '                    ("OT+csotMultiCGH",out_multi), ("OT+csotMultiMRAF",out_multi_mraf)]\n',
            '    err = Φ isa LF{<:Phase} ? SchroffError(C, square(sft(abs(I) * Φ))) : SchroffError(C, Φ)\n',
            '    println(name, ": ", err)\n',
            "end\n"
        ]
        break

# Update visualization cell (388c6bb6) — 6 methods: 4 single + 2 multi-CGH
for c in pg['cells']:
    if '388c6bb6' in c.get('id',''):
        c['source'] = [
            "# Merge images — 6 methods (4 single-CGH + 2 multi-CGH)\n",
            "phases = [Φgs, Φot, Φotgs, Φotmraf]\n",
            "outbeams = [square(sft(abs(I) * Φ)) for Φ in phases]\n",
            "push!(outbeams, out_multi)       # csotMultiCGH 重建\n",
            "push!(outbeams, out_multi_mraf)   # csotMultiCGH_MRAF 重建\n",
            # C + 6 outbeams + I + 4 phases = 12 images, 6 per row
            "imgs = [C, outbeams..., square(I), phases...]\n",
            "anns = handAnnotate(look.(imgs),\n",
            '    ("(a)","(g)","(b)","(h)","(c)","(i)","(d)","(j)","(e)","(k)","(f)","(l)"),6,(20,20))\n',
            "anns[:,1] = padadd.(anns[:,1],15,:r,1)\n",
            "fig = mergeStrict(anns; padright = 2, padbottom=2, fillval=1)\n"
        ]
        break

# Update efficiency cell (059b73a0)
for c in pg['cells']:
    if '059b73a0' in c.get('id',''):
        c['source'] = [
            "# Get efficiencies for all methods\n",
            "function boxEfficiency(F::LF{Intensity},roi::CartesianIndices)\n",
            "    sum(F[roi].data)/sum(F.data)\n",
            "end\n",
            "function boxEfficiency(F::LF{<:Amplitude},roi::CartesianIndices)\n",
            "    boxEfficiency(square(F),roi)\n",
            "end\n",
            "for (name, Φ) in [(\"GS random\",Φgs), (\"OT\",Φot), (\"OT+GS\",Φotgs), (\"OT+MRAF\",Φotmraf),\n",
            '                    ("OT+csotMultiCGH",out_multi), ("OT+csotMultiMRAF",out_multi_mraf)]\n',
            '    η = Φ isa LF{<:Phase} ? boxEfficiency(square(sft(abs(I) * Φ)), CartesianIndices((17:112,17:112))) :\n',
            '        boxEfficiency(Φ, CartesianIndices((17:112,17:112)))\n',
            '    println(name, ": ", η)\n',
            "end\n"
        ]
        break

# Remove fig7-fig10 cells (fig78_multi, fig910_mraf)
pg['cells'] = [c for c in pg['cells'] if c.get('id') not in ('fig78_multi', 'fig910_mraf')]

# Update colorized output cell (f2348448) — only 4 outbeams
for c in pg['cells']:
    if 'f2348448' in c.get('id',''):
        c['source'] = [
            "cl1 = x -> colorize(look(x);cmap = cgrad([RGB(0,0,0),RGB(0.9,0,0),RGB(1,1,1)],256))\n",
            "cl2 = x -> colorize(look(x);cmap = cgrad([RGB(0,0,0),RGB(1,1,1)],256))\n",
            "t = (x->reshape(x,(1,length(x))))\n",
            "# 4 outbeams + 4 phases for 2×4 layout\n",
            "subimgs = vcat( cl1.(outbeams[1:4]) |> t , cl2.(phases) |> t)\n",
            'println("Output beams + phases: ", size(subimgs))\n',
            "subimgs\n"
        ]
        break

with open(pg_path, 'w', encoding='utf-8') as f:
    json.dump(pg, f, indent=1, ensure_ascii=False)

# Verify
with open(pg_path, 'r', encoding='utf-8') as f:
    v = json.load(f)
csot_refs = sum(1 for c in v['cells'] for l in c['source'] if 'csot' in l.lower())
single_cgh = sum(1 for c in v['cells'] if c.get('id') in ('csot_gs_001', 'csot_mraf_001'))
print(f"Phase-generation: {csot_refs} CSOT refs, {single_cgh} single-CGH cells (expect 0), {len(v['cells'])} total cells")

# ============================================================
# Paper-OT notebook
# ============================================================
ot_path = "d:/6.2谈话资料/约束分离/SLMTools-main - 副本/examples/paper-OT-initialization-comparison-final.ipynb"
with open(ot_path, 'r', encoding='utf-8') as f:
    ot = json.load(f)

# Remove single-CGH cells
remove_ids = ('csot_gs100_002', 'csot_gs1000_002', 'csot_mraf100_002', 'csot_mraf1000_002', 'csot_viz_002')
ot['cells'] = [c for c in ot['cells'] if c.get('id') not in remove_ids]

# Update summary cell (summary_002) — remove single-CGH methods
for c in ot['cells']:
    if 'summary_002' in c.get('id',''):
        c['source'] = [
            "# ================================================================\n",
            "# 论文风格对比汇总: 所有方法的 SchroffError 和 Efficiency\n",
            "# ================================================================\n",
            'target_names = ["Diadem", "Shuriken", "Fourgon", "OR Gate", "Squid", "Q-tip"]\n',
            'method_names = ["OT", "GS100", "GS1000", "MRAF100", "MRAF1000", "MRAF1000L",\n',
            '                "csotMultiCGH", "csotMultiMRAF"]\n',
            'multi_methods = Set(["csotMultiCGH", "csotMultiMRAF"])  # 已为重建强度\n',
            "all_phases = [Φots, Φgs100, Φgs1000, Φmrafs100, Φmrafs1000, Φmrafs1000_loose,\n",
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
        ]
        break

# Update jldsave (f4e5df2d) — remove single-CGH variables
for c in ot['cells']:
    if 'f4e5df2d' in c.get('id',''):
        c['source'] = [
            'jldsave("target_comparisons.jld2"; I2, targets, trois, rois2,\n',
            "    Φots, Φmrafs100, Φmrafs1000, Φgs100, Φgs1000, Φmrafs1000_loose,\n",
            "    Φcsotmulti, Φcsotmulti_mraf);\n"
        ]
        break

# Also update the csot_multiviz_002 cell to not reference single-CGH
for c in ot['cells']:
    if 'csot_multiviz_002' in c.get('id',''):
        c['source'] = [
            "# =========== CSOT 多CGH 可视化 ===========\n",
            'println("\\n=== CSOT MultiCGH GS ===")\n',
            "fig_csotmulti = lookSix(Tuple(getStats(I2_f64, targets[i], Φcsotmulti[i], trois[i], rois2[i]) for i=1:6), ds)\n",
            "\n",
            'println("\\n=== CSOT MultiCGH MRAF ===")\n',
            "fig_csotmulti_mraf = lookSix(Tuple(getStats(I2_f64, targets[i], Φcsotmulti_mraf[i], trois[i], rois2[i]) for i=1:6), ds)\n"
        ]
        break

with open(ot_path, 'w', encoding='utf-8') as f:
    json.dump(ot, f, indent=1, ensure_ascii=False)

# Verify
with open(ot_path, 'r', encoding='utf-8') as f:
    v = json.load(f)
csot_refs = sum(1 for c in v['cells'] for l in c['source'] if 'csot' in l.lower())
single_cells = sum(1 for c in v['cells'] if c.get('id') in remove_ids)
print(f"Paper-OT: {csot_refs} CSOT refs, {single_cells} single-CGH cells (expect 0), {len(v['cells'])} total cells")

print("\nDONE - Both notebooks updated.")
