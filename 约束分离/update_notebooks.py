import json, copy, sys

nb_path = sys.argv[1]
with open(nb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# ==== Phase-generation notebook ====
if 'phase-generation' in nb_path:
    # 1. Insert csotMultiCGH cell after csotMRAF (a4025eb1)
    multi_cell = {
        "id": "csmulti001",
        "cell_type": "code",
        "metadata": {},
        "source": [
            "# =========== CSOT: 约束分离多CGH 时间复用 (论文方法) ===========\n",
            "# N=3 → 9组, 每组 10000÷9≈1111 迭代\n",
            "CGHs_multi, multi_groups, multi_errs = csotMultiCGH(abs(I), sqrt(C), 1111, Φot; n_groups=9, verbose=true)\n",
            "out_multi = reconstructMultiCGH(abs(I), CGHs_multi)\n",
            "look(C, out_multi)\n"
        ],
        "outputs": [],
        "execution_count": None
    }

    for i, cell in enumerate(nb['cells']):
        if cell.get('id') == 'a4025eb1':
            nb['cells'].insert(i + 1, multi_cell)
            print("Inserted csotMultiCGH after csotMRAF\n")
            break

    # 2. Update RMS evaluation (0d1b883b) — add multi-CGH
    for cell in nb['cells']:
        if cell.get('id') == '0d1b883b-5591-4eb5-b03d-dbb351be6f9c':
            cell['source'] = [
                "# Evaluate RMS error for each phase\n",
                "for (name, Φ) in [(\"GS random\",Φgs), (\"OT\",Φot), (\"OT+GS\",Φotgs), (\"OT+MRAF\",Φotmraf),\n",
                '                    ("OT+csotGS",Φotcsotgs), ("OT+csotMRAF",Φotcsotmraf), ("OT+csotMultiCGH\",out_multi)]\n',
                '    err = Φ isa LF ? SchroffError(C, square(sft(abs(I) * Φ))) : SchroffError(C, Φ)\n',
                '    println(name, ": ", err)\n',
                "end\n"
            ]
            print("Updated RMS evaluation\n")
            break

    # 3. Update visualization (388c6bb6) — add multi-CGH, 14 labels for 7 methods
    for cell in nb['cells']:
        if cell.get('id') == '388c6bb6-18d3-4212-845a-a8d0aea0cb3f':
            cell['source'] = [
                "# Merge images — 7 methods (original 4 + CSOT 3)\n",
                "phases = [Φgs, Φot, Φotgs, Φotmraf, Φotcsotgs, Φotcsotmraf]\n",
                "outbeams = [square(sft(abs(I) * Φ)) for Φ in phases]\n",
                "push!(outbeams, out_multi)  # 多CGH重建结果\n",
                "anns = handAnnotate(look.([C outbeams... ; square(I) phases..., Φot]),\n",
                '    ("(a)","(i)","(b)","(j)","(c)","(k)","(d)","(l)","(e)","(m)","(f)","(n)","(g)","(o)","(h)","(p)"),12,(20,20))\n',
                "anns[:,1] = padadd.(anns[:,1],15,:r,1)\n",
                "fig = mergeStrict(anns; padright = 2, padbottom=2, fillval=1)\n"
            ]
            print("Updated visualization\n")
            break

    # 4. Update efficiency (059b73a0) — add multi-CGH
    for cell in nb['cells']:
        if cell.get('id') == '059b73a0-64a1-439e-9b8b-c87be2fa0655':
            cell['source'] = [
                "# Get efficiencies for all methods\n",
                "function boxEfficiency(F::LF{Intensity},roi::CartesianIndices)\n",
                "    sum(F[roi].data)/sum(F.data)\n",
                "end\n",
                "function boxEfficiency(F::LF{<:Amplitude},roi::CartesianIndices)\n",
                "    boxEfficiency(square(F),roi)\n",
                "end\n",
                "for (name, Φ) in [(\"GS random\",Φgs), (\"OT\",Φot), (\"OT+GS\",Φotgs), (\"OT+MRAF\",Φotmraf),\n",
                '                    ("OT+csotGS",Φotcsotgs), ("OT+csotMRAF",Φotcsotmraf), ("OT+csotMultiCGH\",out_multi)]\n',
                '    η = Φ isa LF ? boxEfficiency(square(sft(abs(I) * Φ)), CartesianIndices((17:112,17:112))) : boxEfficiency(Φ, CartesianIndices((17:112,17:112)))\n',
                '    println(name, ": ", η)\n',
                "end\n"
            ]
            print("Updated efficiency\n")
            break

# ==== Paper-OT notebook ====
elif 'paper-OT' in nb_path:
    # 1. Insert csotMultiCGH cell after csotMRAF 1000 (04b2fd69)
    multi_cell = {
        "id": "csmulti002",
        "cell_type": "code",
        "metadata": {},
        "source": [
            "# =========== CSOT: 约束分离多CGH 时间复用 (论文方法) ===========\n",
            "# 9组 × 100 迭代 = 900 总迭代\n",
            "Φcsotmulti = ntuple(6) do i\n",
            "    CGHs_i, _, _ = csotMultiCGH(I2_f64, tgts_f64[i], 100, Φots[i]; n_groups=9, verbose=false)\n",
            "    reconstructMultiCGH(I2_f64, CGHs_i)\n",
            "end\n",
            'println("csotMultiCGH done")\n'
        ],
        "outputs": [],
        "execution_count": None
    }

    for i, cell in enumerate(nb['cells']):
        if cell.get('id') == '04b2fd69':
            nb['cells'].insert(i + 1, multi_cell)
            print("Inserted csotMultiCGH after csotMRAF 1000\n")
            break

    # 2. Add multi-CGH visualization after CSOT viz (181cc12a)
    multi_viz = {
        "id": "csmulti003",
        "cell_type": "code",
        "metadata": {},
        "source": [
            "# =========== CSOT 多CGH 可视化 ===========\n",
            'println("\\n=== CSOT MultiCGH (9组时间复用) ===")\n',
            "fig_csotmulti = lookSix(Tuple(getStats(I2_f64, targets[i], Φcsotmulti[i], trois[i], rois2[i]) for i=1:6), ds)\n"
        ],
        "outputs": [],
        "execution_count": None
    }

    for i, cell in enumerate(nb['cells']):
        if cell.get('id') == '181cc12a':
            nb['cells'].insert(i + 1, multi_viz)
            print("Inserted multi-CGH visualization\n")
            break

    # 3. Update summary table (fe76b209) — add csotMultiCGH column
    for cell in nb['cells']:
        if cell.get('id') == 'fe76b209':
            cell['source'] = [
                "# ================================================================\n",
                "# 论文风格对比汇总: 所有方法的 SchroffError 和 Efficiency\n",
                "# ================================================================\n",
                'target_names = ["Diadem", "Shuriken", "Fourgon", "OR Gate", "Squid", "Q-tip"]\n',
                'method_names = ["OT", "GS100", "GS1000", "MRAF100", "MRAF1000", "MRAF1000L",\n',
                '                "csotGS100", "csotGS1000", "csotMRAF1000", "csotMultiCGH"]\n',
                "all_phases = [Φots, Φgs100, Φgs1000, Φmrafs100, Φmrafs1000, Φmrafs1000_loose,\n",
                "              Φcsotgs100, Φcsotgs1000, Φcsotmraf1000, Φcsotmulti]\n",
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
                "        ε = SchroffError(targets[ti], square(sft(in_beam * Φ)), 0.1)\n",
                "        η = boxEfficiency(square(sft(in_beam * Φ)), rois2[ti])\n",
                '        println(rpad(mname, 16), " ", lpad(round(ε, digits=6), 12), "  ", round(η*100, digits=2), "%")\n',
                "    end\n",
                "end\n"
            ]
            print("Updated summary table\n")
            break

    # 4. Update jldsave (f4e5df2d) — add Φcsotmulti
    for cell in nb['cells']:
        if cell.get('id') == 'f4e5df2d-1729-4496-bbe7-dce9ba953f13':
            cell['source'] = [
                'jldsave("target_comparisons.jld2"; I2, targets, trois, rois2,\n',
                "    Φots, Φmrafs100, Φmrafs1000, Φgs100, Φgs1000, Φmrafs1000_loose,\n",
                "    Φcsotgs100, Φcsotgs1000, Φcsotmraf100, Φcsotmraf1000, Φcsotmulti);\n"
            ]
            print("Updated jldsave\n")
            break

with open(nb_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
print("Done!")
