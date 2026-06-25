import json, sys

nb_path = sys.argv[1]
with open(nb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

if 'phase-generation' in nb_path:
    # ==== phase-generation ====
    # 1. Insert csotMultiCGH_MRAF after csotMultiCGH (csmulti001)
    mraf_cell = {
        "id": "csmraf001",
        "cell_type": "code", "metadata": {},
        "source": [
            "# =========== CSOT: 约束分离多CGH MRAF 时间复用 ===========\n",
            "# N=3 → 9组, MRAF 混合参数 m=0.48\n",
            "CGHs_mraf, mraf_groups, mraf_errs = csotMultiCGH_MRAF(abs(I), sqrt(C), 1111, Φot,\n",
            "    CartesianIndices((17:112,17:112)), 0.48; n_groups=9, verbose=true)\n",
            "out_multi_mraf = reconstructMultiCGH(abs(I), CGHs_mraf; groups=mraf_groups)\n",
            "look(C, out_multi_mraf)\n"
        ],
        "outputs": [], "execution_count": None
    }
    for i, cell in enumerate(nb['cells']):
        if cell.get('id') == 'csmulti001':
            nb['cells'].insert(i + 1, mraf_cell)
            print("Inserted csotMultiCGH_MRAF\n")
            break

    # 2. Update RMS eval (0d1b883b) — add OT+csotMultiCGH_MRAF
    for cell in nb['cells']:
        if cell.get('id') == '0d1b883b-5591-4eb5-b03d-dbb351be6f9c':
            cell['source'] = [
                "# Evaluate RMS error for each phase\n",
                "for (name, Φ) in [(\"GS random\",Φgs), (\"OT\",Φot), (\"OT+GS\",Φotgs), (\"OT+MRAF\",Φotmraf),\n",
                '                    ("OT+csotGS",Φotcsotgs), ("OT+csotMRAF",Φotcsotmraf),\n',
                '                    ("OT+csotMultiCGH\",out_multi), (\"OT+csotMultiMRAF\",out_multi_mraf)]\n',
                '    err = Φ isa LF ? SchroffError(C, square(sft(abs(I) * Φ))) : SchroffError(C, Φ)\n',
                '    println(name, ": ", err)\n',
                "end\n"
            ]
            print("Updated RMS eval\n"); break

    # 3. Update efficiency (059b73a0) — add multi-MRAF
    for cell in nb['cells']:
        if cell.get('id') == '059b73a0-64a1-439e-9b8b-c87be2fa0655':
            cell['source'] = [
                "# Get efficiencies for all methods\n",
                "function boxEfficiency(F::LF{Intensity},roi::CartesianIndices)\n    sum(F[roi].data)/sum(F.data)\nend\n",
                "function boxEfficiency(F::LF{<:Amplitude},roi::CartesianIndices)\n    boxEfficiency(square(F),roi)\nend\n",
                "for (name, Φ) in [(\"GS random\",Φgs), (\"OT\",Φot), (\"OT+GS\",Φotgs), (\"OT+MRAF\",Φotmraf),\n",
                '                    ("OT+csotGS",Φotcsotgs), ("OT+csotMRAF",Φotcsotmraf),\n',
                '                    ("OT+csotMultiCGH",out_multi), (\"OT+csotMultiMRAF\",out_multi_mraf)]\n',
                '    η = Φ isa LF ? boxEfficiency(square(sft(abs(I) * Φ)), CartesianIndices((17:112,17:112))) :\n',
                '        boxEfficiency(Φ, CartesianIndices((17:112,17:112)))\n',
                '    println(name, ": ", η)\n',
                "end\n"
            ]
            print("Updated efficiency\n"); break

elif 'paper-OT' in nb_path:
    # ==== paper-OT ====
    # 1. Insert csotMultiCGH_MRAF after csotMultiCGH (csmulti002)
    mraf_cell = {
        "id": "csmraf002",
        "cell_type": "code", "metadata": {},
        "source": [
            "# =========== CSOT: 约束分离多CGH MRAF (论文方法) ===========\n",
            "# 9组 × 100 迭代, MRAF 混合参数复用 ms[i]\n",
            "Φcsotmulti_mraf = ntuple(6) do i\n",
            "    CGHs_i, groups_i, _ = csotMultiCGH_MRAF(I2_f64, tgts_f64[i], 100, Φots[i],\n",
            "        rois2[i], ms[i]; n_groups=9, verbose=false)\n",
            "    reconstructMultiCGH(I2_f64, CGHs_i; groups=groups_i)\n",
            "end\n",
            'println("csotMultiCGH MRAF done")\n'
        ],
        "outputs": [], "execution_count": None
    }
    for i, cell in enumerate(nb['cells']):
        if cell.get('id') == 'csmulti002':
            nb['cells'].insert(i + 1, mraf_cell)
            print("Inserted csotMultiCGH_MRAF\n"); break

    # 2. Add MRAF viz after multi-CGH viz (csmulti003)
    mraf_viz = {
        "id": "csmraf003",
        "cell_type": "code", "metadata": {},
        "source": [
            "# =========== CSOT 多CGH MRAF 可视化 ===========\n",
            'println("\\n=== CSOT MultiCGH MRAF (9组时间复用) ===")\n',
            "fig_csotmulti_mraf = lookSix(Tuple(getStats(I2_f64, targets[i], Φcsotmulti_mraf[i], trois[i], rois2[i]) for i=1:6), ds)\n"
        ],
        "outputs": [], "execution_count": None
    }
    for i, cell in enumerate(nb['cells']):
        if cell.get('id') == 'csmulti003':
            nb['cells'].insert(i + 1, mraf_viz)
            print("Inserted MRAF viz\n"); break

    # 3. Update summary (fe76b209) — add csotMultiMRAF
    for cell in nb['cells']:
        if cell.get('id') == 'fe76b209':
            cell['source'] = [
                "# ================================================================\n",
                "# 论文风格对比汇总: 所有方法的 SchroffError 和 Efficiency\n",
                "# ================================================================\n",
                'target_names = ["Diadem", "Shuriken", "Fourgon", "OR Gate", "Squid", "Q-tip"]\n',
                'method_names = ["OT", "GS100", "GS1000", "MRAF100", "MRAF1000", "MRAF1000L",\n',
                '                "csotGS100", "csotGS1000", "csotMRAF1000",\n',
                '                "csotMultiCGH", "csotMultiMRAF"]\n',
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
                "        ε = SchroffError(targets[ti], square(sft(in_beam * Φ)), 0.1)\n",
                "        η = boxEfficiency(square(sft(in_beam * Φ)), rois2[ti])\n",
                '        println(rpad(mname, 16), " ", lpad(round(ε, digits=6), 12), "  ", round(η*100, digits=2), "%")\n',
                "    end\n",
                "end\n"
            ]
            print("Updated summary\n"); break

    # 4. Update jldsave — add Φcsotmulti_mraf
    for cell in nb['cells']:
        if cell.get('id') == 'f4e5df2d-1729-4496-bbe7-dce9ba953f13':
            cell['source'] = [
                'jldsave("target_comparisons.jld2"; I2, targets, trois, rois2,\n',
                "    Φots, Φmrafs100, Φmrafs1000, Φgs100, Φgs1000, Φmrafs1000_loose,\n",
                "    Φcsotgs100, Φcsotgs1000, Φcsotmraf100, Φcsotmraf1000,\n",
                "    Φcsotmulti, Φcsotmulti_mraf);\n"
            ]
            print("Updated jldsave\n"); break

with open(nb_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
print("Done!")
