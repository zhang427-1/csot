import json, sys

with open(sys.argv[1], 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Fix csotMultiCGH cell (csmulti002): store reconstruction as LF{Intensity}
for cell in nb['cells']:
    if cell.get('id') == 'csmulti002':
        cell['source'] = [
            "# =========== CSOT: 约束分离多CGH 时间复用 ===========\n",
            "# 9组 × 100 迭代, 返回时间复用重建结果 (LF{Intensity})\n",
            "Φcsotmulti = ntuple(6) do i\n",
            "    CGHs_i, _, _ = csotMultiCGH(I2_f64, tgts_f64[i], 100, Φots[i]; n_groups=9, verbose=false)\n",
            "    reconstructMultiCGH(I2_f64, CGHs_i)  # → LF{Intensity}\n",
            "end\n",
            'println("csotMultiCGH done")\n'
        ]

    if cell.get('id') == 'csmraf002':
        cell['source'] = [
            "# =========== CSOT: 约束分离多CGH MRAF 时间复用 ===========\n",
            "# 9组 × 100 迭代, 返回重建结果 (LF{Intensity})\n",
            "Φcsotmulti_mraf = ntuple(6) do i\n",
            "    CGHs_i, groups_i, _ = csotMultiCGH_MRAF(I2_f64, tgts_f64[i], 100, Φots[i],\n",
            "        rois2[i], ms[i]; n_groups=9, verbose=false)\n",
            "    reconstructMultiCGH(I2_f64, CGHs_i; groups=groups_i)\n",
            "end\n",
            'println("csotMultiCGH MRAF done")\n'
        ]

    # Fix summary table: handle multi-CGH intensity results
    if cell.get('id') == 'fe76b209':
        cell['source'] = [
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
            "        # 多CGH方法: Φ 已是重建强度; 其他: 需计算 sft\n",
            "        out_intensity = mname in multi_methods ? Φ : square(sft((mi <= 6 ? I2 : I2_f64) * Φ))\n",
            "        ε = SchroffError(targets[ti], out_intensity, 0.1)\n",
            "        η = boxEfficiency(out_intensity, rois2[ti])\n",
            '        println(rpad(mname, 16), " ", lpad(round(ε, digits=6), 12), "  ", round(η*100, digits=2), "%")\n',
            "    end\n",
            "end\n"
        ]

with open(sys.argv[1], 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
print("Fixed!")
