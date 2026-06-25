import json, sys

with open(sys.argv[1], 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Maps cell id -> new source
fixes = {}

# Cell 2cfd4e17: csotGS 100 — needs [1] to extract phase from tuple
fixes['2cfd4e17'] = [
    "# =========== CSOT: 约束分离分组GS (新增) ===========\n",
    "# 统一类型为 Float64\n",
    "I2_f64 = LF{Modulus}(Float64.(I2.data), I2.L, I2.flambda)\n",
    "tgts_f64 = ntuple(6) do i\n",
    "    LF{Modulus}(Float64.(sqrt(targets[i]).data), targets[i].L, targets[i].flambda)\n",
    "end\n",
    "\n",
    "# 100 迭代\n",
    "Φcsotgs100 = ntuple(6) do i\n",
    "    result = csotGS(I2_f64, tgts_f64[i], 100, Φots[i]; n_groups=4, verbose=false)\n",
    "    result[1]\n",
    "end\n",
    "println(\"csotGS 100 done\")\n"
]

# Cell e7dd429d: csotGS 1000 — needs [1]
fixes['e7dd429d'] = [
    "# 900 迭代 (总计 1000)\n",
    "Φcsotgs1000 = ntuple(6) do i\n",
    "    result = csotGS(I2_f64, tgts_f64[i], 900, Φcsotgs100[i]; n_groups=4, verbose=false)\n",
    "    result[1]\n",
    "end\n",
    "println(\"csotGS 1000 done\")\n"
]

# Cell 7e4558e7: csotMRAF 100 — needs [1]
fixes['7e4558e7'] = [
    "# =========== CSOT: 约束分离分组MRAF (新增) ===========\n",
    "# 100 迭代\n",
    "Φcsotmraf100 = ntuple(6) do i\n",
    "    result = csotMRAF(I2_f64, tgts_f64[i], 100, Φots[i], rois2[i], ms[i]; n_groups=4, verbose=false)\n",
    "    result[1]\n",
    "end\n",
    "println(\"csotMRAF 100 done\")\n"
]

# Cell 04b2fd69: csotMRAF 1000 — needs [1]
fixes['04b2fd69'] = [
    "# 900 迭代 (总计 1000)\n",
    "Φcsotmraf1000 = ntuple(6) do i\n",
    "    result = csotMRAF(I2_f64, tgts_f64[i], 900, Φcsotmraf100[i], rois2[i], ms[i]; n_groups=4, verbose=false)\n",
    "    result[1]\n",
    "end\n",
    "println(\"csotMRAF 1000 done\")\n"
]

# Cell 181cc12a: CSOT visualization — use I2_f64
fixes['181cc12a'] = [
    "# =========== CSOT 可视化与评价 ===========\n",
    "println(\"=== CSOT GS 100 ===\")\n",
    "fig_csotgs100 = lookSix(Tuple(getStats(I2_f64, targets[i], Φcsotgs100[i], trois[i], rois2[i]) for i=1:6), ds)\n",
    "\n",
    "println(\"\\n=== CSOT GS 1000 ===\")\n",
    "fig_csotgs1000 = lookSix(Tuple(getStats(I2_f64, targets[i], Φcsotgs1000[i], trois[i], rois2[i]) for i=1:6), ds)\n",
    "\n",
    "println(\"\\n=== CSOT MRAF 1000 ===\")\n",
    "fig_csotmraf1000 = lookSix(Tuple(getStats(I2_f64, targets[i], Φcsotmraf1000[i], trois[i], rois2[i]) for i=1:6), ds)\n"
]

# Cell fe76b209: Summary table — use I2_f64 for CSOT methods
fixes['fe76b209'] = [
    "# ================================================================\n",
    "# 论文风格对比汇总: 所有方法的 SchroffError 和 Efficiency\n",
    "# ================================================================\n",
    "target_names = [\"Diadem\", \"Shuriken\", \"Fourgon\", \"OR Gate\", \"Squid\", \"Q-tip\"]\n",
    "method_names = [\"OT\", \"GS100\", \"GS1000\", \"MRAF100\", \"MRAF1000\", \"MRAF1000L\",\n",
    "                \"csotGS100\", \"csotGS1000\", \"csotMRAF1000\"]\n",
    "all_phases = [Φots, Φgs100, Φgs1000, Φmrafs100, Φmrafs1000, Φmrafs1000_loose,\n",
    "              Φcsotgs100, Φcsotgs1000, Φcsotmraf1000]\n",
    "\n",
    "println(\"=\"^90)\n",
    "println(\"  Full Comparison: SchroffError (ε) and Efficiency (η)\")\n",
    "println(\"=\"^90)\n",
    "\n",
    "for (ti, tname) in enumerate(target_names)\n",
    "    println(\"\\n--- $tname ---\")\n",
    "    println(rpad(\"Method\", 16), \" SchroffErr ε  \", \" Efficiency η\")\n",
    "    println(\"-\"^50)\n",
    "    for (mi, mname) in enumerate(method_names)\n",
    "        Φ = all_phases[mi][ti]\n",
    "        in_beam = mi <= 6 ? I2 : I2_f64  # CSOT use f64 input\n",
    "        ε = SchroffError(targets[ti], square(sft(in_beam * Φ)), 0.1)\n",
    "        η = boxEfficiency(square(sft(in_beam * Φ)), rois2[ti])\n",
    "        println(rpad(mname, 16), \" \", lpad(round(ε, digits=6), 12), \"  \", round(η*100, digits=2), \"%\")\n",
    "    end\n",
    "end\n"
]

# Also fix cell 1: add Pkg.activate
fixes['2a29b041-ff61-48da-8eac-91a87a3accac'] = [
    "# ★ 指定使用包含CSOT的SLMTools副本\n",
    "using Pkg\n",
    "Pkg.activate(\"d:/6.2谈话资料/约束分离/SLMTools-main - 副本\")\n",
    "\n",
    "using SLMTools\n",
    "using Images, FFTW, JLD2, LinearAlgebra, IJuliaBell, FileIO, DSP, Plots, FreeTypeAbstraction\n",
    "using SLMTools: wrap\n",
    "import Interpolations: Linear\n"
]

# Apply fixes
for cell in nb['cells']:
    cid = cell.get('id', '')
    if cid in fixes:
        cell['source'] = fixes[cid]
        print(f"Fixed cell {cid}\n")

with open(sys.argv[1], 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("Done!")
