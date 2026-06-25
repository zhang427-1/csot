using Pkg
Pkg.activate("d:/6.2谈话资料/约束分离/SLMTools-main - 副本")
using SLMTools, Images, FFTW, Statistics, FileIO

# 128×128 测试 — 用论文里类似的圆形+细节目标
N = 128
L = natlat((N, N))

# 目标: 字母 "C" + 低背景 (类似论文图3)
I_in = LF{Intensity}(exp.(-r2(L) ./ (4.0^2)), L) |> normalizeLF
target = LF{Intensity}(imfilter(
    (lfRect(I_in, (4,4))*0.1 + lfText(Intensity, L, "C"; pixelsize=150)).data,
    Kernel.gaussian(2.0)), L) |> normalizeLF

U = sqrt(I_in)
V = sqrt(target)
Φot = otPhase(I_in, target, 0.001)

# ROI for display
roi = CartesianIndices((28:100, 40:88))

# ============================================================
# 方法1: 无约束分离 (标准 GS)
# ============================================================
println("方法1: 标准 GS (无像素分离)...")
Φ_none = gs(U, V, 900, Φot)
out_none = normalizeLF(square(sft(U * Φ_none)))[roi]

# ============================================================
# 方法2: 约束分离单CGH (csotGS, N=3→9组)
# ============================================================
println("方法2: csotGS 单CGH (9组约束分离)...")
Φ_csot, _, _ = csotGS(U, V, 900, Φot; n_groups=9, verbose=false)
out_csot = normalizeLF(square(sft(U * Φ_csot)))[roi]

# ============================================================
# 方法3: 约束分离多CGH (论文方法, 9组×150iter=1350总)
# ============================================================
println("方法3: 多CGH 时间复用 (9组×150iter)...")
CGHs, multi_groups, _ = csotMultiCGH(U, V, 150, Φot; n_groups=9, verbose=false)
out_multi = reconstructMultiCGH(U, CGHs; groups=multi_groups)[roi]

# ============================================================
# 数值评估
# ============================================================
function evaluate(out, target_lf, tag)
    t = target_lf.data
    o = out.data
    sig = t .> 0.01 * maximum(t)
    rms = sqrt(mean((o[sig] .- t[sig]).^2))
    # Speckle contrast = std/mean of signal region ratios
    ratio = o[sig] ./ t[sig]
    sc = std(ratio) / mean(ratio)
    # PSNR-like metric (dB)
    mse = mean((o[sig] .- t[sig]).^2)
    psnr = 10 * log10(1.0 / mse)
    println(rpad(tag, 20), "RMS=", round(rms,digits=6),
            "  Speckle=", round(sc,digits=4), "  PSNR=", round(psnr,digits=2), "dB")
end

println("\n========== 结 果 对 比 ==========")
evaluate(out_none,  target[roi], "无像素分离 (GS)")
evaluate(out_csot,  target[roi], "约束分离单CGH")
evaluate(out_multi, target[roi], "约束分离多CGH")

# ============================================================
# 保存对比图: target | 无分离 | csotGS | 多CGH
# ============================================================
imgs = hcat(look(target[roi]), look(out_none), look(out_csot), look(out_multi))
save("d:/6.2谈话资料/约束分离/paper_comparison_result.png", imgs)
println("\n图已保存: Target | 无分离 | csotGS | 多CGH(论文)")
