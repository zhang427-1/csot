using Pkg
Pkg.activate("d:/6.2谈话资料/约束分离/SLMTools-main - 副本")
using SLMTools, Images, FFTW, Statistics, FileIO

N = 128
L = natlat((N, N))
I_in = LF{Intensity}(exp.(-r2(L) ./ (3.0^2)), L) |> normalizeLF
target = LF{Intensity}(imfilter(
    (lfRect(I_in, (4,4))*0.1 + lfText(Intensity, L, "C"; pixelsize=150)).data,
    Kernel.gaussian(2.0)), L) |> normalizeLF

U = sqrt(I_in)
V = sqrt(target)
Φot = otPhase(I_in, target, 0.001)

roi = CartesianIndices((25:103, 38:90))
target_crop = target[roi]

# 总迭代预算公平对比: 都是 900 次 gsIter 调用
# 多CGH: 9组 × 100 = 900, 单CGH: 900
println("="^55)
println("公平对比: 同等迭代预算 (900 次 gsIter)")
println("="^55)

println("\n--- 单CGH: csotGS 900 iter ---")
Φ1, _, _ = csotGS(U, V, 900, Φot; n_groups=9, verbose=false)
out1 = normalizeLF(square(sft(U * Φ1)))[roi]

println("\n--- 多CGH: 9组 × 100 iter ---")
CGHs, _, _ = csotMultiCGH(U, V, 100, Φot; n_groups=9, verbose=false)
out2 = reconstructMultiCGH(U, CGHs)[roi]

# 再加一组: 多CGH 每组 500 iter (总计 4500)
println("\n--- 多CGH: 9组 × 500 iter ---")
CGHs3, _, _ = csotMultiCGH(U, V, 500, Φot; n_groups=9, verbose=false)
out3 = reconstructMultiCGH(U, CGHs3)[roi]

# 计算信号区的归一化方差 (speckle contrast)
function speckle_contrast(output, target_lf, roi_sig)
    t = target_lf.data[roi_sig]
    o = output.data[roi_sig]
    ratio = o ./ t
    ratio = ratio[t .> 0.05 * maximum(t)]  # 只统计信号区
    return std(ratio) / mean(ratio)
end

# 信号 ROI
sig_mask = target_crop.data .> 0.01 * maximum(target_crop.data)

println("\n" * "="^55)
println("结果汇总:")
println("="^55)
println(rpad("方法", 22), rpad("RMS", 14), "Speckle Contrast")
println("-"^50)

for (name, out) in [("单CGH 900iter", out1), ("多CGH 9×100", out2), ("多CGH 9×500", out3)]
    rms = sqrt(mean((out.data[sig_mask] .- target_crop.data[sig_mask]).^2))
    sc = speckle_contrast(out, target_crop, sig_mask)
    println(rpad(name, 22), rpad(round(rms, digits=8), 14), round(sc, digits=6))
end

# 保存对比图
imgs = hcat(look(target_crop), look(out1), look(out2), look(out3))
save("d:/6.2谈话资料/约束分离/comparison_v2.png", imgs)
println("\n图已保存: target | 单CGH | 多CGH100 | 多CGH500")
