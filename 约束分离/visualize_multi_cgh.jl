using Pkg
Pkg.activate("d:/6.2谈话资料/约束分离/SLMTools-main - 副本")
using SLMTools, Images, FFTW, Plots, Statistics

# 128x128 目标 "C"
N = 128
L = natlat((N, N))
I_in = LF{Intensity}(exp.(-r2(L) ./ (3.0^2)), L) |> normalizeLF
target = LF{Intensity}(imfilter(
    (lfRect(I_in, (4,4))*0.1 + lfText(Intensity, L, "C"; pixelsize=150)).data,
    Kernel.gaussian(2.0)), L) |> normalizeLF

U = sqrt(I_in)
V = sqrt(target)

# OT 初始相位 (小尺寸可以跑)
Φot = otPhase(I_in, target, 0.001)

println("=== 单 CGH: csotGS 900 迭代 ===")
Φ_single, _, _ = csotGS(U, V, 900, Φot; n_groups=9, verbose=false)
out_single = normalizeLF(square(sft(U * Φ_single)))
err_single = sqrt(sum((out_single.data .- normalizeLF(target).data).^2) / N^2)
println("  单CGH RMS: ", round(err_single, digits=8))

println("\n=== 多 CGH: csotMultiCGH 9组 × 100 迭代 ===")
CGHs, groups, errs = csotMultiCGH(U, V, 100, Φot; n_groups=9, verbose=false)
out_multi = reconstructMultiCGH(U, CGHs)
err_multi = sqrt(sum((out_multi.data .- normalizeLF(target).data).^2) / N^2)
println("  多CGH RMS: ", round(err_multi, digits=8))

# 可视化对比
roi = CartesianIndices((30:98, 40:88))
target_crop = look(target[roi])
single_crop = look(out_single[roi])
multi_crop = look(out_multi[roi])

println("\n=== 保存对比图 ===")
# 拼成一行: target | single | multi
using FileIO
hcat_imgs = hcat(target_crop, single_crop, multi_crop)
save("d:/6.2谈话资料/约束分离/comparison_single_vs_multi.png", hcat_imgs)
println("保存到 comparison_single_vs_multi.png")

# 数值对比
println("\n  Target  |  Single CGH  |  Multi CGH (9组时间复用)")
println("  RMS: $err_single  |  $err_multi")

# 局部方差对比 (衡量斑点噪声)
function local_std(img, window::Int=5)
    result = similar(img, Float64)
    pad = window ÷ 2
    for y in (1+pad):(size(img,1)-pad), x in (1+pad):(size(img,2)-pad)
        patch = img[y-pad:y+pad, x-pad:x+pad]
        result[y,x] = std(patch)
    end
    return result
end

roi_signal = target.data .> 0.01 * maximum(target.data)
speckle_single = std(out_single.data[roi_signal] ./ target.data[roi_signal])
speckle_multi  = std(out_multi.data[roi_signal] ./ target.data[roi_signal])
println("\n  斑点噪声 (信号区归一化标准差):")
println("    单CGH:   ", round(speckle_single, digits=6))
println("    多CGH:   ", round(speckle_multi, digits=6))
println("    抑制比:   ", round(speckle_single / speckle_multi, digits=2), "x")
