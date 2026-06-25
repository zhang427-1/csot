using Pkg
Pkg.activate("d:/6.2谈话资料/约束分离/SLMTools-main - 副本")
using SLMTools, Images, FFTW

# 64x64 测试
N = 64
L = natlat((N, N))
I_in = LF{Intensity}(exp.(-r2(L) ./ (3.0^2)), L) |> normalizeLF
target = LF{Intensity}(imfilter(
    (lfRect(I_in, (20,20))*0.1 + lfText(Intensity, L, "C"; pixelsize=80)).data,
    Kernel.gaussian(2.0)), L) |> normalizeLF

U = sqrt(I_in)
V = sqrt(target)
Φ0 = LF{ComplexPhase}(exp.(2pi * im * rand(N, N)), L, 1.0)

# 多CGH生成: N=3 → 9组, 每组30迭代
println("=== csotMultiCGH: 9组, 每组30迭代 ===")
CGHs, groups, errs = csotMultiCGH(U, V, 30, Φ0; n_groups=9, verbose=true)

println("\n生成了 ", length(CGHs), " 个CGH")
recon = reconstructMultiCGH(U, CGHs; groups=groups)
println("重建完成")

# 对比: 单CGH (csotGS)
println("\n=== 对比: csotGS 单CGH ===")
Φ_single, _, _ = csotGS(U, V, 30*9, Φ0; n_groups=9, verbose=false)
recon_single = normalizeLF(square(sft(U * Φ_single)))

# RMS 对比
rms_multi = sqrt(sum((recon.data .- normalizeLF(target).data).^2) / N^2)
rms_single = sqrt(sum((recon_single.data .- normalizeLF(target).data).^2) / N^2)
println("\n多CGH时间复用 RMS: ", round(rms_multi, digits=8))
println("单CGH (csotGS)  RMS: ", round(rms_single, digits=8))
