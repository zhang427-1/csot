# ============================================================
# 约束分离 (CSOT) 测试脚本
# 测试: 分组逻辑、GS迭代、MRAF迭代
# ============================================================

using Pkg
Pkg.activate("d:/6.2谈话资料/约束分离/SLMTools-main - 副本")

using SLMTools
using FFTW, LinearAlgebra, Random, Images

# ============================================================
# Test 1: 像素分组逻辑
# ============================================================
println("="^60)
println("Test 1: createPixelGroups & verifyGroupConstraint")
println("="^60)

for shape in [(16,16), (32,32), (128,128)]
    for ng in [4, 8, 16]
        groups = createPixelGroups(shape, ng; seed=42)
        info = verifyGroupConstraint(groups)
        @assert info.valid "Constraint violated! shape=$shape ng=$ng"
    end
end
println("✓ All group patterns passed constraint verification")

# Verify layout correctness for 4 groups
groups4 = createPixelGroups((4,4), 4; seed=nothing, shuffle_labels=false)
println("  4-group pattern (4x4, unshuffled):")
for y in 1:4
    println("    ", [groups4[y,x] for x in 1:4])
end

# Verify each group has roughly equal size
groups = createPixelGroups((100,100), 4; seed=42)
masks = createGroupMasks(groups)
sizes = [sum(m) for m in masks]
println("  Group sizes (100×100, 4 groups): $sizes (expect ~2500 each)")

# ============================================================
# Test 2: 小规模 csotGS 运行
# ============================================================
println("\n" * "="^60)
println("Test 2: csotGS on small 64×64 problem")
println("="^60)

N = 64
L = natlat((N, N))
I_in = LF{Intensity}(exp.(-r2(L) ./ (3.0^2)), L) |> normalizeLF
target = lfRect(I_in, (20,20)) * 0.1 + lfText(Intensity, L, "C"; pixelsize=80)
target = LF{Intensity}(imfilter(target.data, Kernel.gaussian(2.0)), L) |> normalizeLF

println("  Grid: $(N)×$(N)")
println("  Input power: $(round(sum(I_in.data), digits=4))")
println("  Target power: $(round(sum(target.data), digits=4))")

# Random initial phase
Φ0 = LF{ComplexPhase}(exp.(2π * im * rand(N, N)), L, 1.0)

U = sqrt(I_in)
V = sqrt(target)

println("  Running csotGS 50 iterations...")
Φ_csot, grps, errs = csotGS(U, V, 50, Φ0; n_groups=4, verbose=true)

println("  Final RMS errors: ", round.(errs, digits=8))
println("  Return type: ", typeof(Φ_csot))
println("  Groups type: ", typeof(grps), " size=", size(grps))

out = square(sft(U * Φ_csot)) |> normalizeLF
rms_err = sqrt(sum((out.data .- normalizeLF(target).data).^2) / length(target.data))
println("  RMS error vs target: ", round(rms_err, digits=6))

# ============================================================
# Test 3: 小规模 csotMRAF 运行
# ============================================================
println("\n" * "="^60)
println("Test 3: csotMRAF on small 64×64 problem")
println("="^60)

roi = CartesianIndices((12:53, 12:53))
m_val = 0.48

println("  ROI: 12:53 × 12:53, m=$m_val")
println("  Running csotMRAF 50 iterations...")
Φ_csot_mraf, grps_m, errs_m = csotMRAF(U, V, 50, Φ0, roi, m_val; n_groups=4, verbose=true)

out_m = square(sft(U * Φ_csot_mraf)) |> normalizeLF
rms_err_m = sqrt(sum((out_m.data .- normalizeLF(target).data).^2) / length(target.data))
println("  RMS error vs target: ", round(rms_err_m, digits=6))

# ============================================================
# Test 4: 与标准 GS 对比
# ============================================================
println("\n" * "="^60)
println("Test 4: csotGS vs standard GS comparison")
println("="^60)

Φ_gs = gs(U, V, 50, Φ0)
out_gs = square(sft(U * Φ_gs)) |> normalizeLF
rms_gs = sqrt(sum((out_gs.data .- normalizeLF(target).data).^2) / length(target.data))

rms_csot = sqrt(sum((out.data .- normalizeLF(target).data).^2) / length(target.data))

println("  Standard GS  (50 iter): RMS = ", round(rms_gs, digits=8))
println("  CSOT GS      (50 iter): RMS = ", round(rms_csot, digits=8))

# ============================================================
# Test 5: 返回类型验证 (确保 notebook 用法正确)
# ============================================================
println("\n" * "="^60)
println("Test 5: Return type verification")
println("="^60)

# csotGS returns (LF{ComplexPhase}, groups::Matrix{Int}, errors::Vector{Float64})
# Notebook uses csotGS(...)[1] to extract phase
result = csotGS(U, V, 5, Φ0; n_groups=4, verbose=false)
println("  csotGS returns: ", typeof(result))
println("  result[1] type: ", typeof(result[1]), " (used as Φ)")
println("  result[2] type: ", typeof(result[2]), " (groups matrix)")
println("  result[3] type: ", typeof(result[3]), " (errors vector)")

@assert result[1] isa LF{ComplexPhase} "Φ must be LF{ComplexPhase}"
@assert result[2] isa Matrix{<:Integer} "groups must be integer matrix"
@assert result[3] isa Vector{Float64} "errors must be Float64 vector"
println("  ✓ All types match expected")

# csotMRAF returns same structure
result_m = csotMRAF(U, V, 5, Φ0, roi, 0.48; n_groups=4, verbose=false)
println("  csotMRAF returns: ", typeof(result_m))
@assert result_m[1] isa LF{ComplexPhase}
@assert result_m[2] isa Matrix{<:Integer}
@assert result_m[3] isa Vector{Float64}
println("  ✓ All types match expected")

# ============================================================
println("\n" * "="^60)
println("ALL TESTS PASSED ✓")
println("="^60)
