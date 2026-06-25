# ============================================================
# 验证 notebook 中的类型流: JLD2 Φots → csotGS
# ============================================================
using Pkg
Pkg.activate("d:/6.2谈话资料/约束分离/SLMTools-main - 副本")
using SLMTools, JLD2, Images

# 复制 notebook 中的 helper 函数
function centeredIntLat(ns::NTuple{N,Integer}) where N
    ls = .-(ns .÷ 2)
    us = (ns .- 1) .÷ 2
    return ((ls[i]:us[i] for i=1:N)...,)
end
function embed(f::LF{S,T,N}, L::Lattice{N}) where {S,T,N}
    d = (length.(L) .- length.(f.L)) .÷ 2
    return embed(f, L, CartesianIndex(d))
end
function embed(f::LF{S,T,N}, L::Lattice{N}, offset::CartesianIndex{N}) where {S,T,N}
    data = zeros(T, length.(L))
    data[offset .+ CartesianIndices(size(f))] .= f.data
    return LF{S}(data, L, f.flambda)
end

# 模拟 notebook 中的 JLD2 加载
jld_path = "d:/6.2谈话资料/约束分离/SLMTools-main - 副本/examples/target_comparisons.jld2"
println("Loading JLD2...")
jldopen(jld_path, "r") do f
    _ots = f["Φots"]
    global Φots = Tuple(LF{ComplexPhase}(p.data, p.L, p.flambda) for p in _ots)
end
println("✓ Loaded $(length(Φots)) phases")

# 检查类型
for i in 1:length(Φots)
    Φ = Φots[i]
    println("  Φots[$i]: $(typeof(Φ))")
    println("    data type: $(typeof(Φ.data)), size: $(size(Φ.data))")
    println("    L type: $(typeof(Φ.L))")
    println("    flambda: $(Φ.flambda)")
end

# 验证类型匹配 csotGS 的签名
# csotGS(U::LF{Modulus,<:Real,N}, V::LF{Modulus,<:Real,N}, nit::Integer, Φ0::LF{<:Phase,<:Number,N})
Φ = Φots[1]
println("\n  ComplexPhase <: Phase? $(ComplexPhase <: Phase)")
println("  eltype(Φ.data) <: Number? $(eltype(Φ.data) <: Number)")
println("  Φ isa LF{<:Phase,<:Number}? $(Φ isa LF{<:Phase,<:Number})")

# 模拟 notebook 中 I2_f64 的创建
println("\n--- Simulating notebook I2_f64 creation ---")
N2 = 768
Lslm = centeredIntLat((N2,N2))
Lbig = centeredIntLat((2*N2,2*N2))
I2 = sqrt(embed(lfGaussian(Intensity, Lslm, 565/2), Lbig))
I2_f64 = LF{Modulus}(Float64.(I2.data), I2.L, I2.flambda)
println("  I2 type: $(typeof(I2)), data eltype: $(eltype(I2.data))")
println("  I2_f64 type: $(typeof(I2_f64)), data eltype: $(eltype(I2_f64.data))")

# 验证 lattice 兼容性
println("  I2.L == Φots[1].L? $(I2.L == Φots[1].L)")
println("  I2.L == I2_f64.L? $(I2.L == I2_f64.L)")

# 测试 tgts_f64 创建 (修复后版本)
println("\n--- Simulating notebook tgts_f64 creation ---")
# 简化版 target 创建
dLbig = dualShiftLattice(Lbig)
L200 = centeredIntLat((200,200))
x = L200[1]; y = L200[2]
diadem_data = [exp(-(xi^2 + yj^2)/(2*14^2)) for xi in x, yj in y]
diadem = LF{Intensity}(diadem_data, L200, 1.0) |> normalizeLF
target = embed(diadem, dLbig)
println("  target type: $(typeof(target))")
tgts_f64 = LF{Modulus}(Float64.(sqrt(target).data), target.L, target.flambda)
println("  tgts_f64 type: $(typeof(tgts_f64)), data eltype: $(eltype(tgts_f64.data))")
println("  ✓ tgts_f64 creation succeeded")

# 最终: 测试 csotGS 调用
println("\n--- Testing csotGS call (same pattern as notebook) ---")
Φ_test, _, _ = csotGS(I2_f64, tgts_f64, 5, Φots[1]; n_groups=4, verbose=false)
println("  ✓ csotGS returned type: $(typeof(Φ_test))")
println("  ✓ Output data eltype: $(eltype(Φ_test.data))")

println("\nALL NOTEBOOK TYPE CHECKS PASSED ✓")
