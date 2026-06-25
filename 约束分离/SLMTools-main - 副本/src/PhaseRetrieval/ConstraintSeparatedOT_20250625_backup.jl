"""
ConstraintSeparatedOT.jl — 约束分离时间复用多CGH生成

## 约束分离原理 (Paper 2: 刘素娟)

将 SLM 像素网格划分为 N×N 块，块内 N² 个像素各属不同组。
保证块内全排列 + 跨块边界相邻不同组 + 全局标签置换去周期化。
每组独立生成子 CGH，时间复用抑制散斑。

## 物理方向

CGH 位于 **全息面/SLM 面**，物面和全息面之间是 Fresnel 传播：
  - 物面→全息面: propagateAS(+z)
  - 全息面→物面: propagateAS(-z)

## 融合模型

  CSOT (Paper 2) 负责: 分解结构 + 时间复用
  OT/GS (Paper 1) 负责: 每个子 CGH 内部相位优化 (只拟合子目标 A_i)

## 方法

  - csotPaperOT(A): 非迭代 — 分组 + 随机相位 + Fresnel 传播 → CGH
  - csotOT_GS(A, V, nit, Φ0): OT+GS — 每组 GS 只优化子目标 A_i
  - reconstructPaperOT(P, CGHs): Fresnel 逆向重建
  - reconstructMultiCGH(U, CGHs; groups): 通用重建
"""

module ConstraintSeparatedOT

using FFTW
using LinearAlgebra
using Random
using ..LatticeTools
using ..Misc

# ===========================================================================
# 导出
# ===========================================================================
export createPixelGroups, verifyGroupConstraint, createGroupMasks,
       csotPaperOT, csotOT_GS,
       reconstructPaperOT, reconstructMultiCGH,
       propagateAS

# ===========================================================================
# Fresnel 传播 (角谱的傍轴近似)
#
# 传递函数: H(ν) = exp(-j·π·λ·z·ν² / flambda)
# 频率换算: ν²_phys = ν²_nat / flambda
#
# flambda 默认 106400 (SLMTools Hogan 值: λ=1.064μm × f=100mm)
# 当 LatticeField.flambda≈1 (未设默认) 时自动替换为此值,
# 否则频率尺度错误 → Fresnel 核过采样溢出 → 噪声。
# ===========================================================================

# 有效 flambda: 若接近 1.0 (未设默认), 用 Hogan 物理值
_eff_flambda(fl::Real) = fl > 10.0 ? fl : 106400.0

"""
    propagateAS(field, L, z; λ=0.532, flambda=106400.0)

Fresnel 传播 (角谱傍轴近似)。

field: 复振幅矩阵 (fftshifted 坐标, DC 在中心)
L: 实空间晶格 (自然单位)
z > 0 正向 (物面→全息面), z < 0 逆向 (全息面→物面)
λ: 波长 (μm), flambda: λ×焦距 参数 (μm²), 默认 Hogan 值

频率换算: ν_phys² = ν_nat² / flambda
Fresnel 核: H = exp(-j·π·λ·z·ν_nat² / flambda)
"""
function propagateAS(field::AbstractMatrix{ComplexF64}, L::Lattice,
                     z::Real; λ::Real=0.532, flambda::Real=106400.0)
    ft  = plan_fft(field)
    ift = plan_ifft(field)

    # 自然频率坐标 (cycles/nat-unit), 从 natlat 自对偶 DFT 约定
    νL = dualShiftLattice(L, 1.0)
    νx = collect(Float64, νL[1])     # length Nx, 列方向频率
    νy = collect(Float64, νL[2])     # length Ny, 行方向频率

    # ν²[i,j] = νy[i]² + νx[j]² (2D 频率平方网格)
    # 必须显式构造 2D — @. 宏在奇数尺寸下会坍缩为 Vector
    ν² = (νy.^2) .+ reshape(νx.^2, 1, :)

    # Fresnel 核: H = exp(-j·π·λ·z·ν² / flambda)
    # z<0 时自动反向传播 (核取共轭)
    H = exp.(-im * π * λ * z .* ν² ./ flambda)

    # FFTW 管道: ifftshift→fft→fftshift, ×H, ifftshift→ift→fftshift
    F = fftshift(ft * ifftshift(field))
    return fftshift(ift * ifftshift(F .* H))
end

# Convenience: accept Real arrays
function propagateAS(field::AbstractMatrix{<:Real}, L::Lattice,
                     z::Real; λ::Real=0.532, flambda::Real=106400.0)
    propagateAS(complex.(field), L, z; λ=λ, flambda=flambda)
end

# ===========================================================================
# 核心: 约束分离像素分组 (N×N 块分离)
# ===========================================================================

"""
    createPixelGroups(shape::Tuple{Int,Int}, n_groups::Int=4;
                      shuffle_labels::Bool=true, seed=nothing)

N×N 块内拒绝采样随机排列 + 全局标签置换去周期化。
返回 (ny, nx) 整数矩阵，值 0..n_groups-1
"""
function createPixelGroups(shape::Tuple{Int,Int}, n_groups::Int=4;
                           shuffle_labels::Bool=true,
                           seed::Union{Nothing,Int}=nothing)
    ny, nx = shape
    N = round(Int, sqrt(n_groups))
    N*N == n_groups || error("n_groups 必须是完全平方数, 收到: $n_groups")
    ny%N == 0 && nx%N == 0 || error("尺寸 ($ny,$nx) 不能被 N=$N 整除")

    rng = isnothing(seed) ? Random.default_rng() : MersenneTwister(abs(seed))
    groups = [((y-1)%N)*N + ((x-1)%N) for y=1:ny, x=1:nx]

    if shuffle_labels
        for by in 0:(ny÷N-1), bx in 0:(nx÷N-1)
            ys, xs = by*N+1, bx*N+1; ye, xe = ys+N-1, xs+N-1
            left = bx>0 ? [groups[y,xs-1] for y in ys:ye] : Int[]
            top  = by>0 ? [groups[ys-1,x] for x in xs:xe] : Int[]
            ok = false
            for _ in 1:5000
                p = randperm(rng, n_groups).-1; block = reshape(p, N, N)
                l = isempty(left) || all(i->block[i,1]!=left[i], 1:N)
                t = isempty(top)  || all(j->block[1,j]!=top[j], 1:N)
                if l && t; groups[ys:ye,xs:xe]=block; ok=true; break; end
            end
            ok || error("块($by,$bx)拒绝采样失败")
        end
        perm = randperm(rng, n_groups).-1
        for i in eachindex(groups); groups[i] = perm[groups[i]+1]; end
    end
    return groups
end

verifyGroupConstraint(groups::AbstractArray{<:Integer,2}) = begin
    ny,nx=size(groups); v=t=0
    @inbounds for y=1:ny,x=1:nx-1; t+=1; v+=(groups[y,x]==groups[y,x+1]); end
    @inbounds for y=1:ny-1,x=1:nx; t+=1; v+=(groups[y,x]==groups[y+1,x]); end
    (n_groups=length(unique(groups)), adjacent_pairs=t,
     violations=v, violation_rate=v/max(t,1), valid=v==0)
end

createGroupMasks(groups::AbstractArray{<:Integer,2}) =
    [groups.==(g-1) for g in 1:length(unique(groups))]

# ===========================================================================
# 填零补足辅助
# ===========================================================================

_pad_block(ny, nx, Nb) = (max(0,ceil(Int,ny/Nb)*Nb-ny), max(0,ceil(Int,nx/Nb)*Nb-nx))

function _pad_data(data::AbstractMatrix{T}, py, px) where T
    s = size(data); z = zeros(T, s[1]+py, s[2]+px); z[1:s[1],1:s[2]] = data; z
end
_crop_data(data::AbstractMatrix, ny, nx) = data[1:ny,1:nx]

function _pad_lf(lf::LF, py, px, ony, onx)
    dp = _pad_data(lf.data, py, px)
    L = lf.L; dy,dx = step(L[1]),step(L[2])
    Lp = (range(first(L[1]), first(L[1])+(ony+py-1)*dy, length=ony+py),
          range(first(L[2]), first(L[2])+(onx+px-1)*dx, length=onx+px))
    LF{typeof(lf).parameters[1]}(dp, Lp, lf.flambda)
end

# ===========================================================================
# 论文方法: 非迭代约束分离 (csotPaperOT)
# ===========================================================================

"""
    csotPaperOT(A::LF{Modulus}; n_groups=9, seed=nothing, verbose=true,
                z=200000, λ=0.532, Φ0=nothing)

论文方法 (Paper 2) — 约束分离非迭代 CGH 生成 (Fresnel 传播)。

每组: A_i = A × mask_g → 相位 → propagateAS(+z) → 取相位 → CGH
  - 若 Φ0 未提供: 用随机相位 (论文原方法)
  - 若 Φ0 提供:   用 OT 相位 (质量优于随机相位)

A 是目标物体振幅 (如 sqrt(C))。Φ0 是 OT 初始相位 (来自 otPhase)。

返回 (CGHs, groups)
"""
function csotPaperOT(A::LF{Modulus,<:Real,Nd};
                     n_groups::Int=9, seed::Union{Nothing,Int}=nothing,
                     verbose::Bool=true, z::Real=200000, λ::Real=0.532,
                     Φ0::Union{Nothing,LF{<:Phase,<:Number,Nd}}=nothing) where {Nd}
    Nb = round(Int, sqrt(n_groups))
    ony, onx = size(A); A_orig = A
    py, px = _pad_block(ony, onx, Nb)
    flam = _eff_flambda(A_orig.flambda)

    if py+px > 0
        A = _pad_lf(A, py, px, ony, onx)
        verbose && println("填零: ($ony,$onx) → $(size(A))")
    end

    groups = createPixelGroups(size(A), n_groups; seed=seed)
    masks  = createGroupMasks(groups)

    # 确定相位来源: OT 相位 > 随机相位
    if Φ0 !== nothing
        # pad Φ0 to match A if needed
        Φ0_data = Φ0 isa LF{ComplexPhase} ? Φ0.data : exp.(im .* Φ0.data)
        if py+px > 0
            Φ0_data = _pad_data(Φ0_data, py, px)
        end
        Φ_phase = Φ0_data ./ (abs.(Φ0_data) .+ eps(Float64))
        verbose && println("  使用 OT 相位")
    else
        rng = isnothing(seed) ? Random.default_rng() : MersenneTwister(abs(seed))
        Φ_phase = nothing  # 每组生成独立随机相位
        verbose && println("  使用随机相位")
    end

    CGHs = LF{ComplexPhase}[]

    for g in 1:n_groups
        verbose && println("  组 $g/$n_groups")
        # A_i × phase → Fresnel → hologram → extract phase → CGH
        phase = Φ_phase !== nothing ? Φ_phase :
                exp.(2π*im .* rand(rng, Float64, size(A)))
        field = A.data .* masks[g] .* phase
        prop  = propagateAS(field, A.L, z; λ=λ, flambda=flam)
        cgh_data = prop ./ (abs.(prop) .+ eps(Float64))
        if py+px > 0; cgh_data = _crop_data(cgh_data, ony, onx); end
        push!(CGHs, LF{ComplexPhase}(cgh_data, A_orig.L, flam))
    end
    py+px>0 && (groups = _crop_data(groups, ony, onx))
    return CGHs, groups
end

# ===========================================================================
# 约束分离 OT + GS (csotOT_GS)
#
# CSOT (Paper 2) 负责: 分组 + 时间复用
# OT   负责: 优良初相
# GS   负责: 迭代优化每个子 CGH
#
# GS 约束 (每组独立):
#   全息面: 纯相位 (CGH 是 phase-only)
#   物面:   子目标振幅 A_i = A × mask_i
#
# 只拟合子目标, 绝对不拟合完整 V!
# ===========================================================================

"""
    csotOT_GS(A::LF{Modulus}, V::LF{Modulus}, nit::Integer,
              Φ0::LF{<:Phase}; n_groups=9, seed=nothing, verbose=true,
              z=200000, λ=0.532)

约束分离 OT + GS — OT 初相 + GS 迭代优化。

每组 GS 迭代:
  物面(A_i) → propagateAS(+z) → 全息面 → phase-only → propagateAS(-z) → 物面(A_i)

只拟合子目标 A_i = A × mask_i, 不拟合完整 V。

返回 (CGHs, groups, errors_per_group)
"""
function csotOT_GS(A::LF{Modulus,<:Real,Nd}, V::LF{Modulus,<:Real,Nd},
                   nit::Integer, Φ0::LF{<:Phase,<:Number,Nd};
                   n_groups::Int=9, seed::Union{Nothing,Int}=nothing,
                   verbose::Bool=true, z::Real=200000, λ::Real=0.532) where {Nd}
    ldq(A, V); elq(A, Φ0)
    Nb = round(Int, sqrt(n_groups))
    ony, onx = size(A); A_orig = A
    py, px = _pad_block(ony, onx, Nb)
    flam = _eff_flambda(A_orig.flambda)

    if py+px > 0
        A  = _pad_lf(A, py, px, ony, onx)
        Φ0 = LF{ComplexPhase}(_pad_data(wrap(Φ0).data, py, px), A.L, Φ0.flambda)
        verbose && println("填零: ($ony,$onx) → $(size(A))")
    end

    groups = createPixelGroups(size(A), n_groups; seed=seed)
    masks  = createGroupMasks(groups)

    CGHs = LF{ComplexPhase}[]
    errs_all = Vector{Float64}[]

    # OT 相位 (全局, 一次性)
    # 注意: padding 后 Φ0 已是 LF{ComplexPhase}, .data 即 exp(jφ);
    # 未 padding 时 Φ0 是 LF{RealPhase}, 需 wrap 转换为 exp(jφ)
    Φot_data = eltype(Φ0.data) <: Complex ? Φ0.data : exp.(im .* Φ0.data)
    Φot_phase = Φot_data ./ (abs.(Φot_data) .+ eps(Float64))

    for g in 1:n_groups
        verbose && println("  组 $g/$n_groups")

        # 子目标振幅 (稀疏, 仅 1/N² 像素非零)
        m  = masks[g]
        Ai = A.data .* m
        Ai² = Ai.^2
        denom = sum(Ai²) + eps(Float64)

        # OT 初相 → 物面复振幅
        guess = Ai .* Φot_phase
        errs  = Float64[]

        for it in 1:nit
            # === 前向: 物面 → 全息面 ===
            field_h = propagateAS(guess, A.L, z; λ=λ, flambda=flam)

            # === 全息面约束: 纯相位 (CGH phase-only) ===
            field_h = field_h ./ (abs.(field_h) .+ eps(Float64))

            # === 反向: 全息面 → 物面 ===
            field_obj = propagateAS(field_h, A.L, -z; λ=λ, flambda=flam)

            # === 物面约束: 子目标振幅 A_i (不是完整 V!) ===
            guess = Ai .* (field_obj ./ (abs.(field_obj) .+ eps(Float64)))

            # 误差追踪: 归一化后比较 (重建能量 N² ≠ 目标能量 Sum(Ai²))
            if it == 1 || it % max(1, nit÷5) == 0 || it == nit
                rec = propagateAS(field_h, A.L, -z; λ=λ, flambda=flam)
                I_rec = abs2.(rec)
                I_rec_n = I_rec ./ (sum(I_rec) + eps(Float64))
                Ai²_n   = Ai² ./ (sum(Ai²) + eps(Float64))
                push!(errs, sqrt(sum(abs2, I_rec_n .- Ai²_n)))
            end
        end

        # 最终 CGH: 优化后的物面 → 全息面 → 取相位
        field_final = propagateAS(guess, A.L, z; λ=λ, flambda=flam)
        cgh_data    = field_final ./ (abs.(field_final) .+ eps(Float64))

        if py+px > 0; cgh_data = _crop_data(cgh_data, ony, onx); end
        push!(CGHs, LF{ComplexPhase}(cgh_data, A_orig.L, flam))
        push!(errs_all, errs)
        verbose && println("    RMS: $(round(errs[end],digits=6))")
    end
    py+px>0 && (groups = _crop_data(groups, ony, onx))
    return CGHs, groups, errs_all
end

# ===========================================================================
# 重建: Fresnel 逆向传播
# ===========================================================================

"""
    reconstructPaperOT(P::LF{Modulus}, CGHs::Vector{LF{ComplexPhase}};
                       z=200000, λ=0.532)

论文方法重建: SLM 入射振幅 P × CGH → propagateAS(-z) → 子像 → 时间平均。

**不逐张归一化** — 保持子图原始亮度, 时间平均后自然合成。
"""
function reconstructPaperOT(P::LF{Modulus,<:Real,Nd},
                             CGHs::Vector{LF{ComplexPhase}};
                             z::Real=200000, λ::Real=0.532) where {Nd}
    flam = _eff_flambda(P.flambda)   # 若未设则用 Hogan 值

    # Pre-compute Fresnel backward kernel
    νL = dualShiftLattice(P.L, 1.0)
    νx = collect(Float64, νL[1])
    νy = collect(Float64, νL[2])
    ν² = (νy.^2) .+ reshape(νx.^2, 1, :)
    Hb = exp.(im * π * λ * z .* ν² ./ flam)   # backward = conjugate of forward

    ft  = plan_fft(zeros(ComplexF64, size(P)))
    ift = plan_ifft(zeros(ComplexF64, size(P)))

    subs = Matrix{Float64}[]

    for cgh in CGHs
        # field at hologram plane: P × CGH
        field_h = complex.(P.data) .* cgh.data

        # propagate backward to object plane
        # Pipeline: ifftshift→fft→fftshift, ×Hb, ifftshift→ift→fftshift
        spec_h   = fftshift(ft * ifftshift(field_h))
        spec_obj = spec_h .* Hb
        rec      = fftshift(ift * ifftshift(spec_obj))   # ← ifftshift before IFFT!

        push!(subs, abs2.(rec))
    end

    # Time-average WITHOUT per-image normalization
    avg = sum(subs) / length(CGHs)
    return LF{Intensity}(avg, P.L, flam)
end

"""
    reconstructMultiCGH(U::LF{Modulus}, CGHs::Vector{LF{ComplexPhase}};
                        groups=nothing, z=200000, λ=0.532)

通用时间复用重建。若提供 groups, 每组用自己的振幅掩码重建。
"""
function reconstructMultiCGH(U::LF{Modulus,<:Real,Nd},
                              CGHs::Vector{LF{ComplexPhase}};
                              groups=nothing, z::Real=200000,
                              λ::Real=0.532) where {Nd}
    flam = _eff_flambda(U.flambda)   # 若未设则用 Hogan 值

    # Pre-compute Fresnel backward kernel
    νL = dualShiftLattice(U.L, 1.0)
    νx = collect(Float64, νL[1])
    νy = collect(Float64, νL[2])
    ν² = (νy.^2) .+ reshape(νx.^2, 1, :)
    Hb = exp.(im * π * λ * z .* ν² ./ flam)

    ft  = plan_fft(zeros(ComplexF64, size(U)))
    ift = plan_ifft(zeros(ComplexF64, size(U)))

    if isnothing(groups)
        subs = Matrix{Float64}[]
        for cgh in CGHs
            field_h  = complex.(U.data) .* cgh.data
            spec_h   = fftshift(ft * ifftshift(field_h))
            spec_obj = spec_h .* Hb
            rec      = fftshift(ift * ifftshift(spec_obj))
            push!(subs, abs2.(rec))
        end
    else
        masks = createGroupMasks(groups)
        subs = Matrix{Float64}[]
        for (g, cgh) in enumerate(CGHs)
            field_h  = complex.(U.data .* masks[g]) .* cgh.data
            spec_h   = fftshift(ft * ifftshift(field_h))
            spec_obj = spec_h .* Hb
            rec      = fftshift(ift * ifftshift(spec_obj))
            push!(subs, abs2.(rec))
        end
    end

    avg = sum(subs) / length(CGHs)
    return LF{Intensity}(avg, U.L, flam)
end

end # module ConstraintSeparatedOT
