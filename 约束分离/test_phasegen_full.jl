# 完整模拟 phase-generation notebook 执行
using SLMTools, Images, FreeTypeAbstraction, FileIO, Plots
include("d:/6.2谈话资料/约束分离/SLMTools-main - 副本/src/SubImages.jl")

# Cell 2-3: Setup
N = 128; L0 = natlat((N,N))

# Cell 3: Input beam
I = LF{ComplexAmp}(exp.(-r2(L0) ./ (3 + 0im)^2), L0)

# Cell 4: Target
C = lfRect(I|>square,(4,4))*0.1 + lfText(Intensity,L0,"c"; pixelsize=150)
C = LF{Intensity}(imfilter(C.data, Kernel.gaussian(2)),C.L)

# Cell 5-8: Standard methods
Φgs = gs(abs(I),sqrt(C),100,SLMTools.wrap(LF{RealPhase}(rand(size(C)...),C.L)))
Φot = otPhase(square(I),C,0.001)
Φotgs = gs(abs(I),sqrt(C),100,SLMTools.wrap(Φot))
Φotmraf = mraf(abs(I),sqrt(C),100,SLMTools.wrap(Φot),CartesianIndices((17:112,17:112)),0.48)

# Cell 9: csotGS
Φotcsotgs, csot_groups, csot_errors = csotGS(abs(I), sqrt(C), 100, Φot; n_groups=4, verbose=true)

# Cell 10: csotMRAF
Φotcsotmraf, csotmraf_groups, csotmraf_errors = csotMRAF(abs(I), sqrt(C), 100, Φot, CartesianIndices((17:112,17:112)), 0.48; n_groups=4, verbose=true)

# Cell 11: csotMultiCGH
println("\n=== csotMultiCGH ===")
CGHs_multi, multi_groups, multi_errs = csotMultiCGH(abs(I), sqrt(C), 100, Φot; n_groups=9, verbose=true)
out_multi = reconstructMultiCGH(abs(I), CGHs_multi; groups=multi_groups)

# Cell 12: csotMultiCGH_MRAF
println("\n=== csotMultiCGH_MRAF ===")
CGHs_mraf, mraf_groups, mraf_errs = csotMultiCGH_MRAF(abs(I), sqrt(C), 100, Φot, CartesianIndices((17:112,17:112)), 0.48; n_groups=9, verbose=true)
out_multi_mraf = reconstructMultiCGH(abs(I), CGHs_mraf; groups=mraf_groups)

# Cell: RMS eval
println("\n=== RMS ===")
for (name, Φ) in [("GS random",Φgs), ("OT",Φot), ("OT+GS",Φotgs), ("OT+MRAF",Φotmraf),
                    ("OT+csotGS",Φotcsotgs), ("OT+csotMRAF",Φotcsotmraf),
                    ("OT+csotMultiCGH",out_multi), ("OT+csotMultiMRAF",out_multi_mraf)]
    err = Φ isa LF{<:Phase} ? SchroffError(C, square(sft(abs(I) * Φ))) : SchroffError(C, Φ)
    println("  ", rpad(name,20), ": ", err)
end

# Cell: Visualization
println("\n=== Visualization ===")
phases = [Φgs, Φot, Φotgs, Φotmraf, Φotcsotgs, Φotcsotmraf]
outbeams = [square(sft(abs(I) * Φ)) for Φ in phases]
push!(outbeams, out_multi)
push!(outbeams, out_multi_mraf)
println("outbeams count: ", length(outbeams))
println("outbeams types: ", [typeof(o) for o in outbeams])

# Cell: Efficiency
println("\n=== Efficiency ===")
function boxEfficiency(F::LF{Intensity},roi::CartesianIndices)
    sum(F[roi].data)/sum(F.data)
end
function boxEfficiency(F::LF{<:Amplitude},roi::CartesianIndices)
    boxEfficiency(square(F),roi)
end
roi = CartesianIndices((17:112,17:112))
for (name, Φ) in [("GS random",Φgs), ("OT",Φot), ("OT+GS",Φotgs), ("OT+MRAF",Φotmraf),
                    ("OT+csotGS",Φotcsotgs), ("OT+csotMRAF",Φotcsotmraf),
                    ("OT+csotMultiCGH",out_multi), ("OT+csotMultiMRAF",out_multi_mraf)]
    η = Φ isa LF{<:Phase} ? boxEfficiency(square(sft(abs(I) * Φ)), roi) : boxEfficiency(Φ, roi)
    println("  ", rpad(name,20), ": ", η)
end

println("\nALL CELLS PASSED!")
