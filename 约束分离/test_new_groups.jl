using SLMTools

for N in [2, 3, 4, 5]
    ng = N * N
    g = createPixelGroups((12, 12), ng; shuffle_labels=false)
    info = verifyGroupConstraint(g)
    println("N=", N, " -> ", ng, "组: violations=", info.violations, ", valid=", info.valid)

    if N == 2
        println("  N=2 基模式 (4x4):")
        for y in 1:4
            row = [Int(g[y,x]) for x in 1:4]
            println("    ", row)
        end
    end
    if N == 3
        println("  N=3 基模式 (6x6):")
        for y in 1:6
            row = [Int(g[y,x]) for x in 1:6]
            println("    ", row)
        end
    end
end

g_shuf = createPixelGroups((10, 10), 9; shuffle_labels=true, seed=42)
info = verifyGroupConstraint(g_shuf)
println("\nN=3 shuffled: violations=", info.violations, ", groups=", info.n_groups)
println("\nDone!")
