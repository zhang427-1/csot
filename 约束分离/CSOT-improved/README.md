# CSOT: 约束分离最优传输相位恢复

将**论文1（约束分离方法）**与**论文2（OT相位恢复, arXiv:2408.17025）**结合的改进算法。

## 约束分离原理

**约束分离 = 像素随机分组 + 相邻像素不能同组 → 非周期、有间隔的最优分布**

- 将 SLM 像素随机分配到 K 个组（默认4组）
- **核心约束**: 相邻像素（上下左右）不能属于同一组（4色图着色）
- 每组独立处理（"分离"），其他组固定 → 分块坐标下降
- 效果: 打破FFT隐含的周期性伪影，实现空间去相关的优化

### 为什么有效？

1. **打破周期性**: FFT 隐含周期性边界条件，全图同步更新产生周期性伪影
   → 分组交替更新打破这种周期性耦合

2. **空间去相关**: 相邻像素不同组 → 每组像素在空间中均匀分散
   → 每组看到的是全局信息，而非局部短程信息

3. **分块坐标下降**: 每组更新时其他组固定
   → 降低耦合维度 → 更容易逃离局部极小值

## 目录结构

```
CSOT-improved/
├── README.md                          # 本文件
├── python/
│   ├── csot_phase.py                  # Python 独立模块
│   ├── csot_demo.ipynb                # Python 演示 notebook
│   ├── benchmark.py                   # 基准测试脚本
│   ├── paper_comparison.py            # 论文风格对比
│   ├── paper_figure.py                # 论文风格图生成
│   └── replicate_paper_fig.py         # 论文实验复现
└── julia/
    ├── ConstraintSeparatedOT.jl       # Julia 模块 (集成到 SLMTools)
    └── csot_demo.ipynb                # Julia 演示 notebook
```

## Julia 集成方法

1. 将 `ConstraintSeparatedOT.jl` 复制到 `SLMTools-main/src/PhaseRetrieval/`
2. 在 `SLMTools-main/Project.toml` 的 `[deps]` 中添加:
   ```
   Random = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
   ```
3. 在 `SLMTools-main/src/SLMTools.jl` 的 OT.jl include 之后添加:
   ```julia
   include("PhaseRetrieval/ConstraintSeparatedOT.jl")
   using .ConstraintSeparatedOT
   export createPixelGroups, verifyGroupConstraint, createGroupMasks,
          csotGS, csotGSLog, csotMRAF, csotPhase
   ```

## 使用方法

### Julia (集成到 SLMTools)

```julia
using SLMTools

# 像素分组
groups = createPixelGroups((256, 256), 4)

# OT 初始化
Φ_ot = otPhase(I_in, I_target, 0.001)

# 约束分离分组GS — 替代标准 gs()
Φ_csot, groups, errors = csotGS(
    sqrt(I_in), sqrt(I_target), 10000, Φ_ot; n_groups=4)

# 约束分离分组MRAF — 替代标准 mraf()
Φ_csot_mraf, groups, errors = csotMRAF(
    sqrt(I_in), sqrt(I_target), 10000, Φ_ot, roi, 0.48; n_groups=4)

# 完整 CSOT 流水线: OT + 约束分离精化
Φ_final, Φ_ot_init, errors = csotPhase(
    I_in, I_target, 0.001; refine_iter=50, n_groups=4)

# 收敛分析
Φ, errs = csotGSLog(sqrt(I_in), sqrt(I_target), 1000, Φ_ot;
                     n_groups=4, every=10)
```

### 论文 notebook 中使用

将 notebook 中的:
```julia
Φotgs = gs(I_in, I_target, 10000, wrap(Φot))
```
替换为:
```julia
Φotcsot, _, _ = csotGS(sqrt(I_in), sqrt(I_target), 10000, Φot; n_groups=4)
```

### Python (独立使用，依赖 OTPhaseExtractor)

```python
import sys
sys.path.insert(0, '<OTPhaseExtractor路径>')

from csot_phase import CSOTPhaseRetriever, create_pixel_groups

csot = CSOTPhaseRetriever(n_groups=4, ot_n_iter=200, refine_n_iter=100)
results = csot(source, target, xs, ys)
phase = results['phase_refined']
```

## API 参考

| 函数 | 说明 |
|---|---|
| `createPixelGroups(shape, n_groups)` | 创建约束分离像素分组 |
| `verifyGroupConstraint(groups)` | 验证相邻像素不同组约束 |
| `createGroupMasks(groups)` | 每组布尔掩码 |
| `csotGS(U, V, nit, Φ0; n_groups)` | 约束分离分组GS精化 |
| `csotMRAF(U, V, nit, Φ0, roi, m; n_groups)` | 约束分离分组MRAF精化 |
| `csotPhase(U, V, ε; refine_iter, n_groups)` | 完整CSOT流水线 |
| `csotGSLog(U, V, nit, Φ0; n_groups, every)` | 分组GS + 收敛日志 |

## 参考文献

- 论文1: "Variational approach to calculation of light field eikonal function for illuminating a prescribed region" (约束分离方法)
- 论文2: "High-Fidelity Holographic Beam Shaping with Optimal Transport and Phase Diversity" (arXiv:2408.17025)
