"""
CSOT 效果基准测试 — 定量验证约束分离像素分组对相位恢复的改进

独立运行 (不需要任何外部依赖，仅 numpy + scipy):
    python benchmark.py
"""

import numpy as np
from scipy.fftpack import fft2, ifft2, fftshift, ifftshift
import time
import matplotlib
matplotlib.use('Agg')  # 无GUI也可运行
import matplotlib.pyplot as plt


# ============================================================
# 傅里叶工具
# ============================================================
def FFT2(x):
    return fftshift(fft2(ifftshift(x)))


def IFFT2(x):
    return fftshift(ifft2(ifftshift(x)))


# ============================================================
# 像素分组 (约束分离核心)
# ============================================================
def create_groups(shape, n_groups=4, seed=42):
    rng = np.random.RandomState(seed)
    ny, nx = shape
    y_idx = np.arange(ny)[:, None]
    x_idx = np.arange(nx)[None, :]

    if n_groups >= 4:
        # 4色棋盘格: 每组2x2块内各有一个像素
        groups = (y_idx % 2) * 2 + (x_idx % 2)
        if n_groups > 4:
            block_offset = ((y_idx // 2) % (n_groups // 4)) * 4
            groups = (groups + block_offset) % n_groups
    elif n_groups == 2:
        # 2组: 棋盘格交替
        groups = (y_idx + x_idx) % 2
    elif n_groups == 3:
        # 3组: 需要更大pattern确保邻接约束
        groups = (y_idx + 2 * x_idx) % 3

    perm = rng.permutation(n_groups)
    return perm[groups].astype(np.int32)


# ============================================================
# 标准 GS (全图同步更新)
# ============================================================
def standard_gs(source_intensity, target_intensity, n_iter):
    source_amp = np.sqrt(source_intensity / source_intensity.sum())
    target_amp = np.sqrt(target_intensity / target_intensity.sum())
    phase = np.random.randn(*source_intensity.shape) * 0.1
    errors = []

    for it in range(n_iter):
        field = source_amp * np.exp(1j * phase)
        field_ft = FFT2(field)
        field_ft = target_amp * np.exp(1j * np.angle(field_ft))
        phase = np.angle(IFFT2(field_ft))

        if it % max(1, n_iter // 10) == 0:
            out = np.abs(FFT2(source_amp * np.exp(1j * phase))) ** 2
            out /= out.sum()
            tgt = target_intensity / target_intensity.sum()
            errors.append(np.sqrt(np.mean((out - tgt) ** 2)))

    return phase, errors


# ============================================================
# 分组GS (约束分离: 分组交替更新)
# ============================================================
def grouped_gs(source_intensity, target_intensity, n_iter, n_groups=4):
    source_amp = np.sqrt(source_intensity / source_intensity.sum())
    target_amp = np.sqrt(target_intensity / target_intensity.sum())
    groups = create_groups(source_intensity.shape, n_groups)
    masks = [(groups == g) for g in range(n_groups)]
    phase = np.random.randn(*source_intensity.shape) * 0.1
    errors = []

    for it in range(n_iter):
        # 随机组顺序 — 关键!
        group_order = np.random.permutation(n_groups)

        for g in group_order:
            mask = masks[g]
            # 用全部像素做FFT
            field = source_amp * np.exp(1j * phase)
            field_ft = FFT2(field)
            field_ft = target_amp * np.exp(1j * np.angle(field_ft))
            phase_new = np.angle(IFFT2(field_ft))

            # ★ 仅更新本组 — 约束分离关键!
            phase[mask] = phase_new[mask]

        if it % max(1, n_iter // 10) == 0:
            out = np.abs(FFT2(source_amp * np.exp(1j * phase))) ** 2
            out /= out.sum()
            tgt = target_intensity / target_intensity.sum()
            errors.append(np.sqrt(np.mean((out - tgt) ** 2)))

    return phase, errors


# ============================================================
# 生成测试目标
# ============================================================
def make_source(N):
    """高斯源"""
    xs = np.linspace(-N / 2, N / 2, N)
    ys = np.linspace(-N / 2, N / 2, N)
    X, Y = np.meshgrid(xs, ys)
    src = np.exp(-(X ** 2 + Y ** 2) / (2 * (N / 6) ** 2))
    return src / src.sum()


def make_target_ring(N):
    """环形目标"""
    xs = np.linspace(-N / 2, N / 2, N)
    ys = np.linspace(-N / 2, N / 2, N)
    X, Y = np.meshgrid(xs, ys)
    R, W = N / 4, N / 20
    tgt = np.exp(-(np.sqrt(X ** 2 + Y ** 2) - R) ** 2 / (2 * W ** 2))
    return tgt / tgt.sum()


def make_target_multi_gauss(N):
    """多高斯目标 (更复杂)"""
    xs = np.linspace(-N / 2, N / 2, N)
    ys = np.linspace(-N / 2, N / 2, N)
    X, Y = np.meshgrid(xs, ys)
    tgt = np.zeros((N, N))
    for x0, y0, s in [(0, 0, N / 10), (-N / 5, N / 5, N / 14),
                       (N / 5, -N / 5, N / 14), (-N / 5, -N / 5, N / 16),
                       (N / 5, N / 5, N / 16), (0, N / 4, N / 20),
                       (0, -N / 4, N / 20)]:
        tgt += np.exp(-((X - x0) ** 2 + (Y - y0) ** 2) / (2 * s ** 2))
    return tgt / tgt.sum()


def make_target_square(N):
    """方形目标 (有尖锐边缘, 最困难)"""
    tgt = np.zeros((N, N))
    s = N // 4
    c = N // 2
    tgt[c - s:c + s, c - s:c + s] = 1.0
    # 轻微高斯模糊
    xs = np.linspace(-N / 2, N / 2, N)
    ys = np.linspace(-N / 2, N / 2, N)
    X, Y = np.meshgrid(xs, ys)
    kernel = np.exp(-(X ** 2 + Y ** 2) / (2 * 1.5 ** 2))
    kernel /= kernel.sum()
    from scipy.signal import convolve2d
    tgt = convolve2d(tgt, kernel, mode='same')
    return tgt / tgt.sum()


# ============================================================
# 多轮统计测试
# ============================================================
def run_benchmark(N=128, n_iter=200, n_trials=5):
    """运行完整基准测试，多轮取平均"""
    targets = {
        '环形': make_target_ring(N),
        '多高斯': make_target_multi_gauss(N),
        '方形(模糊)': make_target_square(N),
    }
    source = make_source(N)

    results = {}

    for tgt_name, target in targets.items():
        print(f"\n{'=' * 60}")
        print(f"  目标: {tgt_name} (N={N})")
        print(f"{'=' * 60}")

        gs_errors = []
        grouped_errors = []

        for trial in range(n_trials):
            seed = 42 + trial
            np.random.seed(seed)

            t0 = time.time()
            _, errs_gs = standard_gs(source, target, n_iter)
            t_gs = time.time() - t0

            np.random.seed(seed)  # 同样初始相位
            t0 = time.time()
            _, errs_grouped = grouped_gs(source, target, n_iter, n_groups=4)
            t_grouped = time.time() - t0

            gs_errors.append(errs_gs[-1])
            grouped_errors.append(errs_grouped[-1])

        gs_mean = np.mean(gs_errors)
        gs_std = np.std(gs_errors)
        grouped_mean = np.mean(grouped_errors)
        grouped_std = np.std(grouped_errors)
        improvement = (1 - grouped_mean / gs_mean) * 100

        print(f"\n  {'方法':20s} {'最终RMS误差':>15s} {'标准差':>10s}")
        print(f"  {'-' * 45}")
        print(f"  {'标准GS (全图同步)':20s} {gs_mean:15.6f} {gs_std:10.6f}")
        print(f"  {'分组GS (约束分离)':20s} {grouped_mean:15.6f} {grouped_std:10.6f}")
        print(f"  {'改进':20s} {improvement:>14.1f}%")

        results[tgt_name] = {
            'gs_mean': gs_mean, 'gs_std': gs_std,
            'grouped_mean': grouped_mean, 'grouped_std': grouped_std,
            'improvement': improvement,
            't_gs': t_gs, 't_grouped': t_grouped,
        }

    return results, source, targets


# ============================================================
# 收敛曲线对比 (单次运行)
# ============================================================
def plot_convergence(source, target, N=128, n_iter=300):
    """绘制详细收敛曲线"""
    np.random.seed(42)

    # 标准GS
    _, errs_gs = standard_gs(source, target, n_iter)

    # 分组GS (不同分组数)
    grouped_results = {}
    for ng in [2, 4, 8]:
        np.random.seed(42)
        _, errs = grouped_gs(source, target, n_iter, n_groups=ng)
        grouped_results[ng] = errs

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 左图: 收敛曲线
    iters_gs = np.arange(0, n_iter, n_iter // 10)
    iters_grouped = np.arange(0, n_iter, n_iter // 10)

    axes[0].plot(iters_gs, errs_gs, 'o-', label='标准GS (全图同步)',
                 markersize=4, linewidth=2)
    for ng, errs in grouped_results.items():
        axes[0].plot(iters_grouped, errs, 's--', label=f'分组GS (n={ng})',
                     markersize=4, linewidth=1.5)
    axes[0].set_xlabel('迭代次数', fontsize=12)
    axes[0].set_ylabel('RMS 误差', fontsize=12)
    axes[0].set_title('收敛曲线对比', fontsize=14)
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_yscale('log')

    # 右图: 最终误差柱状图
    methods = ['GS', '分组GS\n(n=2)', '分组GS\n(n=4)', '分组GS\n(n=8)']
    final_errs = [errs_gs[-1]] + [grouped_results[ng][-1] for ng in [2, 4, 8]]
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']
    bars = axes[1].bar(methods, final_errs, color=colors)

    for bar, val in zip(bars, final_errs):
        axes[1].text(bar.get_x() + bar.get_width() / 2.,
                     bar.get_height() + 0.0005,
                     f'{val:.4f}', ha='center', va='bottom',
                     fontweight='bold', fontsize=11)

    axes[1].set_ylabel('最终 RMS 误差', fontsize=12)
    axes[1].set_title('最终误差对比', fontsize=14)

    # 标注改进百分比
    for ng, errs in grouped_results.items():
        imp = (1 - errs[-1] / errs_gs[-1]) * 100
        axes[1].annotate(f'↓{imp:.0f}%',
                         xy=(list(grouped_results.keys()).index(ng) + 1,
                             errs[-1]),
                         fontsize=10, color='darkred', fontweight='bold')

    plt.tight_layout()
    plt.savefig('benchmark_convergence.png', dpi=150, bbox_inches='tight')
    print("\n[收敛曲线已保存到 benchmark_convergence.png]")

    return fig


# ============================================================
# 主程序
# ============================================================
if __name__ == '__main__':
    print("=" * 60)
    print("  CSOT Constraint Separation Benchmark")
    print("  Standard GS vs Grouped GS (adjacency constraint)")
    print("=" * 60)
    print("\nConfig: N=128, 200 iterations, 5 trials averaged\n")

    # 多目标多轮统计
    results, source, targets = run_benchmark(N=128, n_iter=200, n_trials=5)

    # 总汇总
    print(f"\n{'=' * 60}")
    print("  Summary")
    print(f"{'=' * 60}")
    print(f"\n  {'Target':15s} {'GS err':>12s} {'GroupGS err':>12s} {'Improve':>10s}")
    print(f"  {'-' * 50}")
    for name, r in results.items():
        print(f"  {name:15s} {r['gs_mean']:12.6f} {r['grouped_mean']:12.6f} "
              f"{r['improvement']:9.1f}%")

    avg_imp = np.mean([r['improvement'] for r in results.values()])
    print(f"\n  Average improvement: {avg_imp:.1f}%")

    # ================================================================
    # 早期收敛速度测试: 前50次迭代的逐次误差对比
    # ================================================================
    print(f"\n{'=' * 60}")
    print("  Early Convergence: per-iteration errors (N=256, square)")
    print(f"{'=' * 60}")

    N2 = 256
    source2 = make_source(N2)
    target2 = make_target_square(N2)
    source_amp2 = np.sqrt(source2 / source2.sum())
    tgt_amp2 = np.sqrt(target2 / target2.sum())
    tgt_norm2 = target2 / target2.sum()

    # 记录每次迭代的误差
    np.random.seed(42)
    phi_gs = np.random.randn(N2, N2) * 0.1
    gs_per_iter = []
    for it in range(50):
        field = source_amp2 * np.exp(1j * phi_gs)
        field_ft = FFT2(field)
        field_ft = tgt_amp2 * np.exp(1j * np.angle(field_ft))
        phi_gs = np.angle(IFFT2(field_ft))
        out = np.abs(FFT2(source_amp2 * np.exp(1j * phi_gs))) ** 2
        out /= out.sum()
        gs_per_iter.append(np.sqrt(np.mean((out - tgt_norm2) ** 2)))

    np.random.seed(42)
    phi_grp = np.random.randn(N2, N2) * 0.1
    groups = create_groups((N2, N2), 4, 42)
    masks = [(groups == g) for g in range(4)]
    grp_per_iter = []
    for it in range(50):
        group_order = np.random.permutation(4)
        for g in group_order:
            m = masks[g]
            field = source_amp2 * np.exp(1j * phi_grp)
            field_ft = FFT2(field)
            field_ft = tgt_amp2 * np.exp(1j * np.angle(field_ft))
            pnew = np.angle(IFFT2(field_ft))
            phi_grp[m] = pnew[m]
        out = np.abs(FFT2(source_amp2 * np.exp(1j * phi_grp))) ** 2
        out /= out.sum()
        grp_per_iter.append(np.sqrt(np.mean((out - tgt_norm2) ** 2)))

    # 计算关键指标
    # 1. 前5次迭代的平均改进
    early_gs = np.mean(gs_per_iter[:5])
    early_grp = np.mean(grp_per_iter[:5])
    early_imp = (1 - early_grp / early_gs) * 100

    # 2. 前10次迭代中有多少次分组GS更好
    better_count = sum(1 for g, s in zip(grp_per_iter[:10], gs_per_iter[:10]) if g < s)

    # 3. 最终误差
    final_gs = gs_per_iter[-1]
    final_grp = grp_per_iter[-1]
    final_imp = (1 - final_grp / final_gs) * 100

    better_str = f"{better_count}/10"
    print(f"\n  {'Metric':35s} {'Standard GS':>12s} {'Grouped GS':>12s} {'Improvement':>12s}")
    print(f"  {'-' * 70}")
    print(f"  {'Avg error (first 5 iters)':35s} {early_gs:12.6f} {early_grp:12.6f} {early_imp:>11.1f}%")
    print(f"  {'Better in first 10 iters':35s} {'':>12s} {better_str:>12s}")
    print(f"  {'Final error (50 iters)':35s} {final_gs:12.6f} {final_grp:12.6f} {final_imp:>11.1f}%")

    # 逐次误差对比
    print(f"\n  Per-iteration comparison (first 15):")
    print(f"  {'Iter':>5s} {'GS':>12s} {'GroupGS':>12s} {'Delta':>12s} {'Winner':>8s}")
    print(f"  {'-' * 50}")
    for i in range(15):
        delta = gs_per_iter[i] - grp_per_iter[i]
        w = "Group" if delta > 0 else "GS"
        print(f"  {i+1:5d} {gs_per_iter[i]:12.6f} {grp_per_iter[i]:12.6f} "
              f"{delta:+12.6f} {w:>8s}")

    # 收敛曲线图
    print(f"\n{'=' * 60}")
    print("  Convergence curves (square target, 300 iterations)")
    print(f"{'=' * 60}")
    N3 = 128
    source3 = make_source(N3)
    target3 = make_target_square(N3)
    fig = plot_convergence(source3, target3, N=N3, n_iter=300)
    print("\nDone! See benchmark_convergence.png")
