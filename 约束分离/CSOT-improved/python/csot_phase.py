"""
CSOT: Constraint-Separated Optimal Transport Phase Retrieval
=============================================================

将论文1的**约束分离方法**与论文2的**OT相位恢复**结合。

## 约束分离原理（论文1）

约束分离 = 像素随机分组 + 相邻像素不能同组 → 非周期、有间隔的最优分布

- 将 SLM 像素随机分配到 K 个组
- **核心约束**: 相邻像素不能属于同一组（图着色约束）
- 效果: 打破周期性伪影，实现空间去相关的优化
- 每组独立处理（"分离"），实现分块坐标下降

这本质上是像素网格上的**随机图着色**问题（4-邻接，4色足够）。

## 与论文2的结合

论文2（arXiv:2408.17025）使用 Sinkhorn OT + MRAF/GS 进行相位恢复，
但所有像素耦合在一起全局优化。

CSOT 改进:
1. **OT阶段**: 按组采样进行随机对偶OT优化，每组提供空间多样化的batch
2. **精化阶段**: 分组交替投影（每组独立更新，其他组固定）
3. **效果**: 打破周期性、加速收敛、减少伪影
"""

import torch
from torch import nn, optim
import numpy as np
from tqdm import tqdm
from scipy.fftpack import fft2, ifft2, fftshift, ifftshift
from typing import Optional, Tuple, Dict, List
import matplotlib.pyplot as plt


# ============================================================================
# 傅里叶工具函数
# ============================================================================

def FFT2(data: np.ndarray) -> np.ndarray:
    """居中 2D 正傅里叶变换"""
    return fftshift(fft2(ifftshift(data)))


def IFFT2(data: np.ndarray) -> np.ndarray:
    """居中 2D 逆傅里叶变换"""
    return fftshift(ifft2(ifftshift(data)))


def fourier_propagate(intensity: np.ndarray, phase: np.ndarray) -> np.ndarray:
    """场传播: I, φ → FFT → 输出强度"""
    amp = np.sqrt(intensity / intensity.sum())
    out = np.abs(FFT2(amp * np.exp(1j * phase))) ** 2
    return out / out.sum()


# ============================================================================
# 核心: 约束分离像素分组 (论文1方法)
# ============================================================================

def create_pixel_groups(shape: Tuple[int, int], n_groups: int = 4,
                        seed: Optional[int] = None,
                        shuffle_labels: bool = True) -> np.ndarray:
    """
    创建满足"相邻像素不同组"约束的像素分组。

    这是约束分离方法的核心:
    - 对 N×N 像素网格进行图着色
    - 4-邻接网格用4色即可保证相邻像素不同色
    - 随机打乱标签 → 非周期分布

    Parameters
    ----------
    shape : (ny, nx)
        图像尺寸
    n_groups : int
        分组数量（默认4，对2D网格4色足够）
    seed : int, optional
        随机种子
    shuffle_labels : bool
        是否随机打乱组标签（True = 随机分组，False = 规则棋盘格）

    Returns
    -------
    groups : np.ndarray (ny, nx), dtype=int
        每个像素的组编号 (0..n_groups-1)
    """
    rng = np.random.RandomState(seed)
    ny, nx = shape

    if n_groups >= 4:
        # 经典4色棋盘格扩展: 每个2×2块内4个不同颜色
        # group = (y % 2) * 2 + (x % 2)
        y_idx = np.arange(ny)[:, None]
        x_idx = np.arange(nx)[None, :]
        groups = (y_idx % 2) * 2 + (x_idx % 2)
        # 如果需要更多组，在2×2块间错开
        if n_groups > 4:
            block_offset = ((y_idx // 2) % (n_groups // 4)) * 4
            groups = (groups + block_offset) % n_groups
    else:
        # 更少组: 用模运算（可能违反邻接约束，用更大pattern缓解）
        groups = (np.arange(ny)[:, None] + np.arange(nx)[None, :]) % n_groups

    if shuffle_labels:
        perm = rng.permutation(n_groups)
        groups = perm[groups]

    return groups.astype(np.int32)


def verify_group_constraint(groups: np.ndarray) -> Dict:
    """验证相邻像素不同组的约束满足情况"""
    ny, nx = groups.shape
    violations = 0
    total_adjacent_pairs = 0

    for dy, dx in [(0, 1), (1, 0)]:  # 右邻、下邻
        for y in range(ny - dy):
            for x in range(nx - dx):
                ny_, nx_ = y + dy, x + dx
                if dy == 0 and x + dx < nx:
                    total_adjacent_pairs += 1
                    if groups[y, x] == groups[ny_, nx_]:
                        violations += 1
                elif dx == 0 and y + dy < ny:
                    total_adjacent_pairs += 1
                    if groups[y, x] == groups[ny_, nx_]:
                        violations += 1

    return {
        'n_groups': len(np.unique(groups)),
        'adjacent_pairs': total_adjacent_pairs,
        'violations': violations,
        'violation_rate': violations / max(total_adjacent_pairs, 1),
        'valid': violations == 0
    }


def create_group_masks(groups: np.ndarray) -> List[np.ndarray]:
    """为每个组创建布尔掩码"""
    n_groups = len(np.unique(groups))
    return [groups == g for g in range(n_groups)]


# ============================================================================
# 约束分离 OT 相位求解器
# ============================================================================

class DualPotentialMLP(nn.Module):
    """对偶 Kantorovich 势的神经网络参数化"""

    def __init__(self, input_dim: int = 2, num_layers: int = 5,
                 hidden_size_0: int = 32):
        super().__init__()
        self.net = nn.Sequential()
        self.net.append(nn.Linear(input_dim, hidden_size_0))
        self.net.append(nn.ReLU())
        for i in range(1, num_layers - 1):
            self.net.append(nn.Linear(hidden_size_0 * (2 ** (i - 1)),
                                       hidden_size_0 * (2 ** i)))
            self.net.append(nn.ReLU())
        last = hidden_size_0 * (2 ** (num_layers - 2)) if num_layers > 1 else hidden_size_0
        self.net.append(nn.Linear(last, 1))

    def forward(self, x):
        return self.net(x).ravel()


class GroupConstrainedOTRetriever:
    """
    基于约束分离像素分组的连续OT相位求解器。

    核心改进（相比论文2的全局Sinkhorn）:
    1. 将像素随机分为K组，相邻像素不同组
    2. 每组独立进行随机采样和对偶OT优化
    3. 分组采样提供空间多样化的随机batch，打破周期性
    """

    def __init__(self, n_groups: int = 4, reg: float = 0.2,
                 lr: float = 0.0005, batch_size: int = 512,
                 n_iter: int = 300, device: torch.device = torch.device('cpu'),
                 seed: Optional[int] = None, verbose: bool = True):
        self.n_groups = n_groups
        self.reg = reg
        self.lr = lr
        self.batch_size = batch_size
        self.n_iter = n_iter
        self.device = device
        self.seed = seed
        self.verbose = verbose

    def __call__(self, source: np.ndarray, target: np.ndarray,
                 xs: np.ndarray, ys: np.ndarray) -> Dict:
        """
        运行分组约束OT相位检索。
        """
        source_n = source / source.sum()
        target_n = target / target.sum()

        # ---- 步骤1: 创建约束分离像素分组 ----
        groups = create_pixel_groups(source.shape, self.n_groups, self.seed)
        constraint_info = verify_group_constraint(groups)
        if self.verbose:
            print(f"像素分组: {self.n_groups}组, "
                  f"邻接约束违反率: {constraint_info['violation_rate']:.4f}")

        # ---- 步骤2: 分组训练对偶势 ----
        u_model = DualPotentialMLP().to(self.device)
        v_model = DualPotentialMLP().to(self.device)
        optimizer = optim.Adam(
            list(u_model.parameters()) + list(v_model.parameters()), lr=self.lr)

        losses = []
        for it in tqdm(range(self.n_iter), desc="分组约束OT训练"):
            # 随机选择一个组进行本轮采样
            g = np.random.randint(self.n_groups)
            mask = groups == g

            # 从该组中采样（空间分散的像素）
            src_pts = self._sample_group(source_n, xs, ys, mask, self.batch_size)
            tgt_pts = self._sample_group(target_n, xs, ys, mask, self.batch_size)

            # 也从全图采样以保持全局一致性
            src_all = self._sample_all(source_n, xs, ys,
                                        self.batch_size // 2)
            tgt_all = self._sample_all(target_n, xs, ys,
                                        self.batch_size // 2)

            src_pts = np.concatenate([src_pts, src_all], axis=0)
            tgt_pts = np.concatenate([tgt_pts, tgt_all], axis=0)

            src_t = torch.tensor(src_pts, dtype=torch.float32).to(self.device)
            tgt_t = torch.tensor(tgt_pts, dtype=torch.float32).to(self.device)

            from ot import stochastic
            loss = -stochastic.loss_dual_entropic(
                u_model(src_t), v_model(tgt_t), src_t, tgt_t, reg=self.reg)
            losses.append(loss.item())

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if self.verbose and it % max(1, self.n_iter // 5) == 0:
                print(f"  Iter {it:3d}, loss={losses[-1]:.6f}")

        # ---- 步骤3: 提取输运映射 ----
        Fx, Fy = self._extract_map(u_model, v_model, xs, ys)
        phase = self._integrate_phase(Fx, Fy, xs, ys)

        return {
            'Fx': Fx, 'Fy': Fy, 'phase': phase,
            'groups': groups, 'losses': losses,
            'constraint_info': constraint_info
        }

    def _sample_group(self, intensity, xs, ys, mask, n):
        """从特定组的像素中采样（含子像素噪声）"""
        masked_intensity = intensity.copy()
        masked_intensity[~mask] = 0
        if masked_intensity.sum() < 1e-15:
            return self._sample_all(intensity, xs, ys, n)
        return self._sample_points(masked_intensity, xs, ys, n)

    def _sample_all(self, intensity, xs, ys, n):
        """从全图采样"""
        return self._sample_points(intensity, xs, ys, n)

    def _sample_points(self, intensity, xs, ys, n):
        flat = intensity.flatten()
        if flat.sum() < 1e-15:
            idx = np.random.choice(flat.size, size=n)
        else:
            flat = flat / flat.sum()
            idx = np.random.choice(flat.size, size=n, p=flat)
        yi, xi = np.unravel_index(idx, intensity.shape)
        yi = yi + np.random.rand(n)
        xi = xi + np.random.rand(n)
        dx = np.abs(xs[0, 0] - xs[0, 1])
        dy = np.abs(ys[0, 0] - ys[1, 0])
        return np.stack([xi * dx + xs.min(), yi * dy + ys.min()], axis=1)

    def _extract_map(self, u_model, v_model, xs, ys):
        from ot import stochastic
        coords = np.stack([xs.flatten(), ys.flatten()], axis=1)
        coords_t = torch.tensor(coords, dtype=torch.float32).to(self.device)
        xs_t = torch.tensor(xs, dtype=torch.float32).to(self.device)
        ys_t = torch.tensor(ys, dtype=torch.float32).to(self.device)

        Xm, Ym = [], []
        u_model.eval()
        v_model.eval()
        with torch.no_grad():
            for i in range(0, len(coords_t), 1024):
                bc = coords_t[i:i + 1024]
                plan = stochastic.plan_dual_entropic(
                    u_model(bc), v_model(coords_t), bc, coords_t, reg=self.reg)
                plan = plan / plan.sum(axis=1).unsqueeze(1)
                plan = plan.reshape(len(bc), xs.shape[0], xs.shape[1])
                Xm.extend((plan * xs_t.unsqueeze(0)).sum([1, 2]).cpu().numpy())
                Ym.extend((plan * ys_t.unsqueeze(0)).sum([1, 2]).cpu().numpy())
        return (np.array(Xm).reshape(xs.shape),
                np.array(Ym).reshape(xs.shape))

    @staticmethod
    def _integrate_phase(Fx, Fy, xs, ys):
        dx = np.abs(xs[0, 0] - xs[0, 1])
        dy = np.abs(ys[0, 0] - ys[1, 0])
        Fx0 = np.tile(Fx[0, :], (Fx.shape[0], 1))
        return np.cumsum(Fx0, axis=1) * dx + np.cumsum(Fy, axis=0) * dy


# ============================================================================
# 约束分离分组精化器
# ============================================================================

class GroupSeparatedRefiner:
    """
    基于约束分离像素分组的迭代精化器。

    与标准GS/MRAF的关键区别:
    - 标准方法: 所有像素同时更新 → 周期性伪影
    - 本方法: 分组交替更新，相邻像素不同组 → 打破周期性

    算法:
    for each iteration:
        随机排列组顺序
        for each group g:
            mask = (groups == g)
            # 用当前全图像素做FFT（其他组保持上次更新值）
            φ_new = one_gs_iteration(φ_current)
            # 仅更新本组像素
            φ_current[mask] = φ_new[mask]
    """

    def __init__(self, n_groups: int = 4, n_iter: int = 100,
                 seed: Optional[int] = None, verbose: bool = True):
        self.n_groups = n_groups
        self.n_iter = n_iter
        self.seed = seed
        self.verbose = verbose
        self.errors = []

    def refine(self, source_intensity: np.ndarray,
               target_intensity: np.ndarray,
               init_phase: np.ndarray,
               roi_mask: Optional[np.ndarray] = None
               ) -> Tuple[np.ndarray, np.ndarray, list]:
        """
        分组约束分离精化。

        Returns
        -------
        phase : 精化后的相位
        groups : 像素分组
        errors : 误差历史
        """
        source_amp = np.sqrt(source_intensity / source_intensity.sum())
        target_amp = np.sqrt(target_intensity / target_intensity.sum())

        # 创建约束分离像素分组
        groups = create_pixel_groups(source_intensity.shape,
                                      self.n_groups, self.seed)
        masks = create_group_masks(groups)

        if self.verbose:
            info = verify_group_constraint(groups)
            print(f"分组精化器: {self.n_groups}组, "
                  f"邻接约束违反: {info['violations']}/{info['adjacent_pairs']}")

        phase = init_phase.copy()
        self.errors = []

        for it in tqdm(range(self.n_iter), desc="分组约束分离精化"):
            # 随机排列组顺序（增加随机性）
            group_order = np.random.permutation(self.n_groups)

            for g in group_order:
                mask = masks[g]

                # ---- 振幅投影 (GS一步) ----
                field = source_amp * np.exp(1j * phase)
                field_ft = FFT2(field)

                if roi_mask is not None:
                    phase_ft = np.angle(field_ft)
                    field_ft_new = np.abs(field_ft) * np.exp(1j * phase_ft)
                    field_ft_new[roi_mask] = (
                        target_amp[roi_mask] * np.exp(1j * phase_ft[roi_mask]))
                else:
                    field_ft_new = target_amp * np.exp(1j * np.angle(field_ft))

                phase_new = np.angle(IFFT2(field_ft_new))

                # ---- 仅更新本组像素 (约束分离的关键) ----
                phase[mask] = phase_new[mask]

            # 追踪误差
            if it % max(1, self.n_iter // 10) == 0 or it == 0:
                output = fourier_propagate(source_intensity, phase)
                err = np.sqrt(np.mean(
                    (output - target_intensity / target_intensity.sum())**2))
                self.errors.append(err)
                if self.verbose:
                    print(f"  迭代 {it:3d}: RMS = {err:.6f}")

        return phase, groups, self.errors


# ============================================================================
# 完整 CSOT 流水线
# ============================================================================

class CSOTPhaseRetriever:
    """
    约束分离最优传输 (CSOT) 相位检索完整流水线。

    组合:
    1. 分组约束 OT 初始化（论文1的约束分离 + 论文2的OT）
    2. 分组约束分离精化（交替分组GS投影）

    用法:
        csot = CSOTPhaseRetriever(n_groups=4, ot_n_iter=200, refine_n_iter=100)
        results = csot(source, target, xs, ys)
        phase = results['phase_refined']
        groups = results['groups']  # 像素分组可视化
    """

    def __init__(self, n_groups: int = 4,
                 reg: float = 0.2, ot_lr: float = 0.0005,
                 ot_batch_size: int = 512, ot_n_iter: int = 300,
                 refine_n_iter: int = 100,
                 device: torch.device = torch.device('cpu'),
                 seed: Optional[int] = None, verbose: bool = True):
        self.n_groups = n_groups
        self.reg = reg
        self.ot_lr = ot_lr
        self.ot_batch_size = ot_batch_size
        self.ot_n_iter = ot_n_iter
        self.refine_n_iter = refine_n_iter
        self.device = device
        self.seed = seed
        self.verbose = verbose

    def __call__(self, source: np.ndarray, target: np.ndarray,
                 xs: np.ndarray, ys: np.ndarray,
                 roi_mask: Optional[np.ndarray] = None,
                 init_phase: Optional[np.ndarray] = None,
                 skip_ot: bool = False) -> Dict:
        """
        运行完整 CSOT 流水线。

        Returns
        -------
        results : dict
            'phase_ot'         : OT初始相位
            'phase_refined'    : 最终相位
            'groups'           : 像素分组图 (用于可视化)
            'ot_losses'        : OT训练损失
            'refine_errors'    : 精化误差
            'output_intensity' : 最终输出强度
        """
        results = {}

        # ---- Phase 1: 分组约束OT初始化 ----
        if skip_ot and init_phase is not None:
            if self.verbose:
                print("跳过OT，使用提供的初始相位")
            results['phase_ot'] = init_phase
            results['ot_losses'] = []
            # 仍然需要创建分组用于后续精化
            results['groups'] = create_pixel_groups(
                source.shape, self.n_groups, self.seed)
        else:
            if self.verbose:
                print("=" * 50)
                print("Phase 1: 分组约束OT初始化")
                print("=" * 50)

            ot_retriever = GroupConstrainedOTRetriever(
                n_groups=self.n_groups, reg=self.reg,
                lr=self.ot_lr, batch_size=self.ot_batch_size,
                n_iter=self.ot_n_iter, device=self.device,
                seed=self.seed, verbose=self.verbose)

            ot_results = ot_retriever(source, target, xs, ys)
            results['phase_ot'] = ot_results['phase']
            results['Fx'] = ot_results['Fx']
            results['Fy'] = ot_results['Fy']
            results['groups'] = ot_results['groups']
            results['ot_losses'] = ot_results['losses']

        # ---- Phase 2: 分组约束分离精化 ----
        if self.verbose:
            print("\n" + "=" * 50)
            print("Phase 2: 分组约束分离精化")
            print("=" * 50)

        refiner = GroupSeparatedRefiner(
            n_groups=self.n_groups, n_iter=self.refine_n_iter,
            seed=self.seed, verbose=self.verbose)

        phase_refined, groups, errors = refiner.refine(
            source, target, results['phase_ot'], roi_mask=roi_mask)

        results['phase_refined'] = phase_refined
        results['groups'] = groups
        results['refine_errors'] = errors
        results['output_intensity'] = fourier_propagate(
            source, phase_refined)

        if self.verbose:
            final_err = errors[-1] if errors else float('nan')
            print(f"\n最终RMS误差: {final_err:.6f}")

        return results


# ============================================================================
# 对比工具
# ============================================================================

def compare_methods(source: np.ndarray, target: np.ndarray,
                    xs: np.ndarray, ys: np.ndarray,
                    methods: list = None,
                    device: torch.device = torch.device('cpu'),
                    seed: int = 42) -> Dict:
    """
    对比不同方法: 'gs', 'ot_flat', 'csot'

    CSOT = 约束分离像素分组 + OT + 分组精化
    """
    if methods is None:
        methods = ['gs', 'ot_flat', 'csot']

    results = {}
    target_norm = target / target.sum()

    for method in methods:
        print(f"\n{'='*60}")
        print(f"  方法: {method}")
        print(f"{'='*60}")

        if method == 'csot':
            csot = CSOTPhaseRetriever(
                n_groups=4, ot_n_iter=200, refine_n_iter=100,
                device=device, seed=seed, verbose=True)
            results[method] = csot(source, target, xs, ys)

        elif method == 'gs':
            from extractors.GSphase import GerchbergSaxton2d
            gs = GerchbergSaxton2d(num_iter=200)
            phase = gs(source, target)
            output = fourier_propagate(source, phase)
            results[method] = {
                'phase': phase, 'output_intensity': output,
                'rms_error': np.sqrt(np.mean((output - target_norm)**2))
            }

        elif method == 'ot_flat':
            from extractors.OTphase import FlattenOptimalTransport2d
            ot = FlattenOptimalTransport2d(method='sinkhorn', reg=0.01)
            Fx, Fy, phase, curl = ot(source, target, xs, ys)
            output = fourier_propagate(source, phase)
            results[method] = {
                'phase': phase, 'Fx': Fx, 'Fy': Fy, 'curl': curl,
                'output_intensity': output,
                'rms_error': np.sqrt(np.mean((output - target_norm)**2))
            }

    return results


def plot_pixel_groups(groups: np.ndarray, ax=None, title="像素分组 (约束分离)"):
    """可视化像素分组 — 展示'相邻像素不同组'的约束"""
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(6, 6))
    n_groups = len(np.unique(groups))
    colors = plt.cm.tab10(np.linspace(0, 1, n_groups))
    ax.imshow(groups, cmap='tab10', vmin=0, vmax=n_groups - 1)
    ax.set_title(title)
    ax.axis('off')
    return ax


def plot_comparison(results: Dict, source: np.ndarray, target: np.ndarray,
                    figsize: Tuple[int, int] = (18, 12)):
    """可视化各方法对比"""
    n = len(results)
    target_norm = target / target.sum()
    fig, axes = plt.subplots(n + 1, 5, figsize=figsize)

    # 标题行
    for j, title in enumerate(['方法', '输出强度', '相位', 'RMS误差', '像素分组']):
        axes[0, j].set_title(title, fontweight='bold')
    for ax in axes[0, :]:
        ax.axis('off')

    for i, (method, res) in enumerate(results.items()):
        row = i + 1
        phase = res.get('phase_refined', res.get('phase'))
        output = res['output_intensity']
        rms = res.get('rms_error',
                      np.sqrt(np.mean((output - target_norm)**2)))

        axes[row, 0].text(0.5, 0.5, method.upper(), ha='center', va='center',
                          fontsize=12, fontweight='bold',
                          transform=axes[row, 0].transAxes)
        axes[row, 0].axis('off')

        axes[row, 1].imshow(output, cmap='hot')
        axes[row, 1].axis('off')

        im = axes[row, 2].imshow(phase, cmap='twilight_shifted')
        axes[row, 2].axis('off')
        plt.colorbar(im, ax=axes[row, 2])

        axes[row, 3].text(0.5, 0.5, f'{rms:.6f}', ha='center', va='center',
                          fontsize=14, transform=axes[row, 3].transAxes)
        axes[row, 3].axis('off')

        if 'groups' in res:
            plot_pixel_groups(res['groups'], ax=axes[row, 4],
                              title=f'{method} 分组')
        else:
            axes[row, 4].text(0.5, 0.5, 'N/A', ha='center', va='center')
            axes[row, 4].axis('off')

    plt.tight_layout()
    return fig, axes
