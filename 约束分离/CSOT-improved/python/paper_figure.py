"""
Generate paper-style comparison figure: Standard GS vs CSOT (Grouped GS)
Matching the visual format from arXiv:2408.17025 — 6 targets, method rows.
"""

import numpy as np
from scipy.fftpack import fft2, ifft2, fftshift, ifftshift
from scipy.ndimage import gaussian_filter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import time


def FFT2(x): return fftshift(fft2(ifftshift(x)))
def IFFT2(x): return fftshift(ifft2(ifftshift(x)))


def create_groups(shape, n_groups=4, seed=42):
    rng = np.random.RandomState(seed)
    ny, nx = shape
    yi = np.arange(ny)[:, None]
    xi = np.arange(nx)[None, :]
    if n_groups >= 4:
        groups = (yi % 2) * 2 + (xi % 2)
        if n_groups > 4:
            groups = (groups + ((yi // 2) % (n_groups // 4)) * 4) % n_groups
    elif n_groups == 2: groups = (yi + xi) % 2
    elif n_groups == 3: groups = (yi + 2 * xi) % 3
    return rng.permutation(n_groups)[groups].astype(np.int32)


# ================================================================
# Metrics
# ================================================================
def rms_error(target, output):
    t = target / target.sum()
    o = output / output.sum()
    return np.sqrt(np.mean((t - o)**2))


def schroff_error(target, output, threshold=0.5):
    mask = target > threshold * target.max()
    if mask.sum() < 10: return np.nan
    x = target[mask].copy(); x /= x.sum()
    y = output[mask].copy(); y /= max(y.sum(), 1e-30)
    safe_x = np.where(x > 1e-15, x, 1.0)
    return np.sqrt(np.sum(np.where(x > 1e-15, (x - y)**2 / safe_x**2, 0)) / mask.sum())


# ================================================================
# Targets
# ================================================================
def make_targets(N):
    c = N // 2
    xs = np.arange(N) - c; ys = np.arange(N) - c
    X, Y = np.meshgrid(xs, ys)
    R = np.sqrt(X**2 + Y**2)
    targets = {}

    # 1. Diadem
    r, w = N * 0.21, N * 0.027
    ring = np.exp(-(R - r)**2 / (2 * w**2))
    jewels = np.exp(-(np.abs(X) - r)**2 / (2 * w**2)) * np.exp(-Y**2 / (2 * w**2))
    diadem = gaussian_filter(ring + 3 * jewels, sigma=0.8)
    targets['Diadem'] = diadem / diadem.sum()

    # 2. Shuriken
    n_pts, Rin, Rout = 3, N * 0.14, N * 0.16
    thetas = [np.pi + 2*np.pi*j/n_pts for j in range(n_pts)]
    star = np.ones((N, N), dtype=bool)
    for t in thetas:
        ct, st = np.cos(t), np.sin(t)
        cond = (ct*X + st*Y) < Rin + np.abs(ct*Y - st*X) * \
               (np.cos(np.pi/n_pts) - Rin/max(Rout, Rin+1)) / max(np.sin(np.pi/n_pts), 0.01)
        star = star & cond
    star = gaussian_filter(star.astype(float), sigma=1.2)
    targets['Shuriken'] = star / star.sum()

    # 3. Fourgon
    s = int(N * 0.2)
    square = np.zeros((N, N))
    square[c-s:c+s, c-s:c+s] = 1.0
    square = gaussian_filter(square, sigma=3.0)
    targets['Fourgon'] = square / square.sum()

    # 4. OR Gate
    hw, hh = int(N*0.08), int(N*0.12)
    orgate = np.zeros((N, N))
    orgate[c-hh:c+hh, c-hw:c+hw] = 1.0
    orgate[c-hh-int(N*0.06):c-hh, c-hw:c+hw] = 1.0
    orgate[c+hh:c+hh+int(N*0.06), c-hw:c+hw] = 1.0
    orgate[c-hh-int(N*0.05):c-hh+int(N*0.01), c-hw//2:c+hw//2] = 0.0
    orgate = gaussian_filter(orgate, sigma=0.8)
    targets['OR Gate'] = orgate / orgate.sum()

    # 5. Squid
    r2, w2 = N*0.21, N*0.027
    ring2 = np.exp(-(R - r2)**2 / (2*w2**2))
    tail_len = N*0.07
    tail = ((np.abs(X) < r2+tail_len) & (np.abs(X) >= r2)).astype(float)
    tail *= np.exp(-Y**2 / (2*w2**2))
    squid = np.maximum(ring2, tail)
    squid *= np.where(np.abs(Y) < N*0.04, 0.3, 1.0)
    squid = gaussian_filter(squid, sigma=0.8)
    targets['Squid'] = squid / squid.sum()

    # 6. Q-tip
    wlen, ww, tr = N*0.5, N*0.014, N*0.07
    wire = (np.abs(X) < wlen/2).astype(float) * np.exp(-Y**2 / (2*ww**2))
    tips = np.exp(-(np.abs(X) - wlen/2)**2 / (2*tr**2)) * np.exp(-Y**2 / (2*tr**2))
    qtip = gaussian_filter(np.maximum(wire, tips), sigma=0.6)
    targets['Q-tip'] = qtip / qtip.sum()

    return targets


# ================================================================
# GS Methods
# ================================================================
def standard_gs(src_amp, tgt_amp, init_phase, n_iter):
    phase = init_phase.copy()
    for _ in range(n_iter):
        ft = FFT2(src_amp * np.exp(1j * phase))
        ft = tgt_amp * np.exp(1j * np.angle(ft))
        phase = np.angle(IFFT2(ft))
    return phase


def grouped_gs(src_amp, tgt_amp, init_phase, n_iter, n_groups=4):
    groups = create_groups(src_amp.shape, n_groups)
    masks = [(groups == g) for g in range(n_groups)]
    phase = init_phase.copy()
    for _ in range(n_iter):
        for g in np.random.permutation(n_groups):
            m = masks[g]
            ft = FFT2(src_amp * np.exp(1j * phase))
            ft = tgt_amp * np.exp(1j * np.angle(ft))
            pnew = np.angle(IFFT2(ft))
            phase[m] = pnew[m]
    return phase


# ================================================================
# Paper-style Figure
# ================================================================
def generate_paper_figure(N=256, n_iter=100):
    """Generate figure matching paper's visual comparison format."""
    print(f"Generating paper-style figure (N={N}, iter={n_iter})...")

    # Source
    xs = np.linspace(-N/2, N/2, N); ys = np.linspace(-N/2, N/2, N)
    X, Y = np.meshgrid(xs, ys)
    source = np.exp(-(X**2 + Y**2) / (2 * (N/4.5)**2))
    source /= source.sum()
    src_amp = np.sqrt(source)

    # Targets
    target_names = ['Diadem', 'Shuriken', 'Fourgon', 'OR Gate', 'Squid', 'Q-tip']
    targets = make_targets(N)

    # Run methods
    np.random.seed(42)
    init_phase = np.random.randn(N, N) * 0.1

    results = {}
    print("  Running Standard GS...")
    t0 = time.time()
    for name in target_names:
        tgt_amp = np.sqrt(targets[name])
        ph = standard_gs(src_amp, tgt_amp, init_phase.copy(), n_iter)
        out = np.abs(FFT2(src_amp * np.exp(1j * ph)))**2
        out /= out.sum()
        results[(name, 'Standard GS')] = {
            'phase': ph, 'output': out,
            'rms': rms_error(targets[name], out),
            'schroff': schroff_error(targets[name], out)
        }
    print(f"    done ({time.time()-t0:.1f}s)")

    print("  Running CSOT (Grouped GS)...")
    t0 = time.time()
    for name in target_names:
        tgt_amp = np.sqrt(targets[name])
        ph = grouped_gs(src_amp, tgt_amp, init_phase.copy(), n_iter, 4)
        out = np.abs(FFT2(src_amp * np.exp(1j * ph)))**2
        out /= out.sum()
        results[(name, 'CSOT')] = {
            'phase': ph, 'output': out,
            'rms': rms_error(targets[name], out),
            'schroff': schroff_error(targets[name], out)
        }
    print(f"    done ({time.time()-t0:.1f}s)")

    # ================================================================
    # Create figure: 3 rows x 6 columns
    # Row 0: Target patterns
    # Row 1: Standard GS outputs
    # Row 2: CSOT outputs
    # ================================================================
    fig = plt.figure(figsize=(22, 12))
    gs = GridSpec(4, 6, figure=fig, hspace=0.35, wspace=0.08,
                  height_ratios=[1, 1, 1, 0.15])

    row_labels = ['Target', 'Standard GS\n(Paper 2 method)', 'CSOT\n(Grouped GS, Ours)']
    methods_for_rows = [None, 'Standard GS', 'CSOT']

    for row in range(3):
        for col, name in enumerate(target_names):
            ax = fig.add_subplot(gs[row, col])

            if row == 0:
                # Show target
                img = targets[name]
                ax.imshow(img, cmap='hot', interpolation='bilinear')
                ax.set_title(name, fontsize=12, fontweight='bold', pad=8)
            else:
                method = methods_for_rows[row]
                res = results[(name, method)]
                img = res['output']
                ax.imshow(img, cmap='hot', interpolation='bilinear')
                # Metrics below image
                rms = res['rms']
                se = res['schroff']
                if np.isnan(se):
                    ax.set_xlabel(f'RMS={rms:.2e}', fontsize=9)
                else:
                    ax.set_xlabel(f'RMS={rms:.2e}  SE={se:.4f}', fontsize=9)

            # Row label on leftmost column
            if col == 0:
                ax.set_ylabel(row_labels[row], fontsize=11, fontweight='bold',
                             rotation=90, labelpad=15, va='center')
            ax.set_xticks([])
            ax.set_yticks([])

    # ---- Add improvement annotations on CSOT row ----
    for col, name in enumerate(target_names):
        ax = fig.add_subplot(gs[2, col])
        gs_rms = results[(name, 'Standard GS')]['rms']
        csot_rms = results[(name, 'CSOT')]['rms']
        imp = (1 - csot_rms/gs_rms) * 100
        color = '#2ecc71' if imp > 0 else '#e74c3c'
        # Add improvement text at bottom
        ax.text(0.5, -0.08, f'{imp:+.1f}%', transform=ax.transAxes,
                ha='center', fontsize=13, fontweight='bold', color=color)

    # ---- Colorbar ----
    cbar_ax = fig.add_subplot(gs[3, :])
    norm = plt.Normalize(0, 1)
    sm = plt.cm.ScalarMappable(cmap='hot', norm=norm)
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation='horizontal')
    cbar.set_label('Normalized Intensity', fontsize=11)

    # ---- Title ----
    fig.suptitle(
        f'Beam Shaping Comparison: Standard GS vs CSOT (Constraint-Separated Grouped GS)\n'
        f'Grid: {N}x{N}, {n_iter} iterations, 6 targets from Pasienski & DeMarco 2008',
        fontsize=14, fontweight='bold', y=1.01
    )

    # ---- Metrics summary table below figure ----
    print(f"\n  {'Target':12s} {'GS RMS':>10s} {'CSOT RMS':>10s} "
          f"{'Improv':>10s} {'GS SE':>10s} {'CSOT SE':>10s}")
    print(f"  {'-' * 66}")
    for name in target_names:
        gs_r = results[(name, 'Standard GS')]['rms']
        cs_r = results[(name, 'CSOT')]['rms']
        gs_s = results[(name, 'Standard GS')]['schroff']
        cs_s = results[(name, 'CSOT')]['schroff']
        imp = (1 - cs_r/gs_r) * 100
        print(f"  {name:12s} {gs_r:10.2e} {cs_r:10.2e} {imp:>+9.1f}% "
              f"{gs_s:10.4f} {cs_s:10.4f}")

    avg_imp = np.mean([(1 - results[(n, 'CSOT')]['rms'] / results[(n, 'Standard GS')]['rms']) * 100
                       for n in target_names])
    print(f"\n  Average improvement: {avg_imp:.1f}%")

    # Save
    out_path = 'paper_comparison_figure.png'
    fig.savefig(out_path, dpi=200, bbox_inches='tight', facecolor='white')
    print(f"\n  Figure saved to: {out_path}")

    return fig


if __name__ == '__main__':
    fig = generate_paper_figure(N=256, n_iter=100)
    plt.close()
    print("\nDone!")
