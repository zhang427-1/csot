"""
Paper-Style Comparison: Standard GS vs CSOT (Grouped GS)
=========================================================
Reproduces targets from Pasienski & DeMarco 2008 (as in arXiv:2408.17025),
compares Standard GS vs CSOT (Grouped GS with constraint separation).

Metrics: SchroffError (paper's main metric) + RMS Error
"""

import numpy as np
from scipy.fftpack import fft2, ifft2, fftshift, ifftshift
from scipy.ndimage import gaussian_filter
import time


def FFT2(x): return fftshift(fft2(ifftshift(x)))
def IFFT2(x): return fftshift(ifft2(ifftshift(x)))


def create_groups(shape, n_groups=4, seed=42):
    rng = np.random.RandomState(seed)
    ny, nx = shape; yi = np.arange(ny)[:, None]; xi = np.arange(nx)[None, :]
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
    return np.sqrt(np.mean((target / target.sum() - output / output.sum())**2))


def schroff_error(target, output, threshold=0.5):
    """Paper's SchroffError with safe zero handling"""
    mask = target > threshold * target.max()
    if mask.sum() < 10:
        return np.nan
    x = target[mask].copy(); x /= x.sum()
    y = output[mask].copy(); y /= max(y.sum(), 1e-30)
    # Avoid division by zero
    safe_x = np.where(x > 1e-15, x, 1.0)
    return np.sqrt(np.sum(np.where(x > 1e-15, (x - y)**2 / safe_x**2, 0)) / mask.sum())


# ================================================================
# Targets (simplified but keeping paper's shapes)
# ================================================================
def make_targets(N):
    c = N // 2
    xs = np.arange(N) - c; ys = np.arange(N) - c
    X, Y = np.meshgrid(xs, ys)
    R = np.sqrt(X**2 + Y**2)

    targets = {}

    # 1. Diadem: ring + 2 bright spots
    r, w = N * 0.21, N * 0.027
    ring = np.exp(-(R - r)**2 / (2 * w**2))
    jewels = np.exp(-(np.abs(X) - r)**2 / (2 * w**2)) * np.exp(-Y**2 / (2 * w**2))
    diadem = ring + 3 * jewels
    diadem = gaussian_filter(diadem, sigma=0.8)
    targets['1.Diadem'] = diadem / diadem.sum()

    # 2. Shuriken: 3-pointed star
    n_pts = 3
    Rin = N * 0.14
    Rout = N * 0.16
    thetas = [np.pi + 2 * np.pi * j / n_pts for j in range(n_pts)]
    star = np.ones((N, N), dtype=bool)
    for t in thetas:
        ct, st = np.cos(t), np.sin(t)
        cond = (ct * X + st * Y) < Rin + np.abs(ct * Y - st * X) * \
               (np.cos(np.pi / n_pts) - Rin / max(Rout, Rin + 1)) / max(np.sin(np.pi / n_pts), 0.01)
        star = star & cond
    star = gaussian_filter(star.astype(float), sigma=1.2)
    targets['2.Shuriken'] = star / star.sum()

    # 3. Fourgon: square with rounded corners
    s = int(N * 0.2)
    square = np.zeros((N, N))
    square[c - s:c + s, c - s:c + s] = 1.0
    square = gaussian_filter(square, sigma=3.0)
    targets['3.Fourgon'] = square / square.sum()

    # 4. OR Gate: lithography pattern
    orgate = np.zeros((N, N))
    hw, hh = int(N * 0.08), int(N * 0.12)
    orgate[c - hh:c + hh, c - hw:c + hw] = 1.0     # main bar
    orgate[c - hh - int(N*0.06):c - hh, c - hw:c + hw] = 1.0  # ext
    orgate[c + hh:c + hh + int(N*0.06), c - hw:c + hw] = 1.0  # foot
    orgate[c - hh - int(N*0.05):c - hh + int(N*0.01), c - hw//2:c + hw//2] = 0.0  # hole
    orgate = gaussian_filter(orgate, sigma=0.8)
    targets['4.OR Gate'] = orgate / orgate.sum()

    # 5. Squid: ring + tail with gap
    r2, w2 = N * 0.21, N * 0.027
    ring2 = np.exp(-(R - r2)**2 / (2 * w2**2))
    tail_len = N * 0.07
    tail = ((np.abs(X) < r2 + tail_len) & (np.abs(X) >= r2)).astype(float)
    tail *= np.exp(-Y**2 / (2 * w2**2))
    squid = np.maximum(ring2, tail)
    gap = np.where(np.abs(Y) < N * 0.04, 0.3, 1.0)
    squid *= gap
    squid = gaussian_filter(squid, sigma=0.8)
    targets['5.Squid'] = squid / squid.sum()

    # 6. Q-tip: line + round ends
    wire_len = N * 0.5
    wire_w = N * 0.014
    tip_r = N * 0.07
    wire = (np.abs(X) < wire_len / 2).astype(float) * np.exp(-Y**2 / (2 * wire_w**2))
    tips = np.exp(-(np.abs(X) - wire_len / 2)**2 / (2 * tip_r**2)) * \
           np.exp(-Y**2 / (2 * tip_r**2))
    qtip = np.maximum(wire, tips)
    qtip = gaussian_filter(qtip, sigma=0.6)
    targets['6.Q-tip'] = qtip / qtip.sum()

    return targets


# ================================================================
# GS methods
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
# Main comparison
# ================================================================
def run_comparison(N=256, n_iter=100, n_trials=3):
    print("=" * 75)
    print(f"  Paper-Style Comparison: Standard GS vs CSOT (Grouped GS)")
    print(f"  Grid: {N}x{N}, Iterations: {n_iter}, Trials: {n_trials}")
    print("=" * 75)

    xs = np.linspace(-N/2, N/2, N); ys = np.linspace(-N/2, N/2, N)
    X, Y = np.meshgrid(xs, ys)
    source = np.exp(-(X**2 + Y**2) / (2 * (N/4.5)**2))
    source /= source.sum()
    src_amp = np.sqrt(source)

    targets = make_targets(N)

    # Header
    print(f"\n  {'Target':14s} {'Method':20s} {'RMS Error':>12s} "
          f"{'SchroffErr':>12s} {'Improv(RMS)':>13s}")
    print(f"  {'-' * 73}")

    all_rms_gs, all_rms_grp = [], []

    for name, tgt_intensity in targets.items():
        tgt_amp = np.sqrt(tgt_intensity)

        rms_gs_vals, rms_grp_vals = [], []
        se_gs_vals, se_grp_vals = [], []

        for trial in range(n_trials):
            np.random.seed(42 + trial)
            init = np.random.randn(N, N) * 0.1

            # Standard GS
            ph_gs = standard_gs(src_amp, tgt_amp, init.copy(), n_iter)
            out_gs = np.abs(FFT2(src_amp * np.exp(1j * ph_gs)))**2
            out_gs /= out_gs.sum()
            rms_gs_vals.append(rms_error(tgt_intensity, out_gs))
            se_gs_vals.append(schroff_error(tgt_intensity, out_gs))

            # Grouped GS
            np.random.seed(42 + trial)
            init2 = np.random.randn(N, N) * 0.1
            ph_grp = grouped_gs(src_amp, tgt_amp, init2, n_iter, 4)
            out_grp = np.abs(FFT2(src_amp * np.exp(1j * ph_grp)))**2
            out_grp /= out_grp.sum()
            rms_grp_vals.append(rms_error(tgt_intensity, out_grp))
            se_grp_vals.append(schroff_error(tgt_intensity, out_grp))

        rms_gs = np.mean(rms_gs_vals)
        rms_grp = np.mean(rms_grp_vals)
        se_gs = np.nanmean(se_gs_vals) if not all(np.isnan(se_gs_vals)) else np.nan
        se_grp = np.nanmean(se_grp_vals) if not all(np.isnan(se_grp_vals)) else np.nan
        imp = (1 - rms_grp / rms_gs) * 100 if rms_gs > 0 else 0

        all_rms_gs.append(rms_gs)
        all_rms_grp.append(rms_grp)

        print(f"  {name:14s} {'Standard GS':20s} {rms_gs:12.6f} {se_gs:12.6f}")
        print(f"  {'':14s} {'CSOT (Grouped GS)':20s} {rms_grp:12.6f} {se_grp:12.6f} "
              f"{imp:>+11.1f}%")
        print(f"  {'':14s} {'-' * 55}")

    avg_rms_gs = np.mean(all_rms_gs)
    avg_rms_grp = np.mean(all_rms_grp)
    avg_imp = (1 - avg_rms_grp / avg_rms_gs) * 100

    print(f"\n  {'AVERAGE':14s} {'Standard GS':20s} {avg_rms_gs:12.6f}")
    print(f"  {'':14s} {'CSOT (Grouped GS)':20s} {avg_rms_grp:12.6f} "
          f"{avg_imp:>+11.1f}%")
    print(f"\n  Average improvement across all 6 targets: {avg_imp:.1f}%")

    # Count wins
    wins = sum(1 for g, s in zip(all_rms_grp, all_rms_gs) if g < s)
    print(f"  CSOT better on: {wins}/6 targets")

    return avg_imp


if __name__ == '__main__':
    for N in [128, 256]:
        for n_iter in [50, 100]:
            print("\n")
            run_comparison(N=N, n_iter=n_iter, n_trials=3)

    print("\n\n" + "=" * 75)
    print("  NOTE: Paper uses N=1536 grid with OT initialization.")
    print("  This test uses RANDOM initialization at lower resolution.")
    print("  The relative improvement of Grouped GS over Standard GS")
    print("  is the key metric, as it would compound with OT initialization.")
    print("=" * 75)
