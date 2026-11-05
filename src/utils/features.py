"""Rich ECG feature extraction shared by training and the API (v2)."""

import numpy as np
from scipy.signal import find_peaks, welch, butter, filtfilt

N_LEADS = 12
FS = 100
BEAT_PTS = 16  # points per lead in the downsampled median beat


def _bandpass(x, fs=FS):
    b, a = butter(2, [0.5 / (fs / 2), 40 / (fs / 2)], btype="band")
    return filtfilt(b, a, x, axis=0)


def _beat_template(sig):
    """Median beat (window -0.25s..+0.45s around R peaks) per lead + rate features."""
    ref = sig[:, 1] if np.std(sig[:, 1]) > 0 else sig[:, 0]
    ref = np.abs(ref - np.median(ref))
    thr = 0.5 * np.percentile(ref, 99)
    peaks, _ = find_peaks(ref, height=thr, distance=int(0.3 * FS))
    pre, post = int(0.25 * FS), int(0.45 * FS)
    beats = [
        sig[p - pre : p + post]
        for p in peaks
        if p - pre >= 0 and p + post <= sig.shape[0]
    ]
    if len(beats) >= 2:
        med = np.median(np.stack(beats), axis=0)
    else:
        med = np.zeros((pre + post, N_LEADS))
    idx = np.linspace(0, pre + post - 1, BEAT_PTS).astype(int)
    tmpl = med[idx].T.ravel()
    rr = np.diff(peaks) / FS if len(peaks) > 2 else np.array([0.0])
    rate = [
        len(peaks),
        60.0 / rr.mean() if rr.mean() > 0 else 0.0,
        rr.std(),
        np.sqrt(np.mean(np.diff(rr) ** 2)) if len(rr) > 1 else 0.0,
        len(beats),
    ]
    return tmpl, rate


def extract_features(signal_array: np.ndarray) -> np.ndarray:
    sig = np.nan_to_num(np.asarray(signal_array, dtype=float))
    try:
        sig = _bandpass(sig)
    except Exception:
        pass
    feats = []
    for ch in range(N_LEADS):
        d = sig[:, ch]
        diff = np.diff(d)
        f, p = welch(d, fs=FS, nperseg=min(256, len(d)))
        tot = p.sum() + 1e-12
        bands = [(0.5, 3), (3, 8), (8, 15), (15, 25), (25, 40)]
        bp = [p[(f >= lo) & (f < hi)].sum() / tot for lo, hi in bands]
        c = d - d.mean()
        sd = d.std() + 1e-12
        feats.extend(
            [
                d.mean(), d.std(), d.min(), d.max(), np.percentile(d, 5),
                np.percentile(d, 25), np.percentile(d, 50), np.percentile(d, 75),
                np.percentile(d, 95), np.ptp(d),
                np.mean(c**3) / sd**3, np.mean(c**4) / sd**4,
                np.abs(diff).mean(), diff.std(), np.abs(diff).max(),
                np.mean(np.diff(np.sign(c)) != 0),
                f[np.argmax(p)],
                *bp,
            ]
        )
    tmpl, rate = _beat_template(sig)
    feats.extend(tmpl)
    feats.extend(rate)
    # inter-lead correlations with lead II / V1 / V5
    cc = np.corrcoef(sig.T)
    cc = np.nan_to_num(cc)
    feats.extend(cc[1][[0, 2, 6, 10]])
    feats.extend(cc[6][[10, 11]])
    return np.nan_to_num(np.array(feats, dtype=float))


def validate_signal(sig, min_len=100, max_abs_mv=50.0):
    """Raise ValueError if the ECG array is not a usable (time, 12) signal."""
    sig = np.asarray(sig)
    if sig.ndim != 2 or sig.shape[1] != N_LEADS:
        raise ValueError(f"Signal must have shape (time_steps, {N_LEADS}); got {sig.shape}")
    if sig.shape[0] < min_len:
        raise ValueError(f"Signal too short: {sig.shape[0]} < {min_len} samples")
    if not np.isfinite(sig).all():
        raise ValueError("Signal contains NaN or infinite values")
    if np.abs(sig).max() > max_abs_mv:
        raise ValueError(f"Signal amplitude exceeds +/-{max_abs_mv} mV")
