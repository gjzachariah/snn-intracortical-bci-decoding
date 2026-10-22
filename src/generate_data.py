"""
generate_data.py
================
Synthetic spike-train generator for the M.Tech thesis
"Synthetic Spike-Based Motor Intent Decoding Using Spiking Neural Networks".

What it does
------------
1. Reads the real-primate tuning table you already produced
   (results/tables/primate_tuning_curves.csv: 103 units x 8 direction rates).
2. Fits a Georgopoulos-style cosine tuning curve to every real unit:
        rate(direction) = baseline + amplitude * cos(direction - preferred_direction)
3. Uses those fitted values (baseline rate, tuning amplitude, and optionally
   preferred direction) as the *seed* for a population of synthetic neurons.
4. Simulates 8-direction centre-out reaching trials (hold -> go ramp -> movement)
   and turns each neuron's firing rate into binary spike trains.
5. Saves everything (spikes, labels, train/val/test split, neuron parameters,
   settings) so the same dataset can be re-created or re-loaded later.

Run from the repository root, for example:
    python src/generate_data.py --channels 96 --trials-per-direction 100 --seed 0

Only numpy and pandas are required.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------
N_DIRECTIONS = 8
DIRECTION_NAMES = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
# Direction k sits at k * 45 degrees, in the same order as the CSV columns.
# Only angle *differences* matter for the cosine model, so the absolute
# starting point (N = 0) does not change any result.
DIRECTION_ANGLES = np.arange(N_DIRECTIONS) * 2 * np.pi / N_DIRECTIONS

# The synthetic population is always built at this size, and smaller channel
# counts take the first n neurons. That makes 16, 32, ..., 1024 channel sets
# *nested* (the 64-channel set contains the 32-channel set), so a channel-count
# sweep compares like with like.
MAX_CHANNELS = 1024

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CALIBRATION_CSV = REPO_ROOT / "results" / "tables" / "primate_tuning_curves.csv"
DEFAULT_OUT_DIR = REPO_ROOT / "data" / "synthetic"


# --------------------------------------------------------------------------
# Settings
# --------------------------------------------------------------------------
@dataclass
class GeneratorConfig:
    # --- dataset size -------------------------------------------------
    n_channels: int = 96            # number of input neurons (16 ... 1024)
    trials_per_direction: int = 100  # total trials = 8 x this
    seed: int = 0                   # controls EVERYTHING random -> reproducible

    # --- trial timing (assumed; edit freely) --------------------------
    trial_duration_s: float = 1.0   # length of one trial
    bin_ms: float = 10.0            # time-bin width -> 100 bins per trial
    hold_end_s: float = 0.3         # until here: hold period, baseline rate only
    ramp_end_s: float = 0.5         # hold_end..ramp_end: tuning ramps up 0 -> 1
                                    # after ramp_end: full direction-tuned rate

    # --- calibration from real data -----------------------------------
    min_unit_rate_hz: float = 0.5   # real units below this mean rate are not
                                    # used as templates (near-silent)
    silent_fraction: float = 16 / 96  # share of synthetic channels with no signal
                                    # (real session: 16 of 96 electrodes were silent)
    pref_dir_mode: str = "uniform"  # "uniform": random preferred direction (as in
                                    #   the thesis proposal)
                                    # "empirical": reuse the real unit's own
    param_jitter_sigma: float = 0.10  # +/-10% lognormal jitter on baseline and
                                    # amplitude so copies of one real unit differ

    # --- noise (ASSUMED, not measured from the real data) -------------
    trial_gain_sigma: float = 0.15  # trial-to-trial gain wobble per neuron, on top
                                    # of spiking randomness. 0 = pure Poisson.

    # --- splits -------------------------------------------------------
    val_fraction: float = 0.15
    test_fraction: float = 0.15

    @property
    def n_bins(self) -> int:
        return int(round(self.trial_duration_s * 1000 / self.bin_ms))

    @property
    def dt_s(self) -> float:
        return self.bin_ms / 1000.0


# --------------------------------------------------------------------------
# Step 1: fit cosine tuning to the real units
# --------------------------------------------------------------------------
def fit_cosine_tuning(csv_path: str | Path = DEFAULT_CALIBRATION_CSV) -> pd.DataFrame:
    """Fit rate(k) = b + a*cos(th_k) + c*sin(th_k) to each real unit.

    With 8 equally spaced directions this has a simple exact least-squares form:
        b = mean of the 8 rates
        a = (2/8) * sum(rate_k * cos th_k)
        c = (2/8) * sum(rate_k * sin th_k)
    and then  amplitude = sqrt(a^2 + c^2),  preferred = atan2(c, a).

    Returns one row per real unit with: baseline_hz, amplitude_hz,
    pref_dir_rad, r2 (how well a cosine explains that unit).
    """
    df = pd.read_csv(csv_path)
    rate_cols = [c for c in df.columns if c.startswith("rate_dir_")]
    if len(rate_cols) != N_DIRECTIONS:
        raise ValueError(f"Expected 8 'rate_dir_*' columns, found {len(rate_cols)}")
    rates = df[rate_cols].to_numpy(dtype=float)

    cos_t, sin_t = np.cos(DIRECTION_ANGLES), np.sin(DIRECTION_ANGLES)
    baseline = rates.mean(axis=1)
    a = (2 / N_DIRECTIONS) * (rates * cos_t).sum(axis=1)
    c = (2 / N_DIRECTIONS) * (rates * sin_t).sum(axis=1)
    amplitude = np.hypot(a, c)
    pref = np.mod(np.arctan2(c, a), 2 * np.pi)

    fitted = baseline[:, None] + a[:, None] * cos_t + c[:, None] * sin_t
    ss_res = ((rates - fitted) ** 2).sum(axis=1)
    ss_tot = ((rates - baseline[:, None]) ** 2).sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        r2 = np.where(ss_tot > 0, 1 - ss_res / ss_tot, np.nan)

    return pd.DataFrame(
        {
            "channel": df["channel"],
            "unit": df["unit"],
            "baseline_hz": baseline,
            "amplitude_hz": amplitude,
            "pref_dir_rad": pref,
            "r2": r2,
        }
    )


# --------------------------------------------------------------------------
# Step 2: build the synthetic neuron population
# --------------------------------------------------------------------------
def build_population(calibration: pd.DataFrame, cfg: GeneratorConfig) -> dict:
    """Draw MAX_CHANNELS synthetic neurons, then keep the first cfg.n_channels.

    Each neuron copies the (baseline, amplitude) pair of a randomly chosen real
    unit, so the synthetic population inherits the real spread: a few very fast
    neurons, many slow ones, weak and strong tuners. A random subset of
    channels is silent (rate 0) to mimic dead electrodes.
    """
    if not 1 <= cfg.n_channels <= MAX_CHANNELS:
        raise ValueError(f"n_channels must be between 1 and {MAX_CHANNELS}")

    # separate stream so the population does not depend on how many trials we make
    rng = np.random.default_rng([cfg.seed, 1])

    pool = calibration[calibration["baseline_hz"] >= cfg.min_unit_rate_hz].reset_index(drop=True)
    if len(pool) == 0:
        raise ValueError("No real units pass min_unit_rate_hz; lower it.")

    idx = rng.integers(0, len(pool), size=MAX_CHANNELS)
    jit_b = rng.lognormal(0.0, cfg.param_jitter_sigma, size=MAX_CHANNELS)
    jit_a = rng.lognormal(0.0, cfg.param_jitter_sigma, size=MAX_CHANNELS)
    baseline = pool["baseline_hz"].to_numpy()[idx] * jit_b
    amplitude = pool["amplitude_hz"].to_numpy()[idx] * jit_a

    if cfg.pref_dir_mode == "uniform":
        pref = rng.uniform(0, 2 * np.pi, size=MAX_CHANNELS)
    elif cfg.pref_dir_mode == "empirical":
        pref = np.mod(pool["pref_dir_rad"].to_numpy()[idx]
                      + rng.normal(0, 0.2, size=MAX_CHANNELS), 2 * np.pi)
    else:
        raise ValueError("pref_dir_mode must be 'uniform' or 'empirical'")

    silent = rng.random(MAX_CHANNELS) < cfg.silent_fraction
    baseline = np.where(silent, 0.0, baseline)
    amplitude = np.where(silent, 0.0, amplitude)

    n = cfg.n_channels
    return {
        "baseline_hz": baseline[:n],
        "amplitude_hz": amplitude[:n],
        "pref_dir_rad": pref[:n],
        "is_silent": silent[:n],
        "template_unit": idx[:n],  # which real unit (row of `pool`) each neuron copied
    }


# --------------------------------------------------------------------------
# Step 3: simulate trials
# --------------------------------------------------------------------------
def time_envelope(cfg: GeneratorConfig) -> np.ndarray:
    """0 during hold, linear ramp to 1 during 'go', 1 during movement."""
    t = (np.arange(cfg.n_bins) + 0.5) * cfg.dt_s
    env = np.clip((t - cfg.hold_end_s) / (cfg.ramp_end_s - cfg.hold_end_s), 0.0, 1.0)
    return env


def expected_rate_hz(pop: dict, direction: int) -> np.ndarray:
    """Noise-free movement-phase rate of every neuron for one direction (Hz)."""
    r = pop["baseline_hz"] + pop["amplitude_hz"] * np.cos(
        DIRECTION_ANGLES[direction] - pop["pref_dir_rad"]
    )
    return np.clip(r, 0.0, None)


def simulate_trials(pop: dict, labels: np.ndarray, cfg: GeneratorConfig,
                    chunk: int = 50) -> np.ndarray:
    """Return binary spikes, shape (n_trials, n_channels, n_bins), dtype uint8.

    rate(c, trial, t) = max(0, baseline_c + envelope(t)*amplitude_c*cos(dir - pref_c))
                        * trial_gain(c, trial)
    each bin holds 0 or 1 spike, with probability min(1, rate * dt). This keeps the
    average firing rate equal to the target rate (a 76 Hz neuron really averages
    76 Hz), at the cost of being slightly less variable than a true Poisson
    process for the very fastest neurons.
    """
    rng = np.random.default_rng([cfg.seed, 2])
    n_trials, n_ch, n_bins = len(labels), cfg.n_channels, cfg.n_bins
    env = time_envelope(cfg).astype(np.float32)
    spikes = np.zeros((n_trials, n_ch, n_bins), dtype=np.uint8)

    base = pop["baseline_hz"].astype(np.float32)
    amp = pop["amplitude_hz"].astype(np.float32)
    pref = pop["pref_dir_rad"]
    sigma = cfg.trial_gain_sigma

    for start in range(0, n_trials, chunk):
        stop = min(start + chunk, n_trials)
        ang = DIRECTION_ANGLES[labels[start:stop]][:, None]            # (m, 1)
        tune = np.cos(ang - pref[None, :]).astype(np.float32)           # (m, C)
        if sigma > 0:
            # mean-one lognormal gain, one value per neuron per trial
            gain = rng.lognormal(-0.5 * sigma**2, sigma, size=tune.shape).astype(np.float32)
        else:
            gain = np.ones_like(tune)
        rate = base[None, :, None] + env[None, None, :] * (amp[None, :, None] * tune[:, :, None])
        rate = np.clip(rate, 0.0, None) * gain[:, :, None]              # (m, C, T) in Hz
        p = np.minimum(rate * cfg.dt_s, 1.0)
        spikes[start:stop] = (rng.random(p.shape, dtype=np.float32) < p).astype(np.uint8)
    return spikes


def stratified_split(labels: np.ndarray, cfg: GeneratorConfig):
    """Train/val/test index arrays with the same direction balance in each."""
    rng = np.random.default_rng([cfg.seed, 3])
    train, val, test = [], [], []
    for k in range(N_DIRECTIONS):
        idx = rng.permutation(np.flatnonzero(labels == k))
        n_test = int(round(cfg.test_fraction * len(idx)))
        n_val = int(round(cfg.val_fraction * len(idx)))
        test.append(idx[:n_test])
        val.append(idx[n_test:n_test + n_val])
        train.append(idx[n_test + n_val:])
    return tuple(np.sort(np.concatenate(g)) for g in (train, val, test))


# --------------------------------------------------------------------------
# Public entry points
# --------------------------------------------------------------------------
def generate_dataset(cfg: GeneratorConfig,
                     calibration: pd.DataFrame | None = None,
                     calibration_csv: str | Path = DEFAULT_CALIBRATION_CSV) -> dict:
    """Build one complete synthetic dataset (everything stays in memory)."""
    if calibration is None:
        calibration = fit_cosine_tuning(calibration_csv)
    pop = build_population(calibration, cfg)

    rng = np.random.default_rng([cfg.seed, 4])
    labels = rng.permutation(np.repeat(np.arange(N_DIRECTIONS), cfg.trials_per_direction))
    spikes = simulate_trials(pop, labels, cfg)
    tr, va, te = stratified_split(labels, cfg)
    return {
        "spikes": spikes,          # (trials, channels, time bins), 0/1
        "labels": labels,          # 0..7 -> DIRECTION_NAMES
        "train_idx": tr, "val_idx": va, "test_idx": te,
        **pop,
        "config": asdict(cfg),
    }


def save_dataset(data: dict, out_dir: str | Path = DEFAULT_OUT_DIR,
                 calibration_csv: str | Path = DEFAULT_CALIBRATION_CSV) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = data["config"]
    stem = f"synthetic_ch{cfg['n_channels']}_seed{cfg['seed']}"
    arrays = {k: v for k, v in data.items() if isinstance(v, np.ndarray)}
    npz_path = out_dir / f"{stem}.npz"
    np.savez_compressed(npz_path, **arrays)

    meta = {
        "file": npz_path.name,
        "shape_spikes": list(data["spikes"].shape),
        "axes": ["trial", "channel", "time_bin"],
        "direction_names": DIRECTION_NAMES,
        "calibration_source": str(Path(calibration_csv).name),
        "config": cfg,
        "notes": "Rates calibrated from fitted cosine tuning of real primate units; "
                 "timing structure and trial-gain noise are assumptions.",
    }
    with open(out_dir / f"{stem}.json", "w") as f:
        json.dump(meta, f, indent=2)
    return npz_path


def load_dataset(npz_path: str | Path) -> dict:
    with np.load(npz_path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


# --------------------------------------------------------------------------
# Command line
# --------------------------------------------------------------------------
def main() -> None:
    p = argparse.ArgumentParser(description="Generate synthetic centre-out spike data.")
    p.add_argument("--channels", type=int, nargs="+", default=[96],
                   help="one or more channel counts, e.g. 16 32 64 128")
    p.add_argument("--trials-per-direction", type=int, default=100)
    p.add_argument("--seed", type=int, nargs="+", default=[0],
                   help="one or more random seeds")
    p.add_argument("--calibration-csv", default=str(DEFAULT_CALIBRATION_CSV))
    p.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    p.add_argument("--pref-dir-mode", choices=["uniform", "empirical"], default="uniform")
    p.add_argument("--trial-gain-sigma", type=float, default=0.15)
    args = p.parse_args()

    calibration = fit_cosine_tuning(args.calibration_csv)
    for seed in args.seed:
        for n in args.channels:
            cfg = GeneratorConfig(n_channels=n, trials_per_direction=args.trials_per_direction,
                                  seed=seed, pref_dir_mode=args.pref_dir_mode,
                                  trial_gain_sigma=args.trial_gain_sigma)
            data = generate_dataset(cfg, calibration=calibration)
            path = save_dataset(data, args.out_dir, args.calibration_csv)
            print(f"saved {path}  spikes{data['spikes'].shape}  "
                  f"mean rate {data['spikes'].mean() / cfg.dt_s:.2f} Hz")


if __name__ == "__main__":
    main()
