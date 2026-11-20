# adawait/config.py
# This file contains configuration settings for the adawait package.
from dataclasses import dataclass

@dataclass
class WaitTimeConfig:
    """Configuration for wait times in the adawait package."""

    lambda_base: float = 0.15  # Base wait time in seconds
    alpha_E: float = 1.20
    alpha_O: float = 1.00
    alpha_V: float = 0.40

    gamma_E: float = 0.80
    gamma_O: float = 0.60
    gamma_V: float = 0.30
    T_half: float = 4.0 

    smooth_E: float = 0.80
    smooth_O: float = 0.75
    smooth_V: float = 0.60

    tau_min: float = 5.0
    tau_max: float = 12.0

    # ── Initial client state ─────────────────────────────────────
    # E is the client's patience score (higher = waits longer); each turn's
    # emotion bucket maps onto it — see utils/models/emotions.py.
    E0: float = 0.20   # starts impatient (low = short first wait)
    O0: float = 0.15   # starts closed off
    V0: float = 0.30   # no typing baseline yet

    # ── Velocity (typing speed) ──────────────────────────────────
    V_max_cps: float = 8.0   # chars/sec that maps to V=1.0

    # ── Debug ────────────────────────────────────────────────────
    verbose: bool = False     # print E/O/V scores on each update


# ── Preset ────────────────────────────────────────────────────────


@dataclass
class AdaWaitPreset:
    """High-level preset: a single ``rate`` controls the overall
    wait-time profile.  Rate is unclamped — use 0.25 … 2.0.

    * rate = 2.0  →  tau ∈ [100, 240] s   (very patient)
    * rate = 1.0  →  tau ∈ [ 40, 120] s   (patient)
    * rate = 0.5  →  tau ∈ [ 10,  60] s   (moderate, default)
    * rate = 0.25 →  tau ∈ [  5,  30] s   (eager)
    """

    rate: float = 0.5

    def to_config(self, *, verbose: bool = False) -> WaitTimeConfig:
        """Convert the preset to a full ``WaitTimeConfig``."""
        r = max(0.1, self.rate)  # floor at 0.1 to avoid degenerate configs

        # tau_min = 60·r − 20   →  r=0.5→10, r=1.0→40, r=2.0→100
        # tau_max = 120·r        →  r=0.5→60, r=1.0→120, r=2.0→240
        tau_min = max(2.0, 60.0 * r - 20.0)
        tau_max = 120.0 * r

        # T_half and lambda_base both scale with rate so the
        # distribution fills [tau_min, tau_max] naturally.
        T_half = 10.0 * r        # r=0.5→5, r=1.0→10, r=2.0→20
        lambda_base = 0.0005 / r  # r=0.5→1e-3, r=1.0→5e-4, r=2.0→2.5e-4

        return WaitTimeConfig(
            tau_min=tau_min,
            tau_max=tau_max,
            lambda_base=lambda_base,
            T_half=T_half,
            verbose=verbose,
        )