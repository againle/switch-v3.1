# adawait/controller.py
# Main export: AdaWaitController — computes adaptive wait times from
# CogniLLM response dicts and replaces the fixed timer interval.

from __future__ import annotations

from typing import Any

from .config import WaitTimeConfig
from .utils.parser import Parser
from .utils.sampler import WaitTimeSampler
from .utils.state import State

# Re-use the ResponseDict shape from the orchestrator
ResponseDict = dict[str, Any]


class AdaWaitController:
    """Adaptive wait-time controller driven by client emotional state.

    On every CogniLLM response the controller:
    1. extracts ``emotions`` (list of str) and ``openness`` (str),
    2. parses them into [0, 1] scores,
    3. feeds smoothed scores into a state tracker (E, O, V),
    4. samples the next timer interval from the wait-time distribution.

    Usage::

        adawait = AdaWaitController()
        ...
        next_wait = adawait.compute_wait(response_dict)
        orchestrator.time_threshold = next_wait
    """

    def __init__(self, config: WaitTimeConfig | None = None) -> None:
        self._config = config or WaitTimeConfig()
        self._state = State(self._config)
        self._parser = Parser()
        self._sampler = WaitTimeSampler(self._config)
        self._verbose = self._config.verbose

    # ── Public API ──────────────────────────────────────────────────

    def compute_wait(
        self,
        response: ResponseDict,
        V_raw: float | None = None,
    ) -> float:
        """Return the adaptive wait time (seconds) for the next interval.

        Args:
            response: A full CogniLLM response dict (must contain
                ``emotions`` list and ``openness`` string).
            V_raw: Optional typing-speed score in [0, 1] derived from
                the user's last message (chars / delta_seconds).
                ``None`` means no new user message (e.g. timer-fired).

        Returns:
            The sampled wait time in seconds, clamped to
            ``[tau_min, tau_max]`` from the config.
        """
        emotions: list[str] = response.get("emotions", [])
        openness: str = response.get("openness", "")

        # Parse raw signals → [0, 1] scores
        scores = self._parser.parse(emotions, openness)

        # Update smoothed state  (E, O, V)
        self._state.update(
            E_raw=scores.get("E"),
            O_raw=scores.get("O"),
            V_raw=V_raw,
        )

        # Sample the next wait duration
        wait = self._sampler.sample(self._state.get_state())

        if self._verbose:
            self._debug_print(wait)

        return wait

    # ── Debug ──────────────────────────────────────────────────────

    def _debug_print(self, wait: float) -> None:
        st = self._state.get_state()
        e = st.get("E", 0.5)
        o = st.get("O", 0.5)
        v = st.get("V", 0.5)
        bar_e = _mini_bar(e)
        bar_o = _mini_bar(o)
        bar_v = _mini_bar(v)
        lo = self._config.tau_min
        hi = self._config.tau_max
        print(
            f"\n  \033[36m[AdaWait]\033[0m"
            f"  E={e:.2f} {bar_e}"
            f"  O={o:.2f} {bar_o}"
            f"  V={v:.2f} {bar_v}"
            f"  →  \033[1m{wait:.1f}s\033[0m"
            f"  \033[2m[τ ∈ {lo:.0f}–{hi:.0f}s]\033[0m"
        )

    def reset(self) -> None:
        """Reset internal state to initial values (0.5 for all fields)."""
        self._state.reset()

    @property
    def state(self) -> dict[str, float | None]:
        """Current smoothed (E, O, V) state for inspection."""
        return self._state.get_state()

    # ── Distribution visualisation ─────────────────────────────────

    def plot_distribution(
        self,
        n: int = 10000,
        state: dict[str, float] | None = None,
    ) -> None:
        """Sample *n* wait times and print an ASCII histogram.

        Useful for verifying the sampler produces a spread rather
        than always clamping to the bounds.
        """
        if state is None:
            state = self._state.get_state()

        samples = [self._sampler.sample(state) for _ in range(n)]

        lo = self._config.tau_min
        hi = self._config.tau_max
        bins = 48   # fine-grained to reveal the true distribution shape
        bin_w = (hi - lo) / bins
        counts = [0] * bins
        clamped_lo = 0
        clamped_hi = 0

        for s in samples:
            if s <= lo:
                clamped_lo += 1
                idx = 0
            elif s >= hi:
                clamped_hi += 1
                idx = bins - 1
            else:
                idx = int((s - lo) / bin_w)
                if idx >= bins:
                    idx = bins - 1
            counts[idx] += 1

        max_count = max(counts) if counts else 1
        bar_w = 50

        e = state.get("E", 0.5)
        o = state.get("O", 0.5)
        v = state.get("V", 0.5)

        print(f"\n  \033[36m╭─ Distribution (n={n}) ───────────────────────────────────╮\033[0m")
        print(f"  \033[36m│\033[0m  E={e:.2f}  O={o:.2f}  V={v:.2f}  "
              f"τ ∈ [{lo:.0f}, {hi:.0f}] s")
        print(f"  \033[36m│\033[0m  clamp-lo: {clamped_lo:>5}   clamp-hi: {clamped_hi:>5}")
        print(f"  \033[36m├────────────────────────────────────────────────────────┤\033[0m")

        for i in range(bins):
            edge = lo + i * bin_w
            c = counts[i]
            filled = int(c / max_count * bar_w) if max_count > 0 else 0
            bar = "█" * filled + "░" * (bar_w - filled)
            pct = c / n * 100
            if pct >= 0.05:  # skip near-empty bins to reduce noise
                print(f"  \033[36m│\033[0m {edge:5.0f}s {bar} {pct:5.1f}%")

        print(f"  \033[36m╰────────────────────────────────────────────────────────╯\033[0m")
        print(f"  mean={sum(samples)/n:.1f}s  "
              f"median={sorted(samples)[n//2]:.1f}s  "
              f"min={min(samples):.1f}s  max={max(samples):.1f}s")
        print()


# ── Helpers ───────────────────────────────────────────────────────

def _mini_bar(val: float | None, width: int = 8) -> str:
    """Tiny ASCII bar for debug display."""
    if val is None:
        val = 0.5
    v = max(0.0, min(1.0, float(val)))
    filled = int(v * width)
    return f"\033[32m{'█' * filled}\033[2m{'░' * (width - filled)}\033[0m"
