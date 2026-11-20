# adawait/utils/sampler.py
# This file contains utility functions for sampling wait times in the adawait package.
import math
import random
from ..config import WaitTimeConfig
from typing import Dict

class WaitTimeSampler:
    """Sampler for wait times on next round, based on the current state and configuration."""

    def __init__(self, config: WaitTimeConfig):
        self.config = config

        self.lambda_base = config.lambda_base
        self.alpha_E = config.alpha_E
        self.alpha_O = config.alpha_O
        self.alpha_V = config.alpha_V
        
        self.gamma_E = config.gamma_E
        self.gamma_O = config.gamma_O
        self.gamma_V = config.gamma_V
        self.T_half = config.T_half

        self.tau_min = config.tau_min
        self.tau_max = config.tau_max

    
    def sample(self, state: Dict[str, float]) -> float:
        """
        Sample a wait time based on the current state and configuration.

        Args:
            state (Dict[str, float]): The current state dictionary with keys 'E', 'O', 'V'.
        Returns:
            float: The sampled wait time.
        """
        _E = state.get("E", 0.5)
        _O = state.get("O", 0.5)
        _V = state.get("V", 0.5)

        # Initial lambda calculation 
        # 1-E: E is the client's patience score (see models/emotions.py):
        #      high E = patient, so (1-E) is small, lambda_0 is smaller and
        #      the wait is longer. Low E (anxious) shortens it.
        # 1-O: High openness means the same kind of patience.
        # V:   Faster typing (high V) raises lambda_0 and shortens the wait.
        lambda_0 = self.lambda_base * (self.alpha_E * (1-_E) + self.alpha_O * (1-_O) + self.alpha_V * _V)


        # Decay factor of patience over time
        gamma = self.gamma_E * (1 - _E) + self.gamma_O * (1 - _O) + self.gamma_V * _V

        # Rejection sampling: re-sample until within [tau_min, tau_max].
        # This avoids the artificial spike at the clamp boundary.
        for _ in range(100):  # safety limit
            if gamma < 1e-6:
                time = random.expovariate(lambda_0)
            else:
                Epsilon = random.expovariate(1.0)
                term = 1.0 + (gamma * Epsilon) / (lambda_0 * self.T_half)
                if term <= 1.0:
                    term = 1.0 + 1e-6
                time = (self.T_half / gamma) * math.log(term)

            if self.tau_min <= time <= self.tau_max:
                return time

        # Fallback after 100 rejections — return midpoint
        return (self.tau_min + self.tau_max) / 2.0