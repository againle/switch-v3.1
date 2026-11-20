# adawait/utils/state.py
# State management for the adawait framework.

from typing import Dict, Optional
from ..config import WaitTimeConfig

class State:
    """
    A state management class for the adawait framework.
    Maintains three separate elements of state: `Emotions(Em)` , `Openness(O)` and `Velocities(V)`.
    """

    def __init__(self, config: WaitTimeConfig):
        self.config = config
        self.state: Dict[str, Optional[float]] = {
            "E": config.E0,
            "O": config.O0,
            "V": config.V0,
        }
        self.smooth_E = config.smooth_E
        self.smooth_O = config.smooth_O
        self.smooth_V = config.smooth_V

    def update(
        self,
        E_raw: Optional[float] = None,
        O_raw: Optional[float] = None,
        V_raw: Optional[float] = None
    ) -> Dict[str, float]:
        """
        Update the state with new raw values for Emotions, Openness, and Velocities.
        Applies smoothing to the updates.

        Args:
            E_raw (Optional[float]): New raw value for Emotions.
            O_raw (Optional[float]): New raw value for Openness.
            V_raw (Optional[float]): New raw value for Velocities.

        Returns:
            Dict[str, float]: The updated state dictionary.
        """
        if E_raw is not None:
            self.update_state("E", E_raw)
        if O_raw is not None:
            self.update_state("O", O_raw)
        if V_raw is not None:
            self.update_state("V", V_raw)

        return self.state
    
    def update_state(self, key: str, new_value: float) -> None:
        """
        Update the state for a given key with a new value, applying smoothing.

        Args:
            key (str): The state key ('E', 'O', or 'V').
            new_value (float): The new value to update the state with.
        """
        if key not in self.state:
            raise KeyError(f"Invalid state key: {key}. Valid keys are 'E', 'O', 'V'.")

        current_value = self.state[key]
        if current_value is None:
            self.state[key] = new_value
        else:
            if key == "E":
                smooth_factor = self.smooth_E
            elif key == "O":
                smooth_factor = self.smooth_O
            elif key == "V":
                smooth_factor = self.smooth_V
            else:
                raise KeyError(f"Invalid state key: {key}. Valid keys are 'E', 'O', 'V'.")

            # Apply smoothing
            smoothed_value = (smooth_factor * current_value) + ((1 - smooth_factor) * new_value)
            self.state[key] = smoothed_value

    def reset(self) -> None:
        """Reset the state to its initial values from config."""
        self.state = {
            "E": self.config.E0,
            "O": self.config.O0,
            "V": self.config.V0,
        }

    def get_state(self) -> Dict[str, Optional[float]]:
        """
        Get the current state.

        Returns:
            Dict[str, Optional[float]]: The current state dictionary.
        """
        return self.state