"""
Behavior-Based Bidding Strategy Module

This module implements the behavior-based component of the Solver Agent's
negotiation strategy. It tracks the opponent's utility changes and adjusts
the agent's target utility based on reciprocity principles.

The strategy uses a weighted moving average of recent utility differences
to determine how much to concede.
"""

from collections import deque
from typing import Dict, List

import numpy as np


class BehaviorBasedStrategy:
    """
    Implements behavior-based target utility calculation using opponent modeling.
    
    This strategy tracks the opponent's recent offer patterns and adjusts
    concession behavior based on the weighted average of utility differences.
    
    The formula is:
        TU = prev_utility - (1 - awareness) * (0.5 + 0.5*t) * δ
    
    Where δ is the weighted moving average of opponent utility changes.
    
    Attributes:
        prevUtilsOp (deque): Rolling window of opponent utility differences
        W (dict): Weight vectors for different history lengths
        delta_multiplier (float): Scaling factor for delta (adjustable)
    """
    
    def __init__(self) -> None:
        """Initialize the behavior-based strategy with default parameters."""
        self.prevUtilsOp: deque = deque(maxlen=4)
        
        # Weights for history offers - latest difference weighted more heavily
        self.W: Dict[str, List[float]] = {
            "1": [1],
            "2": [0.25, 0.75],
            "3": [0.11, 0.22, 0.66],
            "4": [0.05, 0.15, 0.3, 0.5],
        }
        
        self.delta_multiplier: float = 1.0

    def get_target_utility(
        self, 
        utility_diff: float, 
        my_prev_offer_utility: float, 
        current_time: float, 
        awareness: float
    ) -> float:
        """
        Calculate target utility based on opponent behavior.
        
        Args:
            utility_diff: Difference in utility between opponent's last two offers
            my_prev_offer_utility: Agent's previous offer utility
            current_time: Normalized negotiation time [0, 1]
            awareness: Agent's confidence in opponent model [0, 1]
        
        Returns:
            Target utility for next offer
        """
        self.prevUtilsOp.append(utility_diff)
        self.delta = self._get_delta()

        return my_prev_offer_utility - (
            (1 - awareness) * (0.5 + 0.5 * current_time) * self.delta
        )

    def _get_delta(self) -> float:
        """
        Calculate weighted moving average of opponent utility changes.
        
        Returns:
            Weighted delta scaled by delta_multiplier
        """
        delta = np.sum([
            a * b
            for a, b in zip(self.prevUtilsOp, self.W[str(len(self.prevUtilsOp))])
        ])
        return delta * self.delta_multiplier

    def set_delta_multiplier(self, delta_multiplier: float) -> None:
        """
        Increase delta multiplier by 50% (used when opponent is Selfish).
        
        Args:
            delta_multiplier: New multiplier value (currently unused, 
                              method increases by 1.5x instead)
        """
        self.delta_multiplier *= 1.5


# Backward compatibility alias (deprecated)
bb = BehaviorBasedStrategy
