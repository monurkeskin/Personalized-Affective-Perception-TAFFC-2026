"""
Time-Based Bidding Strategy Module

This module implements the time-based concession curve for the Solver Agent's
negotiation strategy, as described in Section 4 of the paper.

The strategy uses a quadratic Bezier curve to interpolate between control points
P0, P1, P2 based on the normalized negotiation time t ∈ [0, 1].
"""

from typing import Optional


class TimeBasedStrategy:
    """
    Implements a time-based target utility calculation using a quadratic Bezier curve.
    
    The target utility decreases over time according to the formula:
        TU(t) = (1-t)² * P0 + 2*(1-t)*t * P1 + t² * P2
    
    Where:
        - t: normalized time [0, 1]
        - P0: initial utility target (default: 0.9)
        - P1: midpoint control (default: 0.7)
        - P2: final utility target (default: 0.4)
    
    Attributes:
        P0 (float): Initial target utility at t=0
        P1 (float): Bezier control point for curve shape
        P2 (float): Final target utility at t=1
    """
    
    def __init__(self, P0: float = 0.9, P1: float = 0.7, P2: float = 0.4) -> None:
        """
        Initialize the time-based strategy with control points.
        
        Args:
            P0: Initial target utility (default: 0.9)
            P1: Midpoint control parameter (default: 0.7)
            P2: Final target utility (default: 0.4)
        """
        self.P0 = P0
        self.P1 = P1
        self.P2 = P2

    def update_P0(self, P0: float) -> None:
        """Update the initial target utility."""
        self.P0 = P0

    def update_P1(self, P1: float) -> None:
        """Update the midpoint control parameter."""
        self.P1 = P1

    def update_P2(self, P2: float) -> None:
        """Update the final target utility."""
        self.P2 = P2

    def get_target_utility(self, current_time: float) -> float:
        """
        Calculate target utility based on normalized negotiation time.
        
        Uses quadratic Bezier interpolation between control points.
        
        Args:
            current_time: Normalized time in range [0, 1]
                          where 0 = start, 1 = deadline
        
        Returns:
            Target utility value in approximately [P2, P0] range
        """
        return (
            ((1 - current_time) ** 2) * self.P0
            + 2 * (1 - current_time) * current_time * self.P1
            + (current_time ** 2) * self.P2
        )


# Backward compatibility alias (deprecated)
kek = TimeBasedStrategy
