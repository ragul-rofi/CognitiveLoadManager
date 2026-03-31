"""CLM Scorer for computing cognitive load scores and zone classification."""

from __future__ import annotations
from clm.core.models import Signals


class CLMScorer:
    """
    Computes cognitive load score from signals and classifies zones.
    """
    
    def __init__(self, weights: list[float], green_max: float, amber_max: float):
        """
        Initialize with scoring weights and zone thresholds.
        
        Args:
            weights: [branching, repetition, uncertainty, goal_distance] weights
            green_max: Upper bound for Green zone (default 40)
            amber_max: Upper bound for Amber zone (default 70)
        """
        self.weights = weights
        self.green_max = green_max
        self.amber_max = amber_max
    
    def compute_score(self, signals: Signals) -> float:
        """
        Compute CLM score using weighted formula.
        
        Formula: 100 × (w1×branching + w2×repetition + w3×uncertainty + w4×goal_distance)
        
        Args:
            signals: Normalized cognitive load signals (0-1 range)
            
        Returns:
            CLM score normalized to 0-100
        """
        raw_score = (
            self.weights[0] * signals.branching_factor +
            self.weights[1] * signals.repetition_rate +
            self.weights[2] * signals.uncertainty_density +
            self.weights[3] * signals.goal_distance
        )
        
        # raw_score is in [0, 1], normalize to [0, 100]
        clm_score = raw_score * 100.0
        
        return clm_score
    
    def classify_zone(self, clm_score: float) -> str:
        """
        Classify cognitive load zone.
        
        Args:
            clm_score: CLM score in range 0-100
            
        Returns:
            "Green" (0-40), "Amber" (40-70), or "Red" (70-100)
        """
        if clm_score <= self.green_max:
            return "Green"
        elif clm_score <= self.amber_max:
            return "Amber"
        else:
            return "Red"
