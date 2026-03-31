"""Unit tests for CLMScorer."""

import pytest
from clm.core.scorer import CLMScorer
from clm.core.models import Signals


class TestCLMScorer:
    """Test suite for CLMScorer class."""
    
    def test_init_with_default_values(self):
        """Test initialization with default weights and thresholds."""
        weights = [0.30, 0.25, 0.25, 0.20]
        green_max = 40.0
        amber_max = 70.0
        
        scorer = CLMScorer(weights, green_max, amber_max)
        
        assert scorer.weights == weights
        assert scorer.green_max == green_max
        assert scorer.amber_max == amber_max
    
    def test_compute_score_all_zeros(self):
        """Test score computation with all zero signals."""
        scorer = CLMScorer([0.30, 0.25, 0.25, 0.20], 40.0, 70.0)
        signals = Signals(
            branching_factor=0.0,
            repetition_rate=0.0,
            uncertainty_density=0.0,
            goal_distance=0.0
        )
        
        score = scorer.compute_score(signals)
        
        assert score == 0.0
    
    def test_compute_score_all_ones(self):
        """Test score computation with all maximum signals."""
        scorer = CLMScorer([0.30, 0.25, 0.25, 0.20], 40.0, 70.0)
        signals = Signals(
            branching_factor=1.0,
            repetition_rate=1.0,
            uncertainty_density=1.0,
            goal_distance=1.0
        )
        
        score = scorer.compute_score(signals)
        
        # Sum of weights is 1.0, so score should be 100.0
        assert score == 100.0
    
    def test_compute_score_weighted_formula(self):
        """Test score computation with specific signal values."""
        scorer = CLMScorer([0.30, 0.25, 0.25, 0.20], 40.0, 70.0)
        signals = Signals(
            branching_factor=0.5,
            repetition_rate=0.6,
            uncertainty_density=0.4,
            goal_distance=0.3
        )
        
        score = scorer.compute_score(signals)
        
        # Expected: 100 × (0.30×0.5 + 0.25×0.6 + 0.25×0.4 + 0.20×0.3)
        # = 100 × (0.15 + 0.15 + 0.10 + 0.06)
        # = 100 × 0.46 = 46.0
        assert score == pytest.approx(46.0)
    
    def test_compute_score_custom_weights(self):
        """Test score computation with custom weights."""
        scorer = CLMScorer([0.40, 0.30, 0.20, 0.10], 40.0, 70.0)
        signals = Signals(
            branching_factor=0.8,
            repetition_rate=0.2,
            uncertainty_density=0.5,
            goal_distance=0.1
        )
        
        score = scorer.compute_score(signals)
        
        # Expected: 100 × (0.40×0.8 + 0.30×0.2 + 0.20×0.5 + 0.10×0.1)
        # = 100 × (0.32 + 0.06 + 0.10 + 0.01)
        # = 100 × 0.49 = 49.0
        assert score == pytest.approx(49.0)
    
    def test_classify_zone_green(self):
        """Test zone classification for Green zone."""
        scorer = CLMScorer([0.30, 0.25, 0.25, 0.20], 40.0, 70.0)
        
        assert scorer.classify_zone(0.0) == "Green"
        assert scorer.classify_zone(20.0) == "Green"
        assert scorer.classify_zone(40.0) == "Green"
    
    def test_classify_zone_amber(self):
        """Test zone classification for Amber zone."""
        scorer = CLMScorer([0.30, 0.25, 0.25, 0.20], 40.0, 70.0)
        
        assert scorer.classify_zone(40.1) == "Amber"
        assert scorer.classify_zone(55.0) == "Amber"
        assert scorer.classify_zone(70.0) == "Amber"
    
    def test_classify_zone_red(self):
        """Test zone classification for Red zone."""
        scorer = CLMScorer([0.30, 0.25, 0.25, 0.20], 40.0, 70.0)
        
        assert scorer.classify_zone(70.1) == "Red"
        assert scorer.classify_zone(85.0) == "Red"
        assert scorer.classify_zone(100.0) == "Red"
    
    def test_classify_zone_custom_thresholds(self):
        """Test zone classification with custom thresholds."""
        scorer = CLMScorer([0.30, 0.25, 0.25, 0.20], 30.0, 60.0)
        
        assert scorer.classify_zone(30.0) == "Green"
        assert scorer.classify_zone(30.1) == "Amber"
        assert scorer.classify_zone(60.0) == "Amber"
        assert scorer.classify_zone(60.1) == "Red"
    
    def test_score_and_zone_integration(self):
        """Test integration of score computation and zone classification."""
        scorer = CLMScorer([0.30, 0.25, 0.25, 0.20], 40.0, 70.0)
        
        # Low load signals -> Green zone
        low_signals = Signals(
            branching_factor=0.2,
            repetition_rate=0.1,
            uncertainty_density=0.15,
            goal_distance=0.1
        )
        low_score = scorer.compute_score(low_signals)
        assert scorer.classify_zone(low_score) == "Green"
        
        # Medium load signals -> Amber zone
        medium_signals = Signals(
            branching_factor=0.5,
            repetition_rate=0.5,
            uncertainty_density=0.5,
            goal_distance=0.5
        )
        medium_score = scorer.compute_score(medium_signals)
        assert scorer.classify_zone(medium_score) == "Amber"
        
        # High load signals -> Red zone
        high_signals = Signals(
            branching_factor=0.9,
            repetition_rate=0.8,
            uncertainty_density=0.85,
            goal_distance=0.75
        )
        high_score = scorer.compute_score(high_signals)
        assert scorer.classify_zone(high_score) == "Red"
