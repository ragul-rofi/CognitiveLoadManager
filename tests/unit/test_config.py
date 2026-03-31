"""Unit tests for CLMConfig."""

import pytest
from clm.core.config import CLMConfig
from clm.exceptions import ConfigurationError


class TestCLMConfig:
    """Test CLMConfig dataclass."""
    
    def test_default_config(self):
        """Test creating config with default values."""
        config = CLMConfig()
        
        assert config.branching_threshold == 7
        assert config.repetition_threshold == 0.85
        assert config.uncertainty_threshold == 0.15
        assert config.weights == [0.30, 0.25, 0.25, 0.20]
        assert config.green_max == 40.0
        assert config.amber_max == 70.0
        assert config.storage_type == "sqlite"
        assert config.storage_params == {}
        assert config.embedding_model == "all-MiniLM-L6-v2"
        assert len(config.hedged_tokens) == 10
        assert "maybe" in config.hedged_tokens
        assert "perhaps" in config.hedged_tokens
    
    def test_custom_config(self):
        """Test creating config with custom values."""
        config = CLMConfig(
            branching_threshold=10,
            repetition_threshold=0.90,
            uncertainty_threshold=0.20,
            weights=[0.25, 0.25, 0.25, 0.25],
            green_max=35.0,
            amber_max=75.0,
            storage_type="redis",
            storage_params={"host": "localhost", "port": 6379},
            embedding_model="custom-model",
            hedged_tokens=["maybe", "perhaps"]
        )
        
        assert config.branching_threshold == 10
        assert config.repetition_threshold == 0.90
        assert config.uncertainty_threshold == 0.20
        assert config.weights == [0.25, 0.25, 0.25, 0.25]
        assert config.green_max == 35.0
        assert config.amber_max == 75.0
        assert config.storage_type == "redis"
        assert config.storage_params == {"host": "localhost", "port": 6379}
        assert config.embedding_model == "custom-model"
        assert config.hedged_tokens == ["maybe", "perhaps"]
    
    def test_validate_valid_config(self):
        """Test validation passes for valid config."""
        config = CLMConfig()
        config.validate()  # Should not raise
    
    def test_validate_weights_sum_to_one(self):
        """Test validation passes when weights sum to 1.0."""
        config = CLMConfig(weights=[0.25, 0.25, 0.25, 0.25])
        config.validate()  # Should not raise
    
    def test_validate_weights_within_tolerance(self):
        """Test validation passes when weights sum to 1.0 within tolerance."""
        config = CLMConfig(weights=[0.30, 0.25, 0.25, 0.199])  # Sum = 0.999
        config.validate()  # Should not raise (within 0.01 tolerance)
    
    def test_validate_weights_not_sum_to_one(self):
        """Test validation fails when weights don't sum to 1.0."""
        config = CLMConfig(weights=[0.30, 0.30, 0.30, 0.30])  # Sum = 1.2
        
        with pytest.raises(ConfigurationError, match="Weights must sum to 1.0"):
            config.validate()
    
    def test_validate_weights_too_low(self):
        """Test validation fails when weights sum is too low."""
        config = CLMConfig(weights=[0.20, 0.20, 0.20, 0.20])  # Sum = 0.8
        
        with pytest.raises(ConfigurationError, match="Weights must sum to 1.0"):
            config.validate()
    
    def test_validate_zone_boundaries_valid(self):
        """Test validation passes for valid zone boundaries."""
        config = CLMConfig(green_max=40.0, amber_max=70.0)
        config.validate()  # Should not raise
    
    def test_validate_zone_boundaries_invalid_order(self):
        """Test validation fails when green_max >= amber_max."""
        config = CLMConfig(green_max=70.0, amber_max=40.0)
        
        with pytest.raises(ConfigurationError, match="Zone boundaries must satisfy"):
            config.validate()
    
    def test_validate_zone_boundaries_equal(self):
        """Test validation fails when green_max == amber_max."""
        config = CLMConfig(green_max=50.0, amber_max=50.0)
        
        with pytest.raises(ConfigurationError, match="Zone boundaries must satisfy"):
            config.validate()
    
    def test_validate_zone_boundaries_at_zero(self):
        """Test validation fails when green_max is at 0."""
        config = CLMConfig(green_max=0.0, amber_max=70.0)
        
        with pytest.raises(ConfigurationError, match="Zone boundaries must satisfy"):
            config.validate()
    
    def test_validate_zone_boundaries_at_hundred(self):
        """Test validation fails when amber_max is at 100."""
        config = CLMConfig(green_max=40.0, amber_max=100.0)
        
        with pytest.raises(ConfigurationError, match="Zone boundaries must satisfy"):
            config.validate()
    
    def test_validate_zone_boundaries_above_hundred(self):
        """Test validation fails when amber_max is above 100."""
        config = CLMConfig(green_max=40.0, amber_max=110.0)
        
        with pytest.raises(ConfigurationError, match="Zone boundaries must satisfy"):
            config.validate()
