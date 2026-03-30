"""Pytest configuration and Hypothesis profile setup."""

from hypothesis import settings, Verbosity

# Register Hypothesis profiles for different testing scenarios
settings.register_profile(
    "default",
    max_examples=100,
    deadline=1000
)

settings.register_profile(
    "ci",
    max_examples=500,
    deadline=5000
)

settings.register_profile(
    "debug",
    max_examples=10,
    verbosity=Verbosity.verbose
)

# Load default profile
settings.load_profile("default")
