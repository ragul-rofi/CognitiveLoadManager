"""CLM Quickstart — minimum viable integration."""
from clm import CLM

clm = CLM(verbose=True)

# Simulate an agent loop
fake_outputs = [
    "I'll start by designing the database schema for the user auth system.",
    "Working on the database schema. Also need to handle API endpoints, caching layer, auth tokens, session management, password hashing, email verification, OAuth integration, and rate limiting.",
    "I'm not sure which approach to take. Maybe I should use JWT, or perhaps sessions, possibly both. I might need to reconsider the architecture. It's unclear which is better.",
    "Focusing back on authentication. The core goal is to build secure user login.",
    "Authentication module complete. Moving to the next task.",
]

print("=== CLM Quickstart ===\n")
for i, output in enumerate(fake_outputs, 1):
    print(f"[Step {i}] {output[:60]}...")
    result = clm.observe_raw(output)
    print()

print("\n=== Session Summary ===")
import json
print(json.dumps(clm.summary(), indent=2))
