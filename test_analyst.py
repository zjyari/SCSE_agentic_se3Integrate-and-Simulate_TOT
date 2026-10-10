"""
test_analyst.py
===============
Level 1 - Agent / Pipeline smoke test for the Analyst Agent.

Takes the original brief.txt, sends it to the Analyst Agent, lets the agent
run as usual (its own validation accepts or rejects the output), stores the
result in artifacts/requirements.json and displays it after the test ends.

Ensures the agent turns the brief text into a usable requirements artifact.
"""

import json
import os

from analyst_agent import run_analyst

HERE = os.path.dirname(os.path.abspath(__file__))
BRIEF_PATH = os.path.join(HERE, "brief.txt")
ARTIFACTS_DIR = os.path.join(HERE, "artifacts")
OUTPUT_PATH = os.path.join(ARTIFACTS_DIR, "requirements.json")


def main() -> None:
    print("=== Test: Analyst Agent (Level 1 smoke test) ===\n")

    with open(BRIEF_PATH, "r", encoding="utf-8") as f:
        brief_text = f.read()

    # Run the actual agent with its real input.
    requirements = run_analyst(brief_text)

    # Store the output in the artifacts folder under requirements.json.
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(requirements, f, indent=2, ensure_ascii=False)

    print(f"Analyst Agent output stored in: {OUTPUT_PATH}\n")
    print("Validated requirements artifact:")
    print(json.dumps(requirements, indent=2, ensure_ascii=False))

    # Level 1 pass condition: the artifact is usable (correct keys, types).
    required_keys = {"goal", "allowed_actions", "safe_stop", "avoid.obstacles"}
    assert set(requirements.keys()) == required_keys, "requirements artifact keys mismatch"
    assert isinstance(requirements["goal"], str)
    assert isinstance(requirements["allowed_actions"], list)
    assert set(requirements["allowed_actions"]) == {"FORWARD", "LEFT", "RIGHT", "STOP"}
    assert isinstance(requirements["safe_stop"], bool)
    assert isinstance(requirements["avoid.obstacles"], bool)
    print("\nANALYST TEST PASSED: requirements artifact is valid and usable.")


if __name__ == "__main__":
    main()
