"""
test_planner.py
===============
Level 1 - Agent / Pipeline smoke test for the Planner Agent.

Takes the requirements artifact (artifacts/requirements.json), sends it to the
Planner Agent, lets the agent run as usual (its own validation accepts or
rejects the output), stores the result in artifacts/plan.json and displays it
after the test ends.

Ensures the agent turns the requirements artifact into a plan that the
Developer Agent can use.
"""

import json
import os

from planner_agent import run_planner

HERE = os.path.dirname(os.path.abspath(__file__))
REQUIREMENTS_PATH = os.path.join(HERE, "artifacts", "requirements.json")
ARTIFACTS_DIR = os.path.join(HERE, "artifacts")
OUTPUT_PATH = os.path.join(ARTIFACTS_DIR, "plan.json")


def main() -> None:
    print("=== Test: Planner Agent (Level 1 smoke test) ===\n")

    if not os.path.exists(REQUIREMENTS_PATH):
        raise FileNotFoundError(
            f"Requirements artifact not found: {REQUIREMENTS_PATH}\n"
            "Run `python test_analyst.py` (or `python run_analyst.py`) first."
        )

    with open(REQUIREMENTS_PATH, "r", encoding="utf-8") as f:
        requirements = json.load(f)

    # Run the actual agent with its real input (context isolation: only the
    # requirements artifact, not the whole Analyst conversation).
    plan = run_planner(requirements)

    # Store the output in the artifacts folder under plan.json.
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(plan, f, indent=2, ensure_ascii=False)

    print(f"Planner Agent output stored in: {OUTPUT_PATH}\n")
    print("Validated plan artifact:")
    print(json.dumps(plan, indent=2, ensure_ascii=False))

    # Level 1 pass condition: the plan is usable by the Developer Agent.
    assert set(plan.keys()) == {"strategy", "decisions", "stop_condition"}, (
        "plan artifact keys mismatch"
    )
    assert isinstance(plan["strategy"], str) and plan["strategy"].strip()
    assert isinstance(plan["decisions"], list) and len(plan["decisions"]) > 0
    assert all(isinstance(d, str) and d.strip() for d in plan["decisions"])
    assert isinstance(plan["stop_condition"], str) and plan["stop_condition"].strip()
    print("\nPLANNER TEST PASSED: plan artifact is valid and usable.")


if __name__ == "__main__":
    main()
