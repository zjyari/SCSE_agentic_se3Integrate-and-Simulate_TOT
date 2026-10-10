"""
test_generated_navigation_logic.py
==================================
Level 2 - Behavioral testing for the generated navigation logic.

Calls decide_next_move(state) from artifacts/navigation_logic.py with known
robot states and checks that the returned action is a valid action AND the
correct expected move.

The state dict is the input the robot receives from its sensor:
    {
        "goal_ahead": bool,      # goal is directly ahead
        "goal_on_left": bool,    # goal is to the left
        "goal_on_right": bool,   # goal is to the right
        "front_blocked": bool,   # path in front is blocked
        "left_blocked": bool,    # path to the left is blocked
        "right_blocked": bool    # path to the right is blocked
    }

Expected behaviour (from the requirements / plan):
    * prefer moving toward the goal whenever it can be done safely
    * never move into a blocked direction
    * if no safe direction is available, STOP
    Priority: FORWARD > LEFT > RIGHT > STOP (toward goal first).

A sample test case (from the project instructions):
    {"goal_ahead": True, "goal_on_left": False, "goal_on_right": False,
     "front_blocked": False, "left_blocked": False, "right_blocked": False}
    -> "FORWARD"

All 2^6 = 64 possible states are enumerated, including tricky states where the
logic may not work as expected (goal direction blocked, multiple goal signals,
everything blocked, no goal signal at all).
"""

import importlib.util
import itertools
import os
import sys
import unittest

VALID_ACTIONS = {"FORWARD", "LEFT", "RIGHT", "STOP"}

HERE = os.path.dirname(os.path.abspath(__file__))
NAV_MODULE_PATH = os.path.join(HERE, "artifacts", "navigation_logic.py")

# Load the navigation logic module from the artifact (not the test file).
_spec = importlib.util.spec_from_file_location("navigation_logic", NAV_MODULE_PATH)
nav = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(nav)

STATE_KEYS = [
    "goal_ahead",
    "goal_on_left",
    "goal_on_right",
    "front_blocked",
    "left_blocked",
    "right_blocked",
]


def expected_action(state):
    """The correct action for a state, according to the navigation rules.

    Decision order (matches the revised navigation logic):
      1) goal ahead and front clear          -> FORWARD
      2) goal on left and left clear         -> LEFT
      3) goal on right and right clear       -> RIGHT
      4) otherwise pick any clear direction to manoeuvre around the obstacle
         (prefer FORWARD, then LEFT, then RIGHT)
      5) only if ALL directions are blocked  -> STOP
    """
    if state["goal_ahead"] and not state["front_blocked"]:
        return "FORWARD"
    if state["goal_on_left"] and not state["left_blocked"]:
        return "LEFT"
    if state["goal_on_right"] and not state["right_blocked"]:
        return "RIGHT"
    if not state["front_blocked"]:
        return "FORWARD"
    if not state["left_blocked"]:
        return "LEFT"
    if not state["right_blocked"]:
        return "RIGHT"
    return "STOP"


# --- Sample test case shown in the project instructions ----------------------
SAMPLE_CASE = (
    {
        "goal_ahead": True,
        "goal_on_left": False,
        "goal_on_right": False,
        "front_blocked": False,
        "left_blocked": False,
        "right_blocked": False,
    },
    "FORWARD",
)


class TestGeneratedNavigationLogic(unittest.TestCase):
    def test_sample_case_from_instructions(self):
        state, expected = SAMPLE_CASE
        self.assertEqual(nav.decide_next_move(state), expected)

    def test_all_64_states(self):
        failures = []
        for bits in itertools.product([False, True], repeat=len(STATE_KEYS)):
            state = dict(zip(STATE_KEYS, bits))
            expected = expected_action(state)

            result = nav.decide_next_move(state)

            # The result must be a valid action...
            if result not in VALID_ACTIONS:
                failures.append((state, f"invalid action {result!r}", expected))
                continue
            # ...and the correct expected move.
            if result != expected:
                failures.append((state, result, expected))

        self.assertEqual(
            failures,
            [],
            f"{len(failures)} of 64 states behaved incorrectly: {failures[:5]} ...",
        )


def _print_summary() -> None:
    """Print a summary of all 64 states and the expected moves."""
    print("=== Test: generated navigation logic (Level 2 behavioral test) ===\n")
    print("Loaded module:", NAV_MODULE_PATH)
    print(f"decide_next_move exists: {callable(getattr(nav, 'decide_next_move', None))}\n")
    print(f"Sample case from instructions: {SAMPLE_CASE[1]}\n")

    total = 0
    ok = 0
    bad = []
    for bits in itertools.product([False, True], repeat=len(STATE_KEYS)):
        state = dict(zip(STATE_KEYS, bits))
        total += 1
        try:
            result = nav.decide_next_move(state)
        except Exception as exc:  # noqa: BLE001 - report any crash
            bad.append((state, f"CRASH: {exc!r}"))
            continue
        expected = expected_action(state)
        if result in VALID_ACTIONS and result == expected:
            ok += 1
        else:
            bad.append((state, result, expected))

    print(f"States tested: {total}")
    print(f"Passed:        {ok}")
    print(f"Failed:        {len(bad)}")
    for entry in bad[:10]:
        print("  FAIL:", entry)
    print()


if __name__ == "__main__":
    _print_summary()
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(TestGeneratedNavigationLogic)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
