import pytest
from generated.navigation_logic import decide_next_move
from itertools import product

def test_all_states():
    for goal_ahead, goal_on_left, goal_on_right, front_blocked, left_blocked, right_blocked in product([True, False], repeat=6):
        expected_action = "STOP"
        if goal_ahead and not front_blocked:
            expected_action = "FORWARD"
        elif goal_on_left and not left_blocked:
            expected_action = "LEFT"
        elif goal_on_right and not right_blocked:
            expected_action = "RIGHT"
        elif not front_blocked:
            expected_action = "FORWARD"
        elif not left_blocked:
            expected_action = "LEFT"
        elif not right_blocked:
            expected_action = "RIGHT"
        else:
            expected_action = "STOP"
        
        state = {
            "goal_ahead": goal_ahead,
            "goal_on_left": goal_on_left,
            "goal_on_right": goal_on_right,
            "front_blocked": front_blocked,
            "left_blocked": left_blocked,
            "right_blocked": right_blocked
        }
        
        assert decide_next_move(state) == expected_action, f"Failed for state {state}"
