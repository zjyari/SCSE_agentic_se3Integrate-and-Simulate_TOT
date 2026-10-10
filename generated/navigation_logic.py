def decide_next_move(state):
    """
    Decide the next move for the robot based on the current state.
    
    The state is a dictionary with the following boolean keys:
    - 'goal_ahead': True if the goal is directly ahead
    - 'goal_on_left': True if the goal is on the left
    - 'goal_on_right': True if the goal is on the right
    - 'front_blocked': True if the front path is blocked
    - 'left_blocked': True if the left path is blocked
    - 'right_blocked': True if the right path is blocked
    
    Returns one of: "FORWARD", "LEFT", "RIGHT", or "STOP"
    """
    if state["goal_ahead"] and not state["front_blocked"]:
        return "FORWARD"
    elif state["goal_on_left"] and not state["left_blocked"]:
        return "LEFT"
    elif state["goal_on_right"] and not state["right_blocked"]:
        return "RIGHT"
    else:
        if not state["front_blocked"]:
            return "FORWARD"
        elif not state["left_blocked"]:
            return "LEFT"
        elif not state["right_blocked"]:
            return "RIGHT"
        else:
            return "STOP"
