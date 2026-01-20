from .math_reward import last_boxed_only_string

def compute_score(solution_str, ground_truth) -> float:
    retval = 0.0
    try:
        string_in_last_boxed = last_boxed_only_string(solution_str)
        if string_in_last_boxed is not None:
             retval = 1.0
    except Exception as e:
        print(e)
    return retval
