
from verl.utils.reward_score.math_dapo import compute_score

def test_reward():
    # Test correct answer
    solution_correct = "The answer is \\boxed{42}"
    ground_truth = "42"
    result_correct = compute_score(solution_correct, ground_truth)
    print(f"Correct answer result: {result_correct}")
    assert result_correct['score'] == 1.0, f"Expected 1.0, got {result_correct['score']}"

    # Test incorrect answer
    solution_incorrect = "The answer is \\boxed{0}"
    ground_truth = "42"
    result_incorrect = compute_score(solution_incorrect, ground_truth)
    print(f"Incorrect answer result: {result_incorrect}")
    assert result_incorrect['score'] == 0.0, f"Expected 0.0, got {result_incorrect['score']}"

    print("All tests passed!")

if __name__ == "__main__":
    test_reward()
