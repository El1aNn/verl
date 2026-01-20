
from verl.utils.reward_score.math_dapo import compute_score, is_correct_strict_box

solution = "The answer is \\boxed{5}."
ground_truth = "5"
print(f"Correct case: {compute_score(solution, ground_truth)}")

solution_wrong = "The answer is \\boxed{6}."
print(f"Wrong case: {compute_score(solution_wrong, ground_truth)}")

# Check strict box
print(f"Strict box correct: {is_correct_strict_box(solution, ground_truth)}")
print(f"Strict box wrong: {is_correct_strict_box(solution_wrong, ground_truth)}")
