from santapp_ruler.ruler.grader import (
    grade_task,
    postprocess_prediction,
    string_match_all,
    string_match_part,
)


def test_string_match_all_matches_ruler_recall_behavior():
    predictions = ["alpha and beta", "alpha only"]
    references = [["alpha", "beta"], ["alpha", "beta"]]
    assert string_match_all(predictions, references) == 75.0


def test_string_match_part_accepts_any_reference():
    predictions = ["The answer is Santa Barbara", "none"]
    references = [["Santa Barbara", "UCSB"], ["x", "y"]]
    assert string_match_part(predictions, references) == 50.0


def test_task_family_metric_assignment():
    assert grade_task("qa_1", ["contains second"], [["first", "second"]]).score == 100.0
    assert grade_task("niah_single_1", ["contains first"], [["first", "second"]]).score == 50.0


def test_prediction_postprocessing_matches_control_character_cleanup():
    assert postprocess_prediction("  a\tb\x00c  ") == "a\nb\nc"


def test_bootstrap_task_interval_is_deterministic_and_prompt_level():
    grade = grade_task(
        "qa_1",
        ["answer", "wrong", "answer", "wrong"],
        [["answer"], ["answer"], ["answer"], ["answer"]],
        bootstrap_resamples=2000,
        confidence_level=0.95,
        bootstrap_seed=17,
    )
    repeat = grade_task(
        "qa_1",
        ["answer", "wrong", "answer", "wrong"],
        [["answer"], ["answer"], ["answer"], ["answer"]],
        bootstrap_resamples=2000,
        confidence_level=0.95,
        bootstrap_seed=17,
    )
    assert grade.example_scores == (100.0, 0.0, 100.0, 0.0)
    assert grade.confidence_interval == repeat.confidence_interval
    assert grade.confidence_interval is not None
    assert grade.confidence_interval.lower <= 50.0 <= grade.confidence_interval.upper


def test_stratified_bootstrap_keeps_tasks_equally_weighted():
    from santapp_ruler.ruler.grader import stratified_bootstrap_mean_confidence_interval

    interval = stratified_bootstrap_mean_confidence_interval(
        {"large": [100.0] * 20, "small": [0.0]},
        resamples=100,
        seed=3,
    )
    assert interval.point_estimate == 50.0
    assert interval.lower == interval.upper == 50.0
