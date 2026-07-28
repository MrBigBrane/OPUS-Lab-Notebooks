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
