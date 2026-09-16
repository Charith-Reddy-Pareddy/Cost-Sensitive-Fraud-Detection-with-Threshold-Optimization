from src.models.run_synthetic_cost_sensitivity_study import _run_one


def test_run_one_returns_well_formed_result():
    result = _run_one(n_samples=3000, imbalance=0.05, class_sep=1.5, cost_ratio=10, seed=0)

    assert result is not None
    for key in ("cost_a", "cost_b", "cost_c", "cost_d", "brier_standard", "brier_cost_weighted"):
        assert result[key] >= 0
    assert result["best_config"] in {"A", "B", "C", "D"}
    assert isinstance(result["threshold_tuning_helps"], bool)
    assert isinstance(result["cost_weighting_helps"], bool)
    assert isinstance(result["combining_beats_training_alone"], bool)


def test_run_one_skips_when_too_few_positives():
    # severe imbalance + tiny sample -> expected positives well under the minimum threshold
    result = _run_one(n_samples=200, imbalance=0.001, class_sep=1.0, cost_ratio=10, seed=0)
    assert result is None
