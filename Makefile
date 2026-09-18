.PHONY: reproduce reproduce-primary reproduce-sparkov dashboard manifest test lint

# Regenerates the core, load-bearing artifacts this project's headline numbers come from: the
# chronological splits, both production models, the primary cost/calibration/bootstrap analyses,
# the evaluation plots, SHAP, the dashboard's exported JSON, and a fresh reproducibility manifest.
#
# What this does NOT do, on purpose: every ablation, robustness check, and comparison script in
# src/models/ (there are ~30). Re-running all of them takes far longer than reproducing the
# headline pipeline and most don't persist artifacts other code depends on — they print a table.
# Run any of them individually with `python -m src.models.<script_name>` the same way this target
# does. Requires data/raw/creditcard.csv and data/raw/sparkov/{fraudTrain,fraudTest}.csv already
# present (see README.md "Reproducing locally").
reproduce: reproduce-primary reproduce-sparkov manifest

reproduce-primary:
	python -m src.data.ingest
	python -m src.models.train_production_model
	python -m src.models.run_cost_analysis
	python -m src.models.run_calibration_analysis
	python -m src.models.run_bootstrap_analysis
	python -m src.models.run_evaluation_plots
	python -m src.models.run_shap_analysis
	$(MAKE) dashboard

reproduce-sparkov:
	python -m src.data.ingest_sparkov
	python -m src.models.train_sparkov_production_model
	python -m src.models.run_sparkov_validation

# docs/ (GitHub Pages) keeps its own copy of the figures it embeds, separate from reports/figures/
# — a figure regenerated there does not automatically show up on the live dashboard. This target
# regenerates the dashboard's exported JSON and re-syncs every figure it references, so the two
# copies can't silently drift the way confusion_matrix.png and cost_uncertainty_threshold_distribution.png
# did (caught and fixed on day 5 of the exact-search propagation, see RESEARCH_REPORT.md).
dashboard:
	python -m src.models.export_dashboard_data
	cp reports/figures/confusion_matrix.png docs/figures/confusion_matrix.png
	cp reports/figures/roc_curve.png docs/figures/roc_curve.png
	cp reports/figures/pr_curve.png docs/figures/pr_curve.png
	cp reports/figures/cost_vs_threshold.png docs/figures/cost_vs_threshold.png
	cp reports/figures/cost_ratio_sensitivity.png docs/figures/cost_ratio_sensitivity.png
	cp reports/figures/cost_uncertainty_threshold_distribution.png docs/figures/cost_uncertainty_threshold_distribution.png
	cp reports/figures/shap_summary.png docs/figures/shap_summary.png
	cp reports/figures/latency_histogram.png docs/figures/latency_histogram.png

manifest:
	python -m scripts.generate_manifest

test:
	pytest tests/ -q

lint:
	ruff check .
