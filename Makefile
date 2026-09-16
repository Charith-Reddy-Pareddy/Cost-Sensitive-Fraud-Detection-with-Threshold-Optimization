.PHONY: reproduce reproduce-primary reproduce-sparkov manifest test lint

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
	python -m src.models.export_dashboard_data

reproduce-sparkov:
	python -m src.data.ingest_sparkov
	python -m src.models.train_sparkov_production_model
	python -m src.models.run_sparkov_validation

manifest:
	python -m scripts.generate_manifest

test:
	pytest tests/ -q

lint:
	ruff check .
