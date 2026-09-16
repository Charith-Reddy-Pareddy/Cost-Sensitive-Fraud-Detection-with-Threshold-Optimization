import numpy as np
import pandas as pd
from omegaconf import OmegaConf

from experiments.run import _train_model


def test_train_model_uses_config_hyperparameters():
    rng = np.random.default_rng(0)
    n = 500
    X = pd.DataFrame(rng.normal(size=(n, 5)), columns=[f"f{i}" for i in range(5)])
    y = pd.Series((rng.random(n) < 0.1).astype(int))

    cfg = OmegaConf.create({"model": {"n_estimators": 10, "max_depth": 2, "learning_rate": 0.3}})
    model = _train_model(cfg, X, y)

    assert model.n_estimators == 10
    assert model.max_depth == 2
    proba = model.predict_proba(X)[:, 1]
    assert proba.shape == (n,)
    assert (proba >= 0).all() and (proba <= 1).all()


def test_all_configs_parse_and_resolve():
    import hydra

    for config_name in ("primary", "sparkov", "cost_sweep", "temporal"):
        with hydra.initialize(version_base=None, config_path="../configs"):
            cfg = hydra.compose(config_name=config_name)
        assert cfg.dataset in ("primary", "sparkov")
        assert cfg.mode in ("cost_analysis", "cost_sweep", "temporal")
        assert cfg.cost.fn > 0
        assert cfg.cost.fp > 0
