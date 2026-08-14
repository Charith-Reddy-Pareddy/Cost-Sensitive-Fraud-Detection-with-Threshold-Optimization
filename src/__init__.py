import os

# XGBoost and PyTorch each ship their own OpenMP runtime. Loading both in one process (as the
# autoencoder and the tree-model baselines do across this project's test suite) races during
# thread-pool init and segfaults on macOS unless each library is pinned to a single thread.
# Must be set before torch/xgboost are imported anywhere, hence living in the package's
# top-level __init__ rather than in an individual module.
os.environ.setdefault("OMP_NUM_THREADS", "1")
