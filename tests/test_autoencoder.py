import numpy as np

from src.models.autoencoder import fit_autoencoder, reconstruction_error


def test_fit_and_score_shapes():
    rng = np.random.default_rng(0)
    X_train = rng.normal(size=(200, 10)).astype(np.float32)
    model = fit_autoencoder(X_train, epochs=5)

    errors = reconstruction_error(model, X_train)
    assert errors.shape == (200,)
    assert (errors >= 0).all()


def test_out_of_distribution_rows_score_higher():
    rng = np.random.default_rng(0)
    X_train = rng.normal(size=(500, 10)).astype(np.float32)
    model = fit_autoencoder(X_train, epochs=30)

    in_dist = rng.normal(size=(50, 10)).astype(np.float32)
    out_of_dist = rng.normal(loc=8.0, scale=1.0, size=(50, 10)).astype(np.float32)

    in_dist_error = reconstruction_error(model, in_dist).mean()
    out_of_dist_error = reconstruction_error(model, out_of_dist).mean()

    assert out_of_dist_error > in_dist_error
