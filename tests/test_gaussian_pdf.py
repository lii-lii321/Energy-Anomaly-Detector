import os
import pickle

import numpy as np
import pytest
from conftest import REPO_ROOT
from scipy.stats import multivariate_normal

MODEL_PATH = os.path.join(REPO_ROOT, "models", "energy_model.pkl")


def load_model():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


def test_model_file_contains_two_dimensional_parameters():
    model = load_model()
    assert model["mu"].shape == (2,)
    assert model["sigma"].shape == (2, 2)


def test_covariance_is_symmetric_and_positive_definite():
    sigma = load_model()["sigma"]
    assert np.allclose(sigma, sigma.T)
    eigenvalues = np.linalg.eigvalsh(sigma)
    assert np.all(eigenvalues > 0)


def test_pdf_matches_closed_form_for_the_model_covariance():
    model = load_model()
    mu, sigma = model["mu"], model["sigma"]
    sample = np.array([2.05, 14.9])

    diff = sample - mu
    det = np.linalg.det(sigma)
    closed_form = np.exp(-0.5 * diff @ np.linalg.solve(sigma, diff)) / (
        2 * np.pi * np.sqrt(det)
    )

    assert multivariate_normal.pdf(sample, mean=mu, cov=sigma) == pytest.approx(
        closed_form, rel=1e-12
    )


def test_pdf_peaks_at_the_mean_and_decays_outward():
    mu = np.zeros(2)
    sigma = np.eye(2)
    at_mean = multivariate_normal.pdf(mu, mean=mu, cov=sigma)
    near = multivariate_normal.pdf([0.1, 0.1], mean=mu, cov=sigma)
    far = multivariate_normal.pdf([1.0, 1.0], mean=mu, cov=sigma)

    assert at_mean == pytest.approx(1 / (2 * np.pi), rel=1e-12)
    assert at_mean > near > far > 0


def test_pdf_with_diagonal_covariance_factorizes_into_independent_terms():
    sigma = np.diag([4.0, 9.0])
    sample = np.array([1.0, -1.0])

    joint = multivariate_normal.pdf(sample, mean=np.zeros(2), cov=sigma)
    factorized = multivariate_normal.pdf(sample[0], mean=0, cov=4.0) * (
        multivariate_normal.pdf(sample[1], mean=0, cov=9.0)
    )

    assert joint == pytest.approx(factorized, rel=1e-12)
