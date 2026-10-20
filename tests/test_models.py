"""
RetailPulse - Machine Learning Model Unit Tests
Validates model artifact loading, input feature formatting, and inference behavior.
"""
import pytest
import os
import joblib
import pandas as pd
import numpy as np
import config
from src.models.segmentation import CustomerSegmenter
from src.models.churn import ChurnPredictor
from src.models.forecasting import HybridDemandForecaster
from src.utils.drift_detector import calculate_psi


def test_psi_calculation():
    # Identical distributions should yield ~0 PSI
    ref = np.random.normal(100, 15, 1000)
    curr = np.random.normal(100, 15, 1000)
    psi = calculate_psi(ref, curr)
    assert psi < 0.10

    # Drastically shifted distributions should yield high PSI
    shifted = np.random.normal(150, 20, 1000)
    psi_shifted = calculate_psi(ref, shifted)
    assert psi_shifted >= 0.25


def test_segmentation_model_exists_and_predicts():
    assert config.SEGMENTATION_MODEL_FILE.exists()
    segmenter = CustomerSegmenter.load(config.SEGMENTATION_MODEL_FILE)
    test_rfm = pd.DataFrame({
        "Recency": [10, 150],
        "Frequency": [15, 1],
        "Monetary": [5000.0, 50.0]
    })
    preds = segmenter.predict(test_rfm)
    assert "Cluster" in preds.columns
    assert "Segment_Name" in preds.columns


def test_churn_model_exists_and_predicts():
    assert config.CHURN_MODEL_FILE.exists()
    predictor = ChurnPredictor.load(config.CHURN_MODEL_FILE)
    dummy_input = pd.DataFrame(
        np.random.rand(5, len(predictor.feature_cols)),
        columns=predictor.feature_cols
    )
    risk_df = predictor.predict_risk(dummy_input)
    assert len(risk_df) == 5
    assert "Churn_Probability" in risk_df.columns
    assert "Risk_Tier" in risk_df.columns
    assert all(0.0 <= p <= 1.0 for p in risk_df["Churn_Probability"])
