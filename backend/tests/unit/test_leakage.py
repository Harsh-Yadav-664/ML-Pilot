import pytest
import pandas as pd
import numpy as np
from ml.validation.leakage import LeakageDetector

def test_leakage_detector_target_correlation():
    detector = LeakageDetector()
    df = pd.DataFrame({
        "feature_1": np.random.randn(100),
        "target": np.random.randn(100)
    })
    # Plant leakage (near perfect correlation)
    df["leaky_feature"] = df["target"] * 1.001
    
    warnings = detector.detect_target_leakage(df, "target")
    assert any(w.column == "leaky_feature" and w.severity == "high" for w in warnings)

def test_leakage_detector_target_name():
    detector = LeakageDetector()
    df = pd.DataFrame({
        "feature_1": np.random.randn(100),
        "is_target_yes": np.random.randn(100), # name leakage
        "target": np.random.randn(100)
    })
    
    warnings = detector.detect_target_leakage(df, "target")
    assert any(w.column == "is_target_yes" and w.severity == "high" for w in warnings)

def test_leakage_detector_timestamp():
    detector = LeakageDetector()
    df = pd.DataFrame({
        "feature_1": np.random.randn(100),
        "creation_date": pd.date_range("2020-01-01", periods=100),
        "target": np.random.randn(100)
    })
    
    warnings = detector.detect_timestamp_leakage(df, "target")
    assert any(w.column == "creation_date" and w.leakage_type == "temporal" for w in warnings)

def test_leakage_detector_entity_id():
    detector = LeakageDetector()
    df = pd.DataFrame({
        "feature_1": np.random.randn(100),
        "customer_id": range(100),
        "target": np.random.randn(100)
    })
    
    warnings = detector.detect_entity_leakage(df, "target")
    assert any(w.column == "customer_id" and w.leakage_type == "entity" for w in warnings)

def test_leakage_detector_preprocessing():
    detector = LeakageDetector()
    # Plant preprocessing leakage (perfectly normalized [0,1] feature without train/test split)
    df = pd.DataFrame({
        "feature_1": np.random.randn(100),
        "scaled_feature": np.linspace(0, 1, 100),
        "target": np.random.randn(100)
    })
    
    warnings = detector.detect_preprocessing_leakage(df, "target")
    assert any(w.column == "scaled_feature" and w.leakage_type == "preprocessing" for w in warnings)
