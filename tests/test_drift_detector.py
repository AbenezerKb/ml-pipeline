import pytest
import pandas as pd
import numpy as np
import json
from pathlib import Path
import tempfile
import os
from unittest.mock import patch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data_pipeline.drift_detector import DriftDetector

def run_detect(detector, current_path, ref_path=None):

    import asyncio
    return asyncio.run(detector.detect_drift(current_path, ref_path))


@pytest.fixture
def sample_config():
    return {
        "enabled": True,
        "reference_data_path": "ref.parquet",
        "metrics": ["psi", "ks", "wasserstein"],
        "thresholds": {"psi": 0.2, "ks": 0.1, "wasserstein": 0.15},
        "categorical_columns": ["category"],
        "numerical_columns": ["value"],
    }


@pytest.fixture
def reference_data():
    np.random.seed(42)
    return pd.DataFrame(
        {
            "value": np.random.normal(0, 1, 100),
            "category": np.random.choice(["A", "B", "C"], size=100),
        }
    )


@pytest.fixture
def current_no_drift():
    np.random.seed(42)
    return pd.DataFrame(
        {
            "value": np.random.normal(0, 1, 80),
            "category": np.random.choice(["A", "B", "C"], size=80),
        }
    )


@pytest.fixture
def current_with_drift():
    np.random.seed(123)
    return pd.DataFrame(
        {
            "value": np.random.normal(5, 1, 80),          # shifted mean
            "category": np.random.choice(["X", "Y"], size=80),  # new cats
        }
    )


