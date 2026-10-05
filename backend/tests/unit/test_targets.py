"""Tests for the shared classification target encoder."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ml.core.targets import TargetEncoder


def test_binary_minority_is_positive_and_encoded_as_one():
    enc = TargetEncoder.fit(pd.Series(["No", "No", "No", "Yes"]))
    assert enc.to_dict() == {"classes": ["No", "Yes"], "positive_class": "Yes"}
    assert enc.transform(["Yes", "No"]).tolist() == [1, 0]
    assert enc.inverse_transform([1, 0]) == ["Yes", "No"]


def test_binary_explicit_positive_class():
    enc = TargetEncoder.fit(["No", "No", "Yes"], positive_class="No")
    assert enc.classes == ["Yes", "No"]
    assert enc.transform(["No"]).tolist() == [1]


def test_binary_tie_picks_last_sorted_label():
    assert TargetEncoder.fit(["a", "b"]).positive_class == "b"


def test_numeric_labels_keep_minority_positive_and_are_json_safe():
    enc = TargetEncoder.fit(np.array([1, 1, 1, 0]))
    assert enc.to_dict() == {"classes": [1, 0], "positive_class": 0}
    assert type(enc.classes[0]) is int


def test_multiclass_uses_sorted_order():
    enc = TargetEncoder.fit(["b", "c", "a", "a"])
    assert enc.classes == ["a", "b", "c"] and enc.positive_class is None
    assert enc.transform(["c", "a"]).tolist() == [2, 0]


def test_unseen_label_raises():
    enc = TargetEncoder.fit(["No", "Yes"])
    with pytest.raises(ValueError, match="not seen in training"):
        enc.transform(["Maybe"])


def test_unknown_positive_class_raises():
    with pytest.raises(ValueError, match="positive_class"):
        TargetEncoder.fit(["No", "Yes"], positive_class="Maybe")


def test_single_class_raises():
    with pytest.raises(ValueError, match="at least 2 classes"):
        TargetEncoder.fit(["No", "No"])


def test_round_trip_dict():
    enc = TargetEncoder.fit(["No", "Yes", "No"])
    assert TargetEncoder.from_dict(enc.to_dict()).to_dict() == enc.to_dict()
