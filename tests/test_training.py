"""Unit tests for training modules."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from training.data_setup import Preprocessor, to_onnx_input
from training.model_builders import (
    build_lightgbm,
    build_model,
    build_xgboost,
    calculate_scale_pos_weight,
    load_config,
)
from training.onnx_export import export_and_validate
from training.train import (
    compute_metrics,
    generate_experiment_name,
    get_feature_importances,
    parse_parameters_arg,
)


def test_parse_parameters_arg():
    params = parse_parameters_arg(
        '{"n_estimators": 150, "learning_rate": 0.03, "tree_method": "hist"}'
    )
    assert params["n_estimators"] == 150
    assert params["learning_rate"] == 0.03
    assert params["tree_method"] == "hist"

    params_kv = parse_parameters_arg(
        "n_estimators=200, learning_rate=0.05, enable_categorical=True, max_depth=5"
    )
    assert params_kv["n_estimators"] == 200
    assert params_kv["learning_rate"] == 0.05
    assert params_kv["enable_categorical"] is True
    assert params_kv["max_depth"] == 5

    assert parse_parameters_arg(None) == {}
    assert parse_parameters_arg("") == {}


def test_load_config():
    cfg = load_config("configs/models.yml")
    assert "xgboost" in cfg
    assert "lightgbm" in cfg
    assert "training" in cfg


def test_calculate_scale_pos_weight():
    # 90 negatives, 10 positives -> weight = 9.0
    y = np.array([0] * 90 + [1] * 10)
    weight = calculate_scale_pos_weight(y)
    assert weight == 9.0


def test_generate_experiment_name():
    exp_name = generate_experiment_name("home_credit")
    assert exp_name.startswith("home_credit_")
    # Must contain colons for time
    assert ":" in exp_name


def test_model_builders_with_config():
    y = pd.Series([0] * 80 + [1] * 20)

    xgb = build_xgboost({"n_estimators": 25, "max_depth": 4}, y=y)
    assert xgb.n_estimators == 25
    assert xgb.max_depth == 4
    assert xgb.scale_pos_weight == 4.0

    lgb = build_lightgbm({"n_estimators": 35, "num_leaves": 15}, y=y)
    assert lgb.n_estimators == 35
    assert lgb.num_leaves == 15
    assert lgb.scale_pos_weight == 4.0

    m1 = build_model("xgboost", {"n_estimators": 10}, y=y)
    assert m1.n_estimators == 10

    with pytest.raises(ValueError):
        build_model("unsupported_model")


def test_preprocessor_with_scaling():
    X = pd.DataFrame(
        {
            "num1": [10.0, 20.0, 30.0, 40.0],
            "num2": [100.0, 200.0, 300.0, 400.0],
            "cat1": pd.Series(["A", "B", "A", "B"], dtype="category"),
        }
    )
    prep = Preprocessor.fit(X, scale_features=True)
    transformed = prep.transform(X)

    # Check that numerical columns are scaled to mean ~0
    assert abs(transformed["num1"].mean()) < 1e-5
    assert abs(transformed["num2"].mean()) < 1e-5
    assert isinstance(transformed["cat1"].dtype, pd.CategoricalDtype)


def test_compute_metrics():
    y_true = np.array([0, 0, 1, 1])
    y_prob = np.array([0.1, 0.2, 0.8, 0.9])
    metrics = compute_metrics(y_true, y_prob, prefix="test_")

    assert "test_roc_auc" in metrics
    assert metrics["test_roc_auc"] == 1.0
    assert metrics["test_accuracy"] == 1.0
    assert metrics["test_gini"] == 1.0


def test_feature_importances():
    m = build_lightgbm({"n_estimators": 10, "verbose": -1})
    X = pd.DataFrame({"f1": [1.0, 2.0, 3.0, 4.0], "f2": [4.0, 3.0, 2.0, 1.0]})
    y = pd.Series([0, 0, 1, 1])
    m.fit(X, y)

    df_imp = get_feature_importances(m, ["f1", "f2"])
    assert len(df_imp) == 2
    assert set(df_imp["feature"]) == {"f1", "f2"}


def test_onnx_export_and_validation(tmp_path: Path):
    np.random.seed(42)
    X = pd.DataFrame(
        {
            "num1": np.random.randn(50).astype(np.float32),
            "num2": np.random.randn(50).astype(np.float32),
            "cat1": pd.Series(np.random.choice(["A", "B"], size=50), dtype="category"),
        }
    )
    y = np.random.randint(0, 2, size=50)

    lgb_model = build_lightgbm({"n_estimators": 5, "verbose": -1})
    lgb_model.fit(X, y)

    onnx_file_lgb = export_and_validate(
        lgb_model,
        model_type="lightgbm",
        sample_df=X.iloc[:10],
        output_dir=tmp_path,
        filename="test_lgb.onnx",
    )
    assert onnx_file_lgb.exists()

    xgb_model = build_xgboost({"n_estimators": 5, "max_depth": 2})
    xgb_model.fit(X, y)

    onnx_file_xgb = export_and_validate(
        xgb_model,
        model_type="xgboost",
        sample_df=X.iloc[:10],
        output_dir=tmp_path,
        filename="test_xgb.onnx",
    )
    assert onnx_file_xgb.exists()


def test_to_onnx_input():
    df = pd.DataFrame(
        {
            "num": [1.0, 2.0, 3.0],
            "cat": pd.Series(["A", "B", "A"], dtype="category"),
        }
    )
    arr = to_onnx_input(df)
    assert isinstance(arr, np.ndarray)
    assert arr.shape == (3, 2)
    assert arr.dtype == np.float32
