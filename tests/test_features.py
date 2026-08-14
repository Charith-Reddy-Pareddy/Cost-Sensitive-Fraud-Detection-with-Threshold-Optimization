import pandas as pd

from src.features.pipeline import PIPELINE_OUTPUT_COLUMNS, RAW_FEATURE_COLUMNS, build_preprocessor


def _toy_frame(n: int = 10) -> pd.DataFrame:
    data = {f"V{i}": [float(i)] * n for i in range(1, 29)}
    data["Amount"] = [10.0 * j for j in range(n)]
    data["Time"] = [float(j) for j in range(n)]
    return pd.DataFrame(data)[RAW_FEATURE_COLUMNS]


def test_preprocessor_output_shape():
    df = _toy_frame()
    preprocessor = build_preprocessor()
    out = preprocessor.fit_transform(df)
    assert out.shape == (len(df), len(RAW_FEATURE_COLUMNS))


def test_preprocessor_output_column_order():
    df = _toy_frame()
    preprocessor = build_preprocessor()
    preprocessor.fit(df)
    assert list(preprocessor.get_feature_names_out()) == [
        f"scale__{c}" if c in ("Amount", "Time") else f"remainder__{c}"
        for c in PIPELINE_OUTPUT_COLUMNS
    ]


def test_scaled_columns_are_standardized():
    df = _toy_frame(n=50)
    preprocessor = build_preprocessor()
    out = preprocessor.fit_transform(df)
    amount_col = out[:, 0]
    assert abs(amount_col.mean()) < 1e-6
    assert abs(amount_col.std() - 1.0) < 1e-6
