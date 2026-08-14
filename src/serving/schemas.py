from pydantic import BaseModel, create_model

from src.features.pipeline import RAW_FEATURE_COLUMNS

# One field per raw feature column, built from the same list everywhere else in the project
# uses — so the request schema can never silently drift out of sync with the training contract.
TransactionRequest = create_model(
    "TransactionRequest",
    **{col: (float, ...) for col in RAW_FEATURE_COLUMNS},
)


class PredictionResponse(BaseModel):
    fraud_probability: float
    is_fraud: bool
    threshold: float
    latency_ms: float
