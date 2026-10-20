from src.schemas import PredictionRequest, PredictionResponse
from fastapi import FastAPI, Request, HTTPException
from contextlib import asynccontextmanager
from src.model import Model


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage the classification model's application lifecycle.
    Loads the classification model during application startup and
    removes its reference from the application state during shutdown.

    Parameters
    ----------
    app : FastAPI
        Application instance whose lifecycle is being managed.

    Yields
    ------
    None
        Yields control after model initialization, allowing the
        application to serve requests until shutdown begins.

    Notes
    -----
    The loaded model is stored in app.state.model and can be
    accessed by endpoint handlers throughout the application's
    lifetime.
    """
    app.state.model = Model()

    yield

    del app.state.model


app = FastAPI(title="Credit Fraud Detector API", version="1.0.0", lifespan=lifespan)


@app.get("/health")
def health() -> dict:
    """Check whether the API service is responding.

    Returns
    -------
    dict
        Dictionary containing the service status. The status field
        is set to "ok" when the endpoint responds successfully.
    """
    return {"status": "ok"}


@app.get("/ready")
def ready() -> dict:
    """Check whether the classification model has been initialized.

    Returns
    -------
    dict
        Dictionary containing the readiness status. The status
        field is set to "ready" when the model is available.

    Raises
    ------
    HTTPException
        Raises an HTTP 503 error if the model has not been loaded.
    """
    if not hasattr(app.state, "model"):
        raise HTTPException(
            status_code=503,
            detail="Model not loaded",
        )

    return {
        "status": "ready",
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest) -> PredictionResponse:
    """Predict whether a submitted credit card transaction is fraudulent.
    Retrieves the classification model from the application state,
    passes the validated transaction attributes to the model, and
    returns the resulting prediction using the response schema.

    Parameters
    ----------
    request : PredictionRequest
        Validated transaction attributes submitted by the client.

    Returns
    -------
    PredictionResponse
        Response object containing the binary fraud classification:
        1 indicates fraud and 0 indicates a non-fraudulent transaction.

    Raises
    ------
    AttributeError
        May be raised if the model is unavailable in the application
        state when the endpoint is called.
    """
    model = app.state.model

    prediction = app.state.model.predict(
        request.amount,
        request.transaction_hour,
        request.merchant_category,
        request.foreign_transaction,
        request.location_mismatch,
        request.device_trust_score,
        request.velocity_last_24h,
        request.cardholder_age,
    )

    return PredictionResponse(prediction=prediction)
