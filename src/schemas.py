from pydantic import BaseModel


class PredictionRequest(BaseModel):
    """Validate transaction attributes submitted to the prediction API.
    This Pydantic model defines the request schema for the fraud
    detection endpoint and validates the types of the supplied fields.

    Attributes
    ----------
    amount : float
        Monetary value of the transaction.
    transaction_hour : int
        Hour of the transaction, from 0 to 23.
    merchant_category : str
        Category of the merchant associated with the transaction.
    foreign_transaction : int
        Binary indicator of a foreign transaction.
    location_mismatch : int
        Binary indicator of a mismatch between the transaction location
        and the expected location.
    device_trust_score : int
        Device trust score, expected to be between 0 and 100.
    velocity_last_24h : int
        Number of transactions during the preceding 24 hours.
    cardholder_age : int
        Age of the cardholder in years.
    """

    amount: float
    transaction_hour: int
    merchant_category: str
    foreign_transaction: int
    location_mismatch: int
    device_trust_score: int
    velocity_last_24h: int
    cardholder_age: int


class PredictionResponse(BaseModel):
    """Represent the response returned by the prediction API.

    Attributes
    ----------
    prediction : int
        Binary classification result: 1 indicates a fraudulent
        transaction, while 0 indicates a non-fraudulent transaction.
    """

    prediction: int
