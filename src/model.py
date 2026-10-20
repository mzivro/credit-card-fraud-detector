from huggingface_hub import hf_hub_download

import pandas as pd
import pickle


class Model:
    """Load and execute a pretrained credit card fraud detection model.
    The model is downloaded from the Hugging Face Hub and deserialized
    using pickle. It predicts whether a credit card transaction is
    fraudulent based on its attributes and derived features.

    Attributes
    ----------
    model : object
        Deserialized, pretrained scikit-learn-compatible classification
        pipeline used to generate predictions.
    """

    def __init__(self) -> None:
        """Initialize the model by downloading and loading its artifact.

        Downloads the serialized classification pipeline from the
        mzivro/my-models repository on the Hugging Face Hub and
        loads it into the model attribute.

        Raises
        ------
        Exception
            Propagates exceptions raised during model download, file
            access, or deserialization.
        """
        path = hf_hub_download(
            repo_id="mzivro/my-models",
            filename="credit_card_fraud_detector.pkl",
            repo_type="model",
        )

        with open(path, "rb") as file:
            self.model = pickle.load(file)

    def predict(
        self,
        amount: float,
        transaction_hour: int,
        merchant_category: str,
        foreign_transaction: int,
        location_mismatch: int,
        device_trust_score: int,
        velocity_last_24h: int,
        cardholder_age: int,
    ) -> int:
        """Predict whether a credit card transaction is fraudulent.

        Constructs a single-row DataFrame containing the transaction
        attributes and derives additional features before passing the
        data to the pretrained classification pipeline.

        Parameters
        ----------
        amount : float
            Monetary value of the transaction.
        transaction_hour : int
            Hour of the transaction, represented as an integer from 0
            to 23.
        merchant_category : str
            Category of the merchant associated with the transaction.
        foreign_transaction : int
            Indicator of whether the transaction was made abroad.
            Expected to be encoded as 0 or 1.
        location_mismatch : int
            Indicator of whether the transaction location differs from
            the cardholder's expected location. Expected to be encoded
            as 0 or 1.
        device_trust_score : int
            Trust score assigned to the device, on a scale from 0 to 100.
        velocity_last_24h : int
            Number of transactions recorded during the preceding
            24-hour period.
        cardholder_age : int
            Age of the cardholder in years.

        Returns
        -------
        int
            Binary prediction returned by the classification pipeline:
            1 indicates fraud and 0 indicates a non-fraudulent
            transaction.

        Notes
        -----
        The following derived features are calculated before prediction:

        - nighttime_transaction: 1 when transaction_hour <= 6.
        - suspicious_device: 1 when device_trust_score <= 50.
        - multiple_transaction: 1 when velocity_last_24h >= 5.

        These feature definitions must remain consistent with the
        preprocessing and feature engineering used during training.
        """
        input_data = pd.DataFrame(
            [
                {
                    "amount": amount,
                    "transaction_hour": transaction_hour,
                    "merchant_category": merchant_category,
                    "foreign_transaction": foreign_transaction,
                    "location_mismatch": location_mismatch,
                    "device_trust_score": device_trust_score,
                    "velocity_last_24h": velocity_last_24h,
                    "cardholder_age": cardholder_age,
                    "nighttime_transaction": 1 if transaction_hour <= 6 else 0,
                    "suspicious_device": 1 if device_trust_score <= 50 else 0,
                    "multiple_transaction": 1 if velocity_last_24h >= 5 else 0,
                }
            ]
        )

        return int(self.model.predict(input_data)[0])
