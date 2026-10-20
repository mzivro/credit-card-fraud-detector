# Credit Card Fraud Detector

A **XGBoost**-based machine learning application for detecting potentially fraudulent credit card transactions. The project provides a REST API built with FastAPI and a web interface built with Streamlit. Works with **99% of accuracy**.

Try it out: https://mzivro-credit-fraud-det.streamlit.app/

> **Note** This project was made for educational purposes and using it for **real** fraud detection is not recommended.

## Features

- XGBoost classifier trained to distinguish fraudulent from legitimate transactions working with 99% of accuracy.
- Standard scaling for numerical features and one-hot encoding for categorical features.
- SMOTE oversampling applied during model training.
- Automatically derives additional transaction risk indicators.
- FastAPI endpoints for health checks, readiness checks, and fraud predictions.
- Streamlit client for entering transaction attributes and viewing predictions.
- Downloads the trained model from the Hugging Face Hub at application startup.
- Demo app, training script and notebook with whole training process are provided.

## Technologies

- Pandas
- Numpy
- scikit-learn
- XGBoost
- imbalanced-learn
- FastAPI
- Pydantic
- Streamlit
- Hugging Face Hub
- KaggleHub

## Input Features

The model uses eight transaction attributes.

| Feature | Description |
|---|---|
| `amount` | Transaction amount |
| `transaction_hour` | Hour of the transaction, from 0 to 23 |
| `merchant_category` | Merchant category |
| `foreign_transaction` | Whether the transaction was made abroad |
| `location_mismatch` | Whether the transaction location is inconsistent with the expected location |
| `device_trust_score` | Device trust score, from 0 to 100 |
| `velocity_last_24h` | Number of transactions during the previous 24 hours |
| `cardholder_age` | Cardholder's age in years |

The application also derives three additional features:

| Feature | Rule |
|---|---|
| `nighttime_transaction` | `1` if `transaction_hour <= 6`, otherwise `0` |
| `suspicious_device` | `1` if `device_trust_score <= 50`, otherwise `0` |
| `multiple_transaction` | `1` if `velocity_last_24h >= 5`, otherwise `0` |

These derived features are generated automatically before inference.

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/mzivro/credit-card-fraud-detector.git
cd credit-card-fraud-detector
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on your operating system.

**Windows (PowerShell):**

```powershell
.venv\Scripts\Activate.ps1
```

**Linux / macOS:**

```bash
source .venv/bin/activate
```

### 3. Install dependencies

Install the packages required by the application:

```bash
pip install -r requirements.txt
```

## Running the Application

The application requires a running FastAPI backend and a separate Streamlit frontend.

### 1. Start the API server

From the project root, run:

```bash
fastapi run src/server.py
```

The API will be available at:

```text
http://localhost:8000
```

Interactive API documentation is available at:

```text
http://localhost:8000/docs
```

The model is downloaded from the Hugging Face Hub and loaded during application startup.

### 2. Start the Streamlit client

Open another terminal, activate the same virtual environment, and run:

```bash
streamlit run src/client.py
```

Streamlit will display the local URL for the web interface, typically:

```text
http://localhost:8501
```

Enter the transaction attributes and click **Predict** to receive a classification result.

### 3. Check the API status

**Health check**

```bash
curl http://localhost:8000/health
```

Example response:

```json
{
  "status": "ok"
}
```

**Readiness check**

```bash
curl http://localhost:8000/ready
```

Example response:

```json
{
  "status": "ready"
}
```

The readiness endpoint returns HTTP `503` if the model has not been loaded.

## API Reference

### `POST /predict`

Accepts transaction attributes and returns the predicted fraud classification.

**Example request:**

```json
{
  "amount": 250.0,
  "transaction_hour": 14,
  "merchant_category": "Electronics",
  "foreign_transaction": 0,
  "location_mismatch": 0,
  "device_trust_score": 85,
  "velocity_last_24h": 2,
  "cardholder_age": 28
}
```

**Example response:**

```json
{
  "prediction": 0
}
```

**Prediction values:**

| Value | Meaning |
|---|---|
| `0` | No fraud detected |
| `1` | Fraud detected |

## Dataset

This model was trained using **Credit Card Fraud Detection Dataset** from **Kaggle**. This dataset is under **Apache 2.0** license.

Source: https://www.kaggle.com/datasets/miadul/credit-card-fraud-detection-dataset

## License

MIT License. Feel free to use, modify, and build upon this project.
