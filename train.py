from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from xgboost import XGBClassifier

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_score,
    accuracy_score,
    roc_auc_score,
    recall_score,
    f1_score,
)

import pandas as pd
import numpy as np
import kagglehub
import pickle
import os

# Get data
path = kagglehub.dataset_download("miadul/credit-card-fraud-detection-dataset")

print("Dataset path:", path)

data = pd.read_csv(
    os.path.join(path, "credit_card_fraud_10k.csv"), index_col="transaction_id"
)

# Transform data
data["nighttime_transaction"] = (data["transaction_hour"] <= 6).astype(int)
data["suspicious_device"] = (data["device_trust_score"] <= 50).astype(int)
data["multiple_transaction"] = (data["velocity_last_24h"] >= 5).astype(int)

# Split data to train and test set
num_cols = [
    "amount",
    "transaction_hour",
    "device_trust_score",
    "velocity_last_24h",
    "cardholder_age",
]
cat_cols = [
    "merchant_category",
    "foreign_transaction",
    "location_mismatch",
    "nighttime_transaction",
    "suspicious_device",
    "multiple_transaction",
]

X = data.drop(columns=["is_fraud"])
y = data["is_fraud"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=8)

# Build pipeline
preprocessor = ColumnTransformer(
    [
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
    ],
    remainder="passthrough",
)

xg_pipe = Pipeline(
    [
        ("preprocessor", preprocessor),
        (
            "smote",
            SMOTE(
                random_state=8,
                k_neighbors=8,
                sampling_strategy="auto",
            ),
        ),
        (
            "classifier",
            XGBClassifier(
                random_state=8,
                eval_metric="mlogloss",
                tree_method="hist",
                colsample_bytree=np.float64(0.8335858174003867),
                gamma=np.float64(0.19456148821160923),
                learning_rate=np.float64(0.045647896045325355),
                max_depth=9,
                min_child_weight=4,
                n_estimators=415,
                reg_alpha=np.float64(0.0011928586861882087),
                reg_lambda=np.float64(0.005686571947514156),
                subsample=np.float64(0.8144577226718298),
                n_jobs=-1,
            ),
        ),
    ]
)

# Train and evalute model
xg_pipe.fit(X_train, y_train)

y_pred = xg_pipe.predict(X_test)

print("Accuracy     :", accuracy_score(y_test, y_pred))
print("Precision    :", precision_score(y_test, y_pred))
print("Recall       :", recall_score(y_test, y_pred))
print("F1-score     :", f1_score(y_test, y_pred))
print("ROC-AUC score:", roc_auc_score(y_test, y_pred))
print("Classification report:\n")
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

# Save model
with open("credit_card_fraud_detector.pkl", "wb") as file:
    pickle.dump(xg_pipe, file)

print("Model successfuly trained and saved!")
