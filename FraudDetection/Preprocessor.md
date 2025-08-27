First, we need to preprocess the data before feeding it to the LSTM, since the LSTM works on a timing window basis with the memory of previous transactions.


```python
import math
import os
import numpy as np
import pandas as pd
import re
from requests import get
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import (
    LabelEncoder,
    OneHotEncoder,
    FunctionTransformer,
    MinMaxScaler,
    LabelBinarizer,
)
from sklearn_pandas import DataFrameMapper

```


```python
dist = pd.DataFrame({"No": [], "Yes": []})
df_nf = pd.DataFrame()
df_f = pd.DataFrame()

# Read and process the CSV in chunks
with pd.read_csv("card_transaction.v1.csv", chunksize=1_000_000) as reader:
    for chunk in reader:
        # Sample 5% of non-fraud rows
        df_nf = pd.concat([df_nf, chunk[chunk["Is Fraud?"] == "No"].sample(frac=0.05)])
        # Keep all fraud rows
        df_f = pd.concat([df_f, chunk[chunk["Is Fraud?"] == "Yes"]])
        # Track counts for ratio statistics
        vc = chunk["Is Fraud?"].value_counts()
        new = pd.DataFrame({"No": [vc.get("No", 0)], "Yes": [vc.get("Yes", 0)]})
        dist = pd.concat([dist, new])

# Save the results
df_nf.to_csv("card_transactions_non-frauds.csv", index=False)
df_f.to_csv("card_transactions_frauds.csv", index=False)
print(f"Ratio Fraud/Non-Fraud: {dist['Yes'].sum() / dist['No'].sum()}")
dist

```

    Ratio Fraud/Non-Fraud: 0.00122169500749739





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>No</th>
      <th>Yes</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>999008.0</td>
      <td>992.0</td>
    </tr>
    <tr>
      <th>0</th>
      <td>998768.0</td>
      <td>1232.0</td>
    </tr>
    <tr>
      <th>0</th>
      <td>998803.0</td>
      <td>1197.0</td>
    </tr>
    <tr>
      <th>0</th>
      <td>998856.0</td>
      <td>1144.0</td>
    </tr>
    <tr>
      <th>0</th>
      <td>998891.0</td>
      <td>1109.0</td>
    </tr>
    <tr>
      <th>0</th>
      <td>998675.0</td>
      <td>1325.0</td>
    </tr>
    <tr>
      <th>0</th>
      <td>998830.0</td>
      <td>1170.0</td>
    </tr>
    <tr>
      <th>0</th>
      <td>998686.0</td>
      <td>1314.0</td>
    </tr>
    <tr>
      <th>0</th>
      <td>998749.0</td>
      <td>1251.0</td>
    </tr>
    <tr>
      <th>0</th>
      <td>998721.0</td>
      <td>1279.0</td>
    </tr>
    <tr>
      <th>0</th>
      <td>999017.0</td>
      <td>983.0</td>
    </tr>
    <tr>
      <th>0</th>
      <td>998667.0</td>
      <td>1333.0</td>
    </tr>
    <tr>
      <th>0</th>
      <td>998623.0</td>
      <td>1377.0</td>
    </tr>
    <tr>
      <th>0</th>
      <td>998730.0</td>
      <td>1270.0</td>
    </tr>
    <tr>
      <th>0</th>
      <td>998835.0</td>
      <td>1165.0</td>
    </tr>
    <tr>
      <th>0</th>
      <td>998893.0</td>
      <td>1107.0</td>
    </tr>
    <tr>
      <th>0</th>
      <td>998559.0</td>
      <td>1441.0</td>
    </tr>
    <tr>
      <th>0</th>
      <td>998826.0</td>
      <td>1174.0</td>
    </tr>
    <tr>
      <th>0</th>
      <td>998610.0</td>
      <td>1390.0</td>
    </tr>
    <tr>
      <th>0</th>
      <td>998659.0</td>
      <td>1341.0</td>
    </tr>
    <tr>
      <th>0</th>
      <td>998835.0</td>
      <td>1165.0</td>
    </tr>
    <tr>
      <th>0</th>
      <td>998700.0</td>
      <td>1300.0</td>
    </tr>
    <tr>
      <th>0</th>
      <td>998730.0</td>
      <td>1270.0</td>
    </tr>
    <tr>
      <th>0</th>
      <td>998985.0</td>
      <td>1015.0</td>
    </tr>
    <tr>
      <th>0</th>
      <td>386487.0</td>
      <td>413.0</td>
    </tr>
  </tbody>
</table>
</div>



The transaction features need to be encoded as the LSTM requires numerical input.


```python
import pandas as pd
import numpy as np
import math

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import (
    FunctionTransformer, OneHotEncoder, MinMaxScaler, LabelBinarizer
)
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split


def fraud_encoder(X):
    # Accepts DataFrame or Series
    if isinstance(X, pd.DataFrame):
        X = X.iloc[:, 0]
    return np.where(X == "Yes", 1, 0).reshape(-1, 1)

def amt_encoder(X):
    # Accepts DataFrame or Series
    if isinstance(X, pd.DataFrame):
        X = X.iloc[:, 0]
    amt = (
        X.astype(str).str.replace("$", "", regex=False)
        .astype(float)
        .map(lambda amt: max(1, amt))
        .map(math.log)
    )
    return np.array(amt).reshape(-1, 1)

def decimal_encoder(X, length=5):
    if isinstance(X, pd.DataFrame):
        X = X.iloc[:, 0]
    X = X.astype(str).str.replace(r'\D', '', regex=True)
    X = X.replace('', '0').astype(int)
    arr = []
    for i in range(length):
        arr.append(np.mod(X, 10))
        X = np.floor_divide(X, 10)
    return np.column_stack(arr)

def time_encoder(df):
    X_hm = df["Time"].str.split(":", expand=True)
    d = pd.to_datetime(
        dict(
            year=df["Year"], month=df["Month"], day=df["Day"], hour=X_hm[0], minute=X_hm[1]
        )
    ).astype(int)
    return np.array(d).reshape(-1, 1)

def binarizer_func(x):
    if isinstance(x, pd.DataFrame):
        x = x.iloc[:, 0]
    return LabelBinarizer().fit_transform(x.astype(str)).reshape(-1, 1)

# Load data and define the columns

tdf = pd.read_csv("./card_transaction.v1.csv", nrows=1_000_000)
tdf["Merchant Name"] = tdf["Merchant Name"].astype(str)
tdf.drop(["MCC", "Zip", "Merchant State"], axis=1, inplace=True)
tdf.sort_values(by=["User", "Card"], inplace=True)
tdf.reset_index(inplace=True, drop=True)

fraud_col = "Is Fraud?"
merchant_name_col = "Merchant Name"
merchant_city_col = "Merchant City"
chip_col = "Use Chip"
errors_col = "Errors?"
time_cols = ["Year", "Month", "Day", "Time"]
amt_col = "Amount"

# Preprocessor pipeline

preprocessor = ColumnTransformer(
    transformers=[
        # Target label
        ("fraud", FunctionTransformer(fraud_encoder, validate=False), [fraud_col]),

        # Merchant Name: decimal encoder then one-hot
        ("merchant_name", Pipeline([
            ("decimal", FunctionTransformer(decimal_encoder, validate=False)),
            ("onehot", OneHotEncoder(sparse_output=False, handle_unknown='ignore')),
        ]), [merchant_name_col]),

        # Merchant City: decimal encoder then one-hot
        ("merchant_city", Pipeline([
            ("decimal", FunctionTransformer(decimal_encoder, validate=False)),
            ("onehot", OneHotEncoder(sparse_output=False, handle_unknown='ignore')),
        ]), [merchant_city_col]),

        # Use Chip: impute and binarize
        ("chip", Pipeline([
            ("imputer", SimpleImputer(strategy="constant", fill_value="missing_value")),
            ("onehot", OneHotEncoder(sparse_output=False, handle_unknown='ignore')),
        ]), [chip_col]),

        # Errors?: impute and binarize
        ("errors", Pipeline([
            ("imputer", SimpleImputer(strategy="constant", fill_value="missing_value")),
            ("onehot", OneHotEncoder(sparse_output=False, handle_unknown='ignore')),
        ]), [errors_col]),

        # Year/Month/Day/Time: encode and scale
        ("time", Pipeline([
            ("time_enc", FunctionTransformer(time_encoder, validate=False)),
            ("scaler", MinMaxScaler()),
        ]), time_cols),

        # Amount: custom encode and scale
        ("amount", Pipeline([
            ("amt_enc", FunctionTransformer(amt_encoder, validate=False)),
            ("scaler", MinMaxScaler()),
        ]), [amt_col]),
    ],
    remainder="drop"
)
processed_array = preprocessor.fit_transform(tdf)

# Retrieve feature names

feature_names = [f"feature_{i}" for i in range(processed_array.shape[1])]

print("Processed shape:", processed_array.shape)
print("First few rows:\n", processed_array[:5])
print("Feature names:", feature_names)

y = processed_array[:, 0]      # Label
X = processed_array[:, 1:]     # Features


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
np.savez("transactions_processed.npz", X=X, y=y)

```

    Processed shape: (1000000, 81)
    First few rows:
     [[0.         0.         0.         0.         1.         0.
      0.         0.         0.         0.         0.         0.
      0.         0.         0.         0.         1.         0.
      0.         0.         0.         0.         0.         0.
      0.         0.         0.         0.         0.         0.
      1.         0.         0.         0.         0.         0.
      0.         1.         0.         0.         0.         0.
      0.         0.         0.         0.         0.         0.
      1.         0.         0.         1.         1.         1.
      1.         1.         0.         0.         1.         0.
      0.         0.         0.         0.         0.         0.
      0.         0.         0.         0.         0.         0.
      0.         0.         0.         0.         0.         0.
      1.         0.26837517 0.55490584]
     [0.         0.         0.         0.         1.         0.
      0.         0.         0.         0.         0.         0.
      0.         0.         0.         1.         0.         0.
      0.         0.         0.         1.         0.         0.
      0.         0.         0.         0.         0.         0.
      0.         0.         0.         0.         0.         0.
      0.         1.         0.         0.         0.         0.
      1.         0.         0.         0.         0.         0.
      0.         0.         0.         1.         1.         1.
      1.         1.         0.         0.         1.         0.
      0.         0.         0.         0.         0.         0.
      0.         0.         0.         0.         0.         0.
      0.         0.         0.         0.         0.         0.
      1.         0.26837684 0.41348956]
     [0.         0.         0.         0.         1.         0.
      0.         0.         0.         0.         0.         0.
      0.         0.         0.         1.         0.         0.
      0.         0.         0.         1.         0.         0.
      0.         0.         0.         0.         0.         0.
      0.         0.         0.         0.         0.         0.
      0.         1.         0.         0.         0.         0.
      1.         0.         0.         0.         0.         0.
      0.         0.         0.         1.         1.         1.
      1.         1.         0.         0.         1.         0.
      0.         0.         0.         0.         0.         0.
      0.         0.         0.         0.         0.         0.
      0.         0.         0.         0.         0.         0.
      1.         0.26848975 0.54265   ]
     [0.         1.         0.         0.         0.         0.
      0.         0.         0.         0.         0.         0.
      0.         0.         0.         0.         0.         0.
      1.         0.         0.         0.         0.         0.
      0.         0.         0.         0.         1.         0.
      0.         0.         0.         0.         0.         0.
      0.         1.         0.         0.         0.         1.
      0.         0.         0.         0.         0.         0.
      0.         0.         0.         1.         1.         1.
      1.         1.         0.         0.         1.         0.
      0.         0.         0.         0.         0.         0.
      0.         0.         0.         0.         0.         0.
      0.         0.         0.         0.         0.         0.
      1.         0.26854406 0.5504781 ]
     [0.         0.         0.         0.         0.         0.
      0.         0.         1.         0.         0.         0.
      0.         0.         0.         0.         0.         1.
      0.         0.         0.         0.         0.         1.
      0.         0.         0.         0.         0.         0.
      0.         0.         0.         0.         0.         0.
      0.         1.         0.         0.         0.         0.
      0.         0.         1.         0.         0.         0.
      0.         0.         0.         1.         1.         1.
      1.         1.         0.         0.         1.         0.
      0.         0.         0.         0.         0.         0.
      0.         0.         0.         0.         0.         0.
      0.         0.         0.         0.         0.         0.
      1.         0.26860433 0.52688969]]
    Feature names: ['feature_0', 'feature_1', 'feature_2', 'feature_3', 'feature_4', 'feature_5', 'feature_6', 'feature_7', 'feature_8', 'feature_9', 'feature_10', 'feature_11', 'feature_12', 'feature_13', 'feature_14', 'feature_15', 'feature_16', 'feature_17', 'feature_18', 'feature_19', 'feature_20', 'feature_21', 'feature_22', 'feature_23', 'feature_24', 'feature_25', 'feature_26', 'feature_27', 'feature_28', 'feature_29', 'feature_30', 'feature_31', 'feature_32', 'feature_33', 'feature_34', 'feature_35', 'feature_36', 'feature_37', 'feature_38', 'feature_39', 'feature_40', 'feature_41', 'feature_42', 'feature_43', 'feature_44', 'feature_45', 'feature_46', 'feature_47', 'feature_48', 'feature_49', 'feature_50', 'feature_51', 'feature_52', 'feature_53', 'feature_54', 'feature_55', 'feature_56', 'feature_57', 'feature_58', 'feature_59', 'feature_60', 'feature_61', 'feature_62', 'feature_63', 'feature_64', 'feature_65', 'feature_66', 'feature_67', 'feature_68', 'feature_69', 'feature_70', 'feature_71', 'feature_72', 'feature_73', 'feature_74', 'feature_75', 'feature_76', 'feature_77', 'feature_78', 'feature_79', 'feature_80']


Save all relevant encoding data, to be used by the inference server


```python
import joblib, os, json
os.makedirs("models", exist_ok=True)

# Extract the fitted pieces to match the  ColumnTransformer blocks
mn_onehot   = preprocessor.named_transformers_["merchant_name"].named_steps["onehot"]
mc_onehot   = preprocessor.named_transformers_["merchant_city"].named_steps["onehot"]
chip_imp    = preprocessor.named_transformers_["chip"].named_steps["imputer"]
chip_onehot = preprocessor.named_transformers_["chip"].named_steps["onehot"]
err_imp     = preprocessor.named_transformers_["errors"].named_steps["imputer"]
err_onehot  = preprocessor.named_transformers_["errors"].named_steps["onehot"]
time_scaler = preprocessor.named_transformers_["time"].named_steps["scaler"]
amt_scaler  = preprocessor.named_transformers_["amount"].named_steps["scaler"]

tx_bundle = {
    "merchant_name_onehot": mn_onehot,
    "merchant_city_onehot": mc_onehot,
    "chip_imputer": chip_imp,
    "chip_onehot": chip_onehot,
    "errors_imputer": err_imp,
    "errors_onehot": err_onehot,
    "time_scaler": time_scaler,
    "amount_scaler": amt_scaler,

    # meta: column names the server will expect
    "columns": {
        "fraud": "Is Fraud?",
        "merchant_name": "Merchant Name",
        "merchant_city": "Merchant City",
        "chip": "Use Chip",
        "errors": "Errors?",
        "time": ["Year","Month","Day","Time"],
        "amount": "Amount",
        "group_keys": ["User","Card"]  # used to build sequences
    },
    # the decimal encoder length you used
    "decimal_length": 5
}

joblib.dump(tx_bundle, "models/inference_tx.joblib")
print("Saved models/inference_tx.joblib")
```

    Saved models/inference_tx.joblib

