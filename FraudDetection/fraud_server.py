import os
import json
from collections import defaultdict, deque
from typing import List, Dict, Any

import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
import joblib

# Models
import tensorflow as tf
from tensorflow.keras.models import load_model
from xgboost import XGBClassifier

# Configuration loading
MODELS_DIR = os.environ.get("MODELS_DIR", "models")
CFG_PATH   = os.path.join(MODELS_DIR, "ensemble_config.json")
TX_PATH    = os.path.join(MODELS_DIR, "inference_tx.joblib")
LSTM_PATH  = os.path.join(MODELS_DIR, "lstm.keras")
XGB_PATH   = os.path.join(MODELS_DIR, "xgb.json")

with open(CFG_PATH, "r") as f:
    CFG = json.load(f)

SEQ_LEN      = int(CFG["sequence_length"])   # 7
NUM_FEATS    = int(CFG["num_features"])      # 80
ENS_WEIGHTS  = CFG.get("weights", {"lstm":0.5, "xgb":0.5})

tx   = joblib.load(TX_PATH)
COLS = tx["columns"]
DEC_LEN = int(tx.get("decimal_length", 5))

# Keras + XGB
lstm_model = load_model(LSTM_PATH, compile=False)
xgb = XGBClassifier()
xgb.load_model(XGB_PATH)

# Per-group rolling buffers for streaming
BUFFERS: Dict[tuple, deque] = defaultdict(lambda: deque(maxlen=SEQ_LEN))

# Encoders to match training phase
def amt_encoder_series(s: pd.Series) -> np.ndarray:
    ss = (s.astype(str).str.replace("$", "", regex=False)
            .astype(float)
            .map(lambda amt: max(1, amt))
            .map(np.log))
    return ss.to_numpy(dtype=np.float32).reshape(-1, 1)

def decimal_encoder_series(s: pd.Series, length: int = 5) -> np.ndarray:
    x = s.astype(str).str.replace(r"\D", "", regex=True)
    x = x.replace("", "0").astype(np.int64)
    cols = []
    for _ in range(length):
        cols.append(np.mod(x, 10).to_numpy())
        x = np.floor_divide(x, 10)
    return np.column_stack(cols).astype(np.float32)

def time_encoder_df(df_time: pd.DataFrame) -> np.ndarray:
    # df_time must contain Year, Month, Day, Time (HH:MM)
    hm = df_time["Time"].astype(str).str.split(":", expand=True)
    h = pd.to_numeric(hm[0], errors="coerce").fillna(0).astype(int)
    m = pd.to_numeric(hm[1], errors="coerce").fillna(0).astype(int)
    dt = pd.to_datetime(
        dict(year=df_time["Year"], month=df_time["Month"], day=df_time["Day"], hour=h, minute=m),
        errors="coerce"
    ).fillna(pd.Timestamp("1970-01-01"))
    return dt.astype("int64").to_numpy().reshape(-1,1).astype(np.float32)  # ns since epoch

def preprocess_rows_to_features(df: pd.DataFrame) -> np.ndarray:
    """
    Recreate training-time order:
      merchant_name(decimal->onehot),
      merchant_city(decimal->onehot),
      chip(impute->onehot),
      errors(impute->onehot),
      time(encode->scale),
      amount(encode->scale)
    """
    # Ensure required columns are present
    req = {
        "merchant_name": COLS["merchant_name"],
        "merchant_city": COLS["merchant_city"],
        "chip": COLS["chip"],
        "errors": COLS["errors"],
        "time": COLS["time"],
        "amount": COLS["amount"],
    }
    for k, col in req.items():
        if isinstance(col, list):
            missing = [c for c in col if c not in df.columns]
            if missing:
                raise ValueError(f"Missing columns for '{k}': {missing}")
        else:
            if col not in df.columns:
                raise ValueError(f"Missing column '{col}'")

    # Merchant Name
    mn_dec = decimal_encoder_series(df[COLS["merchant_name"]], length=DEC_LEN)
    mn_oh  = tx["merchant_name_onehot"].transform(mn_dec)

    # Merchant City
    mc_dec = decimal_encoder_series(df[COLS["merchant_city"]], length=DEC_LEN)
    mc_oh  = tx["merchant_city_onehot"].transform(mc_dec)

    # Use Chip
    chip_raw = df[[COLS["chip"]]].astype(str)
    chip_imp = tx["chip_imputer"].transform(chip_raw)
    chip_oh  = tx["chip_onehot"].transform(chip_imp)

    # Errors?
    err_raw = df[[COLS["errors"]]].astype(str)
    err_imp = tx["errors_imputer"].transform(err_raw)
    err_oh  = tx["errors_onehot"].transform(err_imp)

    # Time
    t_enc = time_encoder_df(df[COLS["time"]])
    t_scl = tx["time_scaler"].transform(t_enc)

    # Amount
    a_enc = amt_encoder_series(df[COLS["amount"]])
    a_scl = tx["amount_scaler"].transform(a_enc)

    X = np.concatenate([mn_oh, mc_oh, chip_oh, err_oh, t_scl, a_scl], axis=1).astype(np.float32)
    if X.shape[1] != NUM_FEATS:
        raise ValueError(f"Feature width mismatch: expected {NUM_FEATS}, got {X.shape[1]}")
    return X

def build_sequences(
    X: np.ndarray,
    df: pd.DataFrame,
    group_keys: List[str],
    seq_len: int,
    pad_policy: str = "none"  # "none" | "repeat" | "zero"
) -> Dict[str, Any]:
    """
    Build sliding sequences of length `seq_len` per group.
    If a group has < seq_len rows:
      - "none"  -> skip (default)
      - "repeat"-> left-pad by repeating the first row
      - "zero"  -> left-pad with zeros
    """
    hm = df["Time"].astype(str).str.split(":", expand=True)
    h = pd.to_numeric(hm[0], errors="coerce").fillna(0).astype(int)
    m = pd.to_numeric(hm[1], errors="coerce").fillna(0).astype(int)
    ts = pd.to_datetime(
        dict(year=df["Year"], month=df["Month"], day=df["Day"], hour=h, minute=m),
        errors="coerce"
    ).fillna(pd.Timestamp("1970-01-01"))
    df = df.copy()
    df["_ts"] = ts.values

    seqs, anchors = [], []
    for key_vals, g_idx in df.groupby(group_keys).groups.items():
        g = df.loc[list(g_idx)].sort_values("_ts")
        feats = X[g.index.to_numpy()]
        n = feats.shape[0]
        group_dict = dict(zip(group_keys, key_vals if isinstance(key_vals, tuple) else (key_vals,)))

        if n < seq_len:
            if pad_policy == "none":
                continue
            if pad_policy == "repeat":
                pad = np.repeat(feats[:1], seq_len - n, axis=0)
            elif pad_policy == "zero":
                pad = np.zeros((seq_len - n, feats.shape[1]), dtype=np.float32)
            else:
                raise ValueError(f"Unknown pad_policy: {pad_policy}")
            window = np.concatenate([pad, feats], axis=0)
            seqs.append(window.astype(np.float32))
            end_row = g.iloc[-1]
            anchors.append({
                "group": group_dict,
                "end_index": int(end_row.name),
                "end_time": str(end_row["_ts"]),
                "end_id": None,
            })
            continue

        # Normal sliding windows
        for i in range(n - seq_len + 1):
            window = feats[i:i+seq_len]
            seqs.append(window.astype(np.float32))
            end_row = g.iloc[i+seq_len-1]
            anchors.append({
                "group": group_dict,
                "end_index": int(end_row.name),
                "end_time": str(end_row["_ts"]),
                "end_id": None,
            })

    if not seqs:
        return {"sequences": np.zeros((0, seq_len, X.shape[1]), dtype=np.float32), "anchors": []}
    return {"sequences": np.stack(seqs), "anchors": anchors}

def ensemble_predict(seqs: np.ndarray, w_lstm: float, w_xgb: float) -> Dict[str, np.ndarray]:
    # LSTM on full sequences
    p_lstm = lstm_model.predict(seqs, batch_size=1024, verbose=0).ravel()
    # XGB on last-timestep features
    last_feats = seqs[:, -1, :]
    p_xgb = xgb.predict_proba(last_feats)[:, 1]
    s = w_lstm + w_xgb
    w_lstm /= s; w_xgb /= s
    p_ens = w_lstm * p_lstm + w_xgb * p_xgb
    return {"lstm": p_lstm, "xgb": p_xgb, "ens": p_ens}

# Flask server
app = Flask(__name__)

@app.get("/health")
def health():
    return jsonify({
        "status": "ok",
        "sequence_length": SEQ_LEN,
        "num_features": NUM_FEATS,
        "weights": ENS_WEIGHTS,
        "expected_columns": COLS,
    })

# Batch processor
@app.post("/predict")
def predict():
    """
    Request JSON:
    {
      "records": [ {...}, {...}, ... ],         # raw rows with CSV schema
      "group_keys": ["User","Card"],            # optional, default from saved meta
      "pad_policy": "none"|"repeat"|"zero",     # optional
      "weights": {"lstm":0.5,"xgb":0.5},        # optional
      "threshold": 0.472982,                    # optional => returns 'label'
      "return_components": true,                # optional => include p_lstm / p_xgb
      "id_field": "TransactionID"               # optional (not used in anchors by default)
    }
    """
    try:
        payload = request.get_json(force=True)
        if not payload or "records" not in payload:
            return jsonify({"error": "Missing 'records' in JSON"}), 400

        df = pd.DataFrame(payload["records"])
        if df.empty:
            return jsonify({"predictions": [], "note": "No rows received"}), 200

        # Preprocess to features
        X = preprocess_rows_to_features(df)

        # Build sequences
        group_keys = payload.get("group_keys") or COLS.get("group_keys", ["User","Card"])
        pad_policy = payload.get("pad_policy", "none")
        seq_pack = build_sequences(X, df, group_keys, SEQ_LEN, pad_policy=pad_policy)
        seqs, anchors = seq_pack["sequences"], seq_pack["anchors"]
        if seqs.shape[0] == 0:
            return jsonify({"predictions": [], "note": "Not enough rows per group to form sequences"}), 200

        # Ensemble predict
        w = payload.get("weights") or ENS_WEIGHTS
        p = ensemble_predict(seqs, float(w.get("lstm", 0.5)), float(w.get("xgb", 0.5)))
        thr = payload.get("threshold", None)
        labels = (p["ens"] >= float(thr)).astype(int).tolist() if thr is not None else [None]*len(p["ens"])

        # Build response items, one per sequence
        return jsonify({
            "predictions": [
                {
                    "p_ensemble": float(p["ens"][i]),
                    "label": labels[i],
                    "p_lstm":  float(p["lstm"][i]) if payload.get("return_components") else None,
                    "p_xgb":   float(p["xgb"][i])  if payload.get("return_components") else None,
                    "group":   anchors[i]["group"],
                    "end_index": anchors[i]["end_index"],
                    "end_time":  anchors[i]["end_time"],
                    "end_id":    anchors[i]["end_id"],
                } for i in range(len(anchors))
            ],
            "sequence_length": SEQ_LEN
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Stream processor for real time detection
@app.post("/predict_stream")
def predict_stream():
    """
    Request JSON:
    {
      "record": { ...one row with CSV schema... },
      "group_keys": ["User","Card"],            # optional, default from saved meta
      "weights": {"lstm":0.5,"xgb":0.5},        # optional
      "threshold": 0.472982,                    # optional
      "return_components": true                 # optional
    }
    """
    try:
        payload = request.get_json(force=True)
        if not payload or "record" not in payload:
            return jsonify({"error": "Missing 'record'"}), 400

        rec = payload["record"]
        group_keys = payload.get("group_keys") or COLS.get("group_keys", ["User","Card"])
        key = tuple(rec.get(k) for k in group_keys)

        df = pd.DataFrame([rec])
        X = preprocess_rows_to_features(df)   # (1, NUM_FEATS)

        # Append to rolling buffer
        BUFFERS[key].append(X[0])

        have = len(BUFFERS[key])
        need = SEQ_LEN - have
        if need > 0:
            return jsonify({
                "status": "buffering",
                "group": dict(zip(group_keys, key)),
                "have": have,
                "need": need
            }), 200

        # Form a single sequence from the buffer (oldest->newest)
        seq = np.stack(list(BUFFERS[key]), axis=0)[None, :, :]  # (1, seq_len, NUM_FEATS)

        w = payload.get("weights") or ENS_WEIGHTS
        p = ensemble_predict(seq, float(w.get("lstm", 0.5)), float(w.get("xgb", 0.5)))
        thr = payload.get("threshold", None)
        label = int(p["ens"][0] >= float(thr)) if thr is not None else None

        return jsonify({
            "status": "ok",
            "group": dict(zip(group_keys, key)),
            "p_ensemble": float(p["ens"][0]),
            "label": label,
            "p_lstm": float(p["lstm"][0]) if payload.get("return_components") else None,
            "p_xgb":  float(p["xgb"][0])  if payload.get("return_components") else None
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    # Run on port 8000 by default
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))