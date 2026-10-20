import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

from fetch_data import fetch_data
from indicators import add_all_ind
from config import TICKER, PERIOD, TICKERS, ML_FEATURES

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
except ImportError:
    raise SystemExit(
        "TensorFlow not installed.\n"
    )

LOOKBACK = 20   # days of history fed into the LSTM as one sample


def make_sequences(X, y, lookback):
    Xs, ys = [], []
    for i in range(lookback, len(X)):
        Xs.append(X[i - lookback:i])
        ys.append(y[i])
    return np.array(Xs), np.array(ys)


def build_multi_stock_sequences(tickers, period, lookback):
    """
    For each ticker: scale features (fit on train only), build LOOKBACK-day
    sequences, take the first 80% as training sequences.
    Sequences never cross stock boundaries so there is no contamination.
    Returns combined X_train sequences and the primary stock's val/test splits.
    """
    train_Xs, train_ys = [], []
    primary_val_test   = None

    scaler = StandardScaler()

    for ticker in tickers:
        try:
            data = fetch_data(ticker, period=period)
            data = add_all_ind(data)
            data["Target"] = (data["Close"].shift(-1) > data["Close"]).astype(int)
            data = data.dropna()

            X_raw = data[ML_FEATURES].values
            y_raw = data["Target"].values

            n_total   = len(X_raw)
            train_end = int(n_total * 0.80)

            # Fit scaler on training rows only to avoid look-ahead bias
            X_scaled              = np.empty_like(X_raw, dtype=float)
            X_scaled[:train_end]  = scaler.fit_transform(X_raw[:train_end])
            X_scaled[train_end:]  = scaler.transform(X_raw[train_end:])

            X_seq, y_seq = make_sequences(X_scaled, y_raw, lookback)

            n       = len(X_seq)
            tr_end  = int(n * 0.80)
            val_end = int(n * 0.90)

            train_Xs.append(X_seq[:tr_end])
            train_ys.append(y_seq[:tr_end])

            if ticker == TICKER:
                primary_val_test = (
                    X_seq[tr_end:val_end], y_seq[tr_end:val_end],
                    X_seq[val_end:],       y_seq[val_end:],
                )
            print(f"  [{ticker}] {tr_end} training sequences")
        except Exception as e:
            print(f"  [WARN] Skipping {ticker}: {e}")

    X_train = np.concatenate(train_Xs, axis=0)
    y_train = np.concatenate(train_ys, axis=0)
    return X_train, y_train, primary_val_test


def main():
    # ── Multi-stock training sequences ─────────────────────────────────────
    print(f"Building multi-stock LSTM sequences from: {TICKERS}")
    X_train, y_train, primary = build_multi_stock_sequences(TICKERS, PERIOD, LOOKBACK)
    X_val, y_val, X_test, y_test = primary

    print(f"\nLSTM — Train: {len(X_train)}  Val: {len(X_val)}  Test: {len(X_test)}")

    # ── Model ──────────────────────────────────────────────────────────────
    tf.random.set_seed(42)

    model = keras.Sequential([
        layers.LSTM(64, input_shape=(LOOKBACK, len(ML_FEATURES)), return_sequences=True),
        layers.Dropout(0.2),
        layers.LSTM(32),
        layers.Dropout(0.2),
        layers.Dense(1, activation="sigmoid"),
    ])

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()

    early_stop = keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=5, restore_best_weights=True
    )

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=50,
        batch_size=32,
        callbacks=[early_stop],
        verbose=1,
    )

    # ── Evaluate ───────────────────────────────────────────────────────────
    y_pred = (model.predict(X_test) > 0.5).astype(int).flatten()

    print("\n=== LSTM (test set) ===")
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    # ── Training curves ────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("LSTM Training History", fontsize=13, fontweight="bold")

    axes[0].plot(history.history["loss"],     label="Train Loss")
    axes[0].plot(history.history["val_loss"], label="Val Loss")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(history.history["accuracy"],     label="Train Accuracy")
    axes[1].plot(history.history["val_accuracy"], label="Val Accuracy")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
