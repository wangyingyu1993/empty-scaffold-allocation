#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from empty_scaffold.metrics import classification_summary, regression_summary
from empty_scaffold.modeling import featurize_table


def build_model(input_dim: int, task: str):
    import tensorflow as tf

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(input_dim,)),
        tf.keras.layers.Dense(512, activation="relu"),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(256, activation="relu"),
        tf.keras.layers.Dense(30, activation="relu"),
        tf.keras.layers.Dense(1, activation="sigmoid" if task == "classification" else "linear"),
    ])
    if task == "classification":
        model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="binary_crossentropy")
    else:
        model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="mse")
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a compact FCNN Morgan-fingerprint baseline on provided split CSV files.")
    parser.add_argument("--train", required=True)
    parser.add_argument("--test", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--task", choices=["regression", "classification"], required=True)
    parser.add_argument("--valid")
    parser.add_argument("--smiles-column", default="molecules")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    import tensorflow as tf
    tf.keras.utils.set_random_seed(args.seed)

    train_df = pd.read_csv(args.train)
    test_df = pd.read_csv(args.test)
    valid_df = pd.read_csv(args.valid) if args.valid else None
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    x_train, y_train, _ = featurize_table(train_df, args.target, smiles_col=args.smiles_column, task=args.task)
    x_test, y_test, test_rows = featurize_table(test_df, args.target, smiles_col=args.smiles_column, task=args.task)

    validation_data = None
    if valid_df is not None:
        x_valid, y_valid, _ = featurize_table(valid_df, args.target, smiles_col=args.smiles_column, task=args.task)
    else:
        x_valid = y_valid = None

    y_scaler = None
    if args.task == "regression":
        y_scaler = StandardScaler().fit(y_train.reshape(-1, 1))
        y_train_fit = y_scaler.transform(y_train.reshape(-1, 1)).ravel()
        validation_data = (x_valid, y_scaler.transform(y_valid.reshape(-1, 1)).ravel()) if valid_df is not None else None
    else:
        y_train_fit = y_train.astype(float)
        validation_data = (x_valid, y_valid.astype(float)) if valid_df is not None else None

    model = build_model(x_train.shape[1], args.task)
    callbacks = [tf.keras.callbacks.EarlyStopping(monitor="val_loss" if validation_data else "loss", patience=8, restore_best_weights=True)]
    model.fit(
        x_train,
        y_train_fit,
        validation_data=validation_data,
        epochs=args.epochs,
        batch_size=args.batch_size,
        verbose=0,
        callbacks=callbacks,
    )

    raw_pred = model.predict(x_test, verbose=0).reshape(-1)
    if args.task == "regression":
        pred = y_scaler.inverse_transform(raw_pred.reshape(-1, 1)).ravel()
        metrics = regression_summary(y_test, pred)
        out = pd.DataFrame({args.smiles_column: test_rows[args.smiles_column].to_numpy(), args.target: y_test, f"pred_{args.target}": pred})
    else:
        prob = raw_pred
        pred = (prob >= 0.5).astype(int)
        metrics = classification_summary(y_test, prob, pred)
        out = pd.DataFrame({args.smiles_column: test_rows[args.smiles_column].to_numpy(), args.target: y_test, f"prob_{args.target}": prob, f"pred_{args.target}": pred})

    out.to_csv(out_dir / "test_predictions.csv", index=False)
    pd.DataFrame([{**metrics, "target": args.target, "model": "fcnn_fingerprint"}]).to_csv(out_dir / "test_metrics.csv", index=False)
    model.save(out_dir / "keras_model", include_optimizer=False)
    print(f"[OK] wrote {out_dir}")


if __name__ == "__main__":
    main()
