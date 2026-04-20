"""Step 6: Train LSTM and Transformer models."""

import sys
sys.path.insert(0, ".")

import argparse
from pathlib import Path

import numpy as np
import torch

from src.data.download import load_config
from src.data.dataset import load_processed_data, get_loso_splits, get_stratified_split, create_dataloaders
from src.models.lstm import build_lstm
from src.models.transformer import build_transformer
from src.models.utils import count_parameters, get_device
from src.training.trainer import Trainer
from src.training.evaluator import evaluate_model, print_results


def train_single_fold(
    model_type: str,
    train_indices: np.ndarray,
    val_indices: np.ndarray,
    sequences: np.ndarray,
    labels: np.ndarray,
    config: dict,
    device: torch.device,
    fold_id: str,
) -> dict:
    """Train a model on a single fold."""
    # Build model
    if model_type == "lstm":
        model = build_lstm(config)
        batch_size = config["training"]["lstm"]["batch_size"]
    else:
        model = build_transformer(config)
        batch_size = config["training"]["transformer"]["batch_size"]

    print(f"  Model parameters: {count_parameters(model):,}")

    # Create dataloaders
    train_loader, val_loader = create_dataloaders(
        sequences, labels, train_indices, val_indices, config, batch_size
    )

    # Train
    trainer = Trainer(model, config, model_type=model_type, device=device)
    history = trainer.train(train_loader, val_loader, fold_id=fold_id)

    # Final evaluation
    results = evaluate_model(model, val_loader, device, config["classes"]["names"])

    return {"history": history, "results": results, "model": model}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["lstm", "transformer", "both"], default="both")
    parser.add_argument("--split", choices=["loso", "stratified"], default="stratified")
    parser.add_argument("--max-folds", type=int, default=5, help="Max LOSO folds to run")
    args = parser.parse_args()

    config = load_config()
    device = get_device()
    print(f"Device: {device}")

    sequences, labels, metadata = load_processed_data(config)
    print(f"Data: {sequences.shape[0]} sequences, shape {sequences.shape}")
    print(f"Classes: {np.bincount(labels)}")

    models_to_train = ["lstm", "transformer"] if args.model == "both" else [args.model]

    for model_type in models_to_train:
        print(f"\n{'='*60}")
        print(f"Training: {model_type.upper()}")
        print(f"{'='*60}")

        if args.split == "loso":
            folds = get_loso_splits(metadata, sequences, labels)
            folds = folds[:args.max_folds]
            print(f"Running {len(folds)} LOSO folds")

            all_f1s = []
            for i, fold in enumerate(folds):
                print(f"\n--- Fold {i+1}/{len(folds)} (val match: {fold['fold_match_id']}) ---")
                result = train_single_fold(
                    model_type, fold["train_indices"], fold["val_indices"],
                    sequences, labels, config, device, fold_id=f"fold{i}"
                )
                all_f1s.append(result["results"]["macro_f1"])
                print(f"  Fold F1: {result['results']['macro_f1']:.4f}")

            print(f"\n{'='*60}")
            print(f"{model_type.upper()} LOSO Results:")
            print(f"  Mean Macro-F1: {np.mean(all_f1s):.4f} ± {np.std(all_f1s):.4f}")
            print(f"  Per-fold: {[f'{f:.4f}' for f in all_f1s]}")

        else:
            split = get_stratified_split(
                metadata, sequences, labels,
                tuple(config["data"]["stratified_ratios"]),
                config["data"]["random_seed"],
            )
            print(f"Stratified split: train={len(split['train_indices'])}, "
                  f"val={len(split['val_indices'])}, test={len(split['test_indices'])}")

            result = train_single_fold(
                model_type, split["train_indices"], split["val_indices"],
                sequences, labels, config, device, fold_id="stratified"
            )
            print_results(result["results"], config["classes"]["names"])


if __name__ == "__main__":
    main()
