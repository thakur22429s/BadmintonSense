"""Step 7: Full evaluation — metrics, confusion matrices, training curves, t-SNE."""

import sys
sys.path.insert(0, ".")

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data.download import load_config
from src.data.dataset import load_processed_data, get_stratified_split, StrokeDataset
from src.models.lstm import build_lstm
from src.models.transformer import build_transformer
from src.models.utils import get_device
from src.training.evaluator import evaluate_model, print_results
from src.utils.visualization import (
    plot_confusion_matrix,
    plot_training_curves,
    plot_per_class_f1,
    plot_tsne_embeddings,
)


def extract_embeddings(model, dataloader, device) -> tuple[np.ndarray, np.ndarray]:
    """Extract embeddings from model for t-SNE."""
    model.eval()
    all_embeddings = []
    all_labels = []

    with torch.no_grad():
        for batch_x, batch_y in dataloader:
            batch_x = batch_x.to(device)
            if hasattr(model, "get_embeddings"):
                emb = model.get_embeddings(batch_x)
            else:
                # For LSTM, use the hidden state before FC
                lstm_out, (h_n, _) = model.lstm(batch_x)
                if model.bidirectional:
                    emb = torch.cat([h_n[-2], h_n[-1]], dim=1)
                else:
                    emb = h_n[-1]
            all_embeddings.append(emb.cpu().numpy())
            all_labels.extend(batch_y.numpy())

    return np.concatenate(all_embeddings), np.array(all_labels)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["lstm", "transformer", "both"], default="both")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint")
    args = parser.parse_args()

    config = load_config()
    device = get_device()
    figures_dir = Path(config["paths"]["figures"])

    sequences, labels, metadata = load_processed_data(config)
    split = get_stratified_split(
        metadata, sequences, labels,
        tuple(config["data"]["stratified_ratios"]),
        config["data"]["random_seed"],
    )

    class_names = config["classes"]["names"]
    models_to_eval = ["lstm", "transformer"] if args.model == "both" else [args.model]

    for model_type in models_to_eval:
        print(f"\n{'='*60}")
        print(f"Evaluating: {model_type.upper()}")
        print(f"{'='*60}")

        # Build model and load checkpoint
        if model_type == "lstm":
            model = build_lstm(config)
            batch_size = config["training"]["lstm"]["batch_size"]
        else:
            model = build_transformer(config)
            batch_size = config["training"]["transformer"]["batch_size"]

        checkpoint_path = (
            Path(args.checkpoint) if args.checkpoint
            else Path(config["paths"]["models"]) / f"best_{model_type}_stratified.pt"
        )

        if not checkpoint_path.exists():
            print(f"  No checkpoint found at {checkpoint_path}, skipping.")
            continue

        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)

        # Evaluate on test set
        test_dataset = StrokeDataset(
            sequences[split["test_indices"]],
            labels[split["test_indices"]],
        )
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        results = evaluate_model(model, test_loader, device, class_names)
        print_results(results, class_names)

        # Confusion matrix
        plot_confusion_matrix(
            results["confusion_matrix"],
            class_names,
            figures_dir / f"confusion_matrix_{model_type}.png",
            title=f"Confusion Matrix ({model_type.upper()})",
        )

        # Per-class F1
        plot_per_class_f1(
            results["per_class_f1"],
            class_names,
            figures_dir / f"per_class_f1_{model_type}.png",
            title=f"Per-class F1 ({model_type.upper()})",
        )

        # Training curves (if history saved in checkpoint)
        if "history" in checkpoint:
            plot_training_curves(
                checkpoint["history"],
                figures_dir / f"training_curves_{model_type}.png",
                title=f"Training Curves ({model_type.upper()})",
            )

        # t-SNE
        if config["evaluation"]["save_tsne"]:
            print("  Generating t-SNE embeddings...")
            embeddings, emb_labels = extract_embeddings(model, test_loader, device)
            plot_tsne_embeddings(
                embeddings,
                emb_labels,
                class_names,
                figures_dir / f"tsne_{model_type}.png",
                perplexity=config["evaluation"]["tsne_perplexity"],
            )

    print(f"\nAll figures saved to: {figures_dir}")


if __name__ == "__main__":
    main()
