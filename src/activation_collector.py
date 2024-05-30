"""
This script collects and saves the activations of a target layer in a target model.

Usage:
python src/activation_collector.py [--target_model MODEL] [--target_layer LAYER]
                                   [--dataset DATASET] [--device DEVICE]
                                   [--batch_size_activ BATCH_SIZE]
                                   [--num_workers NUM_WORKERS]
                                   [--activation_dir ACTIVATION_DIR]

Arguments:
--target_model: Which model to analyze, supported options are pretrained pytorch models.
--target_layer: Which layer neurons to describe for pytorch models.
--dataset: Which dataset to use for evaluation.
--device: Whether to use GPU.
--batch_size_activ: Batch size when running activation collector.
--num_workers: Number of workers for dataloader.
--activation_dir: Where to save activations.
"""

import argparse
import os
import torch

import utils

torch.cuda.empty_cache()

torch.manual_seed(42)

parser = argparse.ArgumentParser(description="Activation-Collector")

parser.add_argument(
    "--target_model",
    type=str,
    default="resnet18",
    help=""""Which model to analyze, supported options are pretrained pytorch models.""",
)
parser.add_argument(
    "--target_layer",
    type=str,
    default="avgpool",
    help="""Which layer neurons to describe for pytorch models.""",
)
parser.add_argument(
    "--dataset",
    type=str,
    default="imagenet",
    help="""Which dataset to use for evaluation.""",
)
parser.add_argument("--device", type=str, default="cuda", help="Whether to use GPU.")
parser.add_argument(
    "--batch_size_activ",
    type=int,
    default=256,
    help="Batch size when running activation collector.",
)
parser.add_argument(
    "--num_workers", type=int, default=2, help="Number of workers for dataloader."
)
parser.add_argument(
    "--activation_dir",
    type=str,
    default="activations",
    help="Where to save activations.",
)


parser.parse_args()

if __name__ == "__main__":
    args = parser.parse_args()

    os.makedirs(args.activation_dir, exist_ok=True)

    layer_name = args.target_layer.split(".")[-1]
    model_layer = f"{args.target_model}-{layer_name}"
    target_model, features_layer, preprocess = utils.get_target_model(
        args.target_model, args.device
    )
    n_neurons = utils.get_n_neurons(model_layer)

    print(f"Target: {model_layer}")

    dataset = utils.get_dataset(args.dataset)

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size_activ,
        shuffle=False,
        num_workers=args.num_workers,
    )

    print("Collect activations...")

    TENSOR_PATH = f"{args.activation_dir}/val_{model_layer}.pt"

    A_F = utils.get_activations(
        model=target_model,
        model_name=model_layer,
        tensor_path=TENSOR_PATH,
        dataset=dataset,
        dataloader=dataloader,
        n_neurons=n_neurons,
        device=args.device,
        preprocess=preprocess,
    )

    print("Done!")
