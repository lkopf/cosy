"""
python src/activation_collector.py --target_model densenet161_places --target_layer features --dataset places365_train
python src/activation_collector.py --target_model vit_b_16 --target_layer subset --dataset imagenet_train
"""

import os
import csv
import torch
import argparse

from utils import *

torch.cuda.empty_cache()

torch.manual_seed(42)

parser = argparse.ArgumentParser(description='Activation-Collector')

parser.add_argument("--target_model", type=str, default="resnet18",
                   help=""""Which model to analyze, supported options are pretrained imagenet models from
                        torchvision and vision models from huggingface""")
parser.add_argument("--target_layer", type=str, default="fc",
                    help="""Which layer neurons to describe. String list of layer names to describe, separated by comma(no spaces). 
                          Follows the naming scheme of the Pytorch module used""")
parser.add_argument("--dataset", type=str, default="imagenet_val", 
                    help="""Which dataset to use for probing and evaluation.""")
parser.add_argument("--device", type=str, default="cuda", help="whether to use GPU/which gpu")
parser.add_argument("--batch_size_activ", type=int, default=256, help="Batch size when running activation collector")
parser.add_argument("--num_workers", type=int, default=2, help="Number of workers for dataloader")
parser.add_argument("--activation_dir", type=str, default="activations", help="where to save activations")


parser.parse_args()

if __name__ == '__main__':
    args = parser.parse_args()

    os.makedirs(args.activation_dir, exist_ok=True)

    layer_name = args.target_layer.split(".")[-1]
    model_layer = f"{args.target_model}-{layer_name}"
    target_model, features_layer, preprocess = get_target_model(args.target_model, args.device)
    n_neurons = get_n_neurons(model_layer)
    
    print(f"Target: {model_layer}")

    dataset = get_dataset(args.dataset)

    dataloader = torch.utils.data.DataLoader(dataset,
                                            batch_size=args.batch_size_activ,
                                            shuffle=False,
                                            num_workers=args.num_workers,
                                            )


    print("Collect activations...")

    TENSOR_PATH = f"{args.activation_dir}/val_{model_layer}.pt"

    A_F = get_activations(model=features_layer, #target_model, # features_layer,
                           model_name=model_layer,
                            tensor_path=TENSOR_PATH, dataset=dataset,
                            dataloader=dataloader, n_neurons=n_neurons,
                            device=args.device, preprocess=preprocess)

    print("Done!")
