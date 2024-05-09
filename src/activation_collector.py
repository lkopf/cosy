import os
import torch
import argparse

from utils import *

torch.cuda.empty_cache()

parser = argparse.ArgumentParser(description='Activation-Collector')

parser.add_argument("--target_model", type=str, default="resnet18", 
                   help=""""Which model to analyze, supported options are pretrained imagenet models from
                        torchvision and vision models from huggingface""")
parser.add_argument("--target_layer", type=str, default="fc",
                    help="""Which layer neurons to describe. String list of layer names to describe, separated by comma(no spaces). 
                          Follows the naming scheme of the Pytorch module used""")
parser.add_argument("--dataset", type=str, default="imagenet_val", 
                    choices = ["imagenet_train", "imagenet_val", "ade20k_train", "ade20k_val", "coco_train", "coco_val"],
                    help="""Which dataset to use for probing and evaluation.""")
parser.add_argument("--data_type", type=str, default="val",
                    choices = ["val", "probe"])
parser.add_argument("--device", type=str, default="cuda", help="whether to use GPU/which gpu")
parser.add_argument("--batch_size_activ", type=int, default=256, help="Batch size when running activation collector")
parser.add_argument("--num_workers", type=int, default=2, help="Number of workers for dataloader")
parser.add_argument("--activation_dir", type=str, default="activations", help="where to save activations")


parser.parse_args()

if __name__ == '__main__':
    args = parser.parse_args()

    PATH = os.getcwd()
    ACTIVATION_PATH = os.path.join(PATH, args.activation_dir)
    os.makedirs(ACTIVATION_PATH, exist_ok=True)

    layer_name = args.target_layer.split(".")[-1]
    model_layer = f"{args.target_model}-{layer_name}"
    target_model, preprocess = get_target_model(args.target_model, args.device)
    n_neurons = get_n_neurons(model_layer)
    
    print(f"Target: {model_layer}")

    data_path = get_data_path(args.dataset)

    dataset = ImageDataset(root=data_path,
                            transform=TRANSFORMS_IMGNT,
                            )

    dataloader = torch.utils.data.DataLoader(dataset,
                                            batch_size=args.batch_size_activ,
                                            shuffle=False,
                                            num_workers=args.num_workers,
                                            )

    print("Collect activations...")

    TENSOR_PATH = f"{ACTIVATION_PATH}/{args.data_type}_{model_layer}.pt"

    A_F = get_activations(model=target_model, model_name=model_layer,
                            tensor_path=TENSOR_PATH,
                            dataset=dataset, dataloader=dataloader,
                            n_neurons=n_neurons, device=args.device)

    print("Done!")
