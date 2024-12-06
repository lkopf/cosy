"""
This script performs evaluation on a target model using different explanation methods.
It calculates the area under the ROC curve (AUC), Mann-Whitney U statistic, p-value, and
mean activation difference (MAD) for each neuron and concept.
The script saves the evaluation results in a CSV file.

Usage:
    python src/evaluation.py [--target_model TARGET_MODEL] [--target_layer TARGET_LAYER]
                             [--method METHOD] [--transform TRANSFORM]
                             [--n_neurons_random N_NEURONS_RANDOM] [--device DEVICE]
                             [--batch_size_eval BATCH_SIZE_EVAL] [--num_workers NUM_WORKERS]
                             [--activation_dir ACTIVATION_DIR] [--result_dir RESULT_DIR]
                             [--gen_images_dir GEN_IMAGES_DIR]

Arguments:
    --target_model (str): Which model to analyze, supported options are pretrained pytorch models.
    --target_layer (str): Which layer neurons to describe for pytorch models.
    --method (str): Which explanation method to use for analysis.
    --transform (str): Which transform to use for dataset.
    --n_neurons_random (int): Number of random neurons in model layer.
    --device (str): Whether to use GPU.
    --batch_size_eval (int): Batch size when running evaluation.
    --num_workers (int): Number of workers for dataloader.
    --activation_dir (str): Where to save activations.
    --result_dir (str): Where to save results.
    --gen_images_dir (str): Where to save generated images.
"""

import argparse
import os
import random
from datetime import datetime
import torch
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from scipy.stats import mannwhitneyu

import utils

torch.cuda.empty_cache()

random.seed(42)

start = datetime.now()
print("START: ", start)

parser = argparse.ArgumentParser(description="Evaluation")

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
    "--method",
    type=str,
    default="INVERT",
    help="""Which explanation method to use for analysis.""",
)
parser.add_argument(
    "--transform",
    type=str,
    default="transform_imagenet",
    help="""Which transform to use for dataset.""",
)
parser.add_argument(
    "--n_neurons_random",
    type=int,
    default=50,
    help="Number of random neurons in model layer.",
)
parser.add_argument("--device", type=str, default="cuda", help="Whether to use GPU.")
parser.add_argument(
    "--batch_size_eval",
    type=int,
    default=10,
    help="Batch size when running evaluation.",
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
parser.add_argument(
    "--result_dir", type=str, default="results", help="Where to save results."
)
parser.add_argument(
    "--gen_images_dir",
    type=str,
    default="gen_images",
    help="Where to save generated images.",
)


parser.parse_args()

if __name__ == "__main__":
    args = parser.parse_args()

    os.makedirs(args.result_dir, exist_ok=True)

    layer_name = args.target_layer.split(".")[-1]
    model_layer = f"{args.target_model}-{layer_name}"
    target_model, features_layer, preprocess = utils.get_target_model(
        args.target_model, args.device
    )
    # target_model = features_layer
    n_neurons = utils.get_n_neurons(model_layer)

    print(f"Evaluate target: {model_layer}")

    EXPLANATION_PATH = f"./assets/explanations/{args.method}/{model_layer}.csv"
    NEURON_IDS = random.sample(range(n_neurons), args.n_neurons_random)
    EXPLANATIONS, _ = utils.load_explanations(
        path=EXPLANATION_PATH,
        name=args.method,
        image_path=args.gen_images_dir,
        neuron_ids=NEURON_IDS,
    )

    data_transform = utils.get_transform(args.transform)

    print("Evaluate explanations...")

    # Load activations for control dataset
    A_0 = torch.load(f"{args.activation_dir}/val_{model_layer}.pt").to(args.device)

    csv_filename = f"{args.result_dir}/evaluation_{args.method}_{model_layer}.csv"
    csv_headers = ["neuron", "concept", "AUC", "U1", "p", "MAD"]

    if not os.path.exists(csv_filename):
        utils.create_csv(csv_filename, csv_headers)

    for NEURON_ID, CONCEPT_NAME in tqdm(
        zip(NEURON_IDS, EXPLANATIONS), total=len(NEURON_IDS), desc="Processing"
    ):
        NEURON_ID = int(NEURON_ID)
        concept_raw = CONCEPT_NAME
        concept = concept_raw.replace(" ", "_")
        CONCEPT_PATH = f"{args.gen_images_dir}{concept}/"

        dataset = utils.ImageDataset(
            root=CONCEPT_PATH,
            transform=data_transform,
        )

        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=args.batch_size_eval,
            shuffle=False,
            num_workers=args.num_workers,
        )

        # Load activations for concept dataset
        TENSOR_PATH = f"{args.activation_dir}/method_eval/{args.method}_{model_layer}_neuron-{NEURON_ID}.pt"

        if os.path.exists(TENSOR_PATH):
            A_1 = torch.load(TENSOR_PATH)
        else:
            A_1 = utils.get_activations(
                model=target_model,
                model_name=model_layer,
                tensor_path=TENSOR_PATH,
                dataset=dataset,
                dataloader=dataloader,
                n_neurons=n_neurons,
                device=args.device,
            )

        # all activations for control dataset
        activ_non_concept = A_0[:, NEURON_ID]
        # all activations for concept dataset
        activ_concept = A_1[:, NEURON_ID]
        # Construct tensor with binary labels
        concept_labels = torch.cat(
            (
                torch.zeros([activ_non_concept.shape[0]]),
                torch.ones([activ_concept.shape[0]]),
            ),
            0,
        )
        # Construct dataset A_D with non-concept activations and synthetic concept activations
        A_D = torch.cat((activ_non_concept, activ_concept), 0)
        # Score explanations
        auc_synthetic = roc_auc_score(concept_labels.to("cpu"), A_D.to("cpu"))
        U1, p = mannwhitneyu(concept_labels.to("cpu"), A_D.to("cpu"))
        if activ_non_concept.std().item() == 0:
            mad = 0.0
        else:
            # mad = activ_concept.mean().item() - activ_non_concept.mean().item()
            mad = (
                activ_concept.mean().item() - activ_non_concept.mean().item()
            ) / activ_non_concept.std().item()
        new_rows = [[NEURON_ID, concept_raw, auc_synthetic, U1, p, mad]]
        utils.add_rows_to_csv(csv_filename, new_rows)

    end = datetime.now()
    print("END: ", end)
    print(f"TOTAL TIME: {end - start}")
