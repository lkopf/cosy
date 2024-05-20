"""
python src/evaluation.py --target_model=resnet18 --target_layer=avgpool --method=CLIP-Dissect --gen_images_dir=/mnt/beegfs/share/atbstaff/coval/generated_images/sdxl_base/
"""

import os
import torch
import random
import argparse
from tqdm import tqdm
from datetime import datetime
from sklearn.metrics import roc_auc_score
from scipy.stats import mannwhitneyu,median_test

from utils import *

torch.cuda.empty_cache()

random.seed(42)

start = datetime.now()
print("START: ", start)

parser = argparse.ArgumentParser(description='Evaluation')

parser.add_argument("--target_model", type=str, default="resnet18", 
                   help=""""Which model to analyze, supported options are pretrained imagenet models from
                        torchvision and vision models from huggingface""")
parser.add_argument("--target_layer", type=str, default="fc",
                    help="""Which layer neurons to describe. String list of layer names to describe, separated by comma(no spaces). 
                          Follows the naming scheme of the Pytorch module used""")
parser.add_argument("--dataset", type=str, default="imagenet_val", 
                    help="""Which dataset to use for probing and evaluation.""")
parser.add_argument("--method", type=str, default="INVERT",
                    help="""Which explanation method to use for analysis.""")
parser.add_argument("--transform", type=str, default="transform_imagenet",
                    help="""Which transform to use for dataset.""")
parser.add_argument("--n_neurons_random", type=int, default=50, help="Number of random neurons in model layer")
parser.add_argument("--device", type=str, default="cuda", help="whether to use GPU/which gpu")
parser.add_argument("--batch_size_eval", type=int, default=10, help="Batch size when running evaluation")
parser.add_argument("--num_workers", type=int, default=2, help="Number of workers for dataloader")
parser.add_argument("--activation_dir", type=str, default="activations", help="where to save activations")
parser.add_argument("--result_dir", type=str, default="results", help="where to save results")
parser.add_argument("--gen_images_dir", type=str, default="gen_images", help="where to save generated images")


parser.parse_args()

if __name__ == '__main__':
    args = parser.parse_args()

    os.makedirs(args.result_dir, exist_ok=True)

    layer_name = args.target_layer.split(".")[-1]
    model_layer = f"{args.target_model}-{layer_name}"
    target_model, features_layer, preprocess = get_target_model(args.target_model, args.device)
    # densenet
    if args.target_model == "densenet161_places":
        target_model = features_layer
    
    n_neurons = get_n_neurons(model_layer)
    
    print(f"Evaluate target: {model_layer}")

    EXPLANATION_PATH = f"./assets/explanations/{args.method}/{model_layer}_filter.csv" ###
    NEURON_IDS = random.sample(range(n_neurons), args.n_neurons_random)
    EXPLANATIONS, _ = load_explanations(path=EXPLANATION_PATH,name=args.method,
                                        image_path=args.gen_images_dir,neuron_ids=NEURON_IDS)

    data_path = get_data_path(args.dataset)
    data_transform = get_transform(args.transform)

    print("Evaluate explanations...")

    # Load activations for val dataset
    A_F_val = torch.load(f"{args.activation_dir}/val_{model_layer}.pt").to(args.device)

    csv_filename = f"{args.result_dir}/evaluation_{args.method}_{model_layer}_filter.csv"
    csv_headers = ["neuron", "concept", "auc", "U1", "p", "avg. activation diff"]

    if not csv_file_exists(csv_filename):
        create_csv(csv_filename, csv_headers)

    for NEURON_ID, CONCEPT_NAME in tqdm(zip(NEURON_IDS, EXPLANATIONS), total=len(NEURON_IDS), desc="Processing"):
        NEURON_ID = int(NEURON_ID)
        concept_raw = CONCEPT_NAME
        concept = concept_raw.replace(" ", "_")
        CONCEPT_PATH = f"{args.gen_images_dir}{concept}/"

        dataset = ImageDataset(root=CONCEPT_PATH,
                                transform=data_transform,
                                )

        dataloader = torch.utils.data.DataLoader(dataset,
                                                batch_size=args.batch_size_eval,
                                                shuffle=False,
                                                num_workers=args.num_workers)
        
        TENSOR_PATH = f"{args.activation_dir}/method_eval/{args.method}_{model_layer}_neuron-{NEURON_ID}_filter.pt"

        if os.path.exists(TENSOR_PATH):
            A_F = torch.load(TENSOR_PATH)
        else:
            A_F = get_activations(model=target_model, model_name=model_layer,
                                    tensor_path=TENSOR_PATH,
                                    dataset=dataset, dataloader=dataloader,
                                    n_neurons=n_neurons,  device=args.device)

        # Get binary classification scores for neuron explanation
        activ_non_class = A_F_val[:,NEURON_ID] # all activations for val dataset
        # Construct dataset D with non-class activations and synthetic class activations
        activ_class = A_F[:,NEURON_ID]
        A_D = torch.cat((activ_non_class, activ_class), 0)
        class_labels = torch.cat((torch.zeros([activ_non_class.shape[0]]), torch.ones([activ_class.shape[0]])), 0)
        # Binary classification scores
        auc_synthetic = roc_auc_score(class_labels.to("cpu"), A_D.to("cpu"))
        U1, p = mannwhitneyu(class_labels.to("cpu"), A_D.to("cpu"))
        # statistic, m_pvalue, median, table = median_test(activ_non_class.to("cpu"), activ_class.to("cpu"))
        avg_activation_diff = activ_class.mean().item() - activ_non_class.mean().item()

        new_rows  = [[NEURON_ID, concept_raw, auc_synthetic, U1, p, avg_activation_diff]]
        add_rows_to_csv(csv_filename, new_rows)

    end = datetime.now()
    print("END: ", end)
    print(f"TOTAL TIME: {end - start}")
