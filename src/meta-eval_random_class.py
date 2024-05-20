"""
Sanity Check
"""

import os
import json
import torch
import random
import argparse
import pandas as pd
from tqdm import tqdm
from datetime import datetime
from sklearn.metrics import roc_auc_score
from scipy.stats import median_test
from sklearn.preprocessing import minmax_scale

from utils import *

torch.cuda.empty_cache()

random.seed(42)

start = datetime.now()
print("START: ", start)

parser = argparse.ArgumentParser(description='Meta-Evaluation-Classes')

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

    RESULT_PATH = f"{args.result_dir}/meta-eval/random_method_class/"
    os.makedirs(RESULT_PATH, exist_ok=True)
    RANDOM_ACTIVATIONS_PATH = os.path.join(RESULT_PATH, "random_class_activations/")
    os.makedirs(RANDOM_ACTIVATIONS_PATH, exist_ok=True)

    layer_name = args.target_layer.split(".")[-1]
    if layer_name == "encoder_layer_11":
        layer_name = "layer11"
    model_layer = f"{args.target_model}-{layer_name}"
    target_model, preprocess = get_target_model(args.target_model, args.device)
    n_neurons = get_n_neurons(model_layer)
    data_transform = get_transform(args.transform)
    
    print(f"Target: {model_layer}")

    EXPLANATION_PATH = f"./assets/explanations/{args.method}/{model_layer}.csv"
    NEURON_IDS = random.sample(range(n_neurons), args.n_neurons_random)
    EXPLANATIONS, _ = load_explanations(path=EXPLANATION_PATH,name=args.method,
                                        image_path=args.gen_images_dir,neuron_ids=NEURON_IDS)

    # Load activations for val dataset
    A_F_val = torch.load(f"{args.activation_dir}/{model_layer}.pt").to(args.device)
    # Load ImageNet labels
    df_imgnt = pd.read_csv("./assets/ILSVRC2012_val_labels.csv")

    csv_filename = f"{RESULT_PATH}random_class_psi_{args.method}_{model_layer}.csv"
    csv_headers = ["neuron", "concept", "random concept", "auc", "avg. activ.", "avg. activ. norm", "statistic", "auc random", "avg activ. rnd", "avg. activ. rnd norm", "statistic rnd"]

    if not csv_file_exists(csv_filename):
        create_csv(csv_filename, csv_headers)

    for NEURON_ID, CONCEPT_NAME in tqdm(zip(NEURON_IDS, EXPLANATIONS), total=len(NEURON_IDS), desc="Processing"):
        NEURON_ID = int(NEURON_ID)

        # Load activations for synthetic concept
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

        TENSOR_PATH = f"{RESULT_PATH}{model_layer}_neuron-{NEURON_ID}.pt"

        if os.path.exists(TENSOR_PATH):
            A_F = torch.load(TENSOR_PATH).to(args.device)
        else:
            A_F = get_activations(model=target_model, model_name=model_layer,
                                  tensor_path=TENSOR_PATH, dataset=dataset,
                                  dataloader=dataloader, n_neurons=n_neurons,
                                  device=args.device, preprocess=preprocess)

        ############################################################################################################

        # Load activations for random synthetic concept
        all_images = os.listdir(args.gen_images_dir)
        non_concept_images = [image for image in all_images if concept not in image]
        random_concept_syn = random.choice(non_concept_images)
        RANDOM_CONCEPT_PATH = f"{args.gen_images_dir}{random_concept_syn}/"

        dataset_rnd = ImageDataset(root=RANDOM_CONCEPT_PATH,
                                transform=data_transform,
                                )

        dataloader_rnd = torch.utils.data.DataLoader(dataset_rnd,
                                                batch_size=args.batch_size_eval,
                                                shuffle=False,
                                                num_workers=args.num_workers)

        RANDOM_SYN_TENSOR_PATH = f"{RANDOM_ACTIVATIONS_PATH}{args.method}_{model_layer}_random_neuron-{NEURON_ID}.pt"

        if os.path.exists(RANDOM_SYN_TENSOR_PATH):
            A_F_rnd_syn = torch.load(RANDOM_SYN_TENSOR_PATH).to(args.device)
        else:
            A_F_rnd_syn = get_activations(model=target_model, model_name=model_layer,
                                  tensor_path=RANDOM_SYN_TENSOR_PATH, dataset=dataset_rnd,
                                  dataloader=dataloader_rnd, n_neurons=n_neurons,
                                  device=args.device, preprocess=preprocess)

        ############################################################################################################       
        # Get binary classification scores for neuron explanation
        activ_non_class = A_F_val[:,NEURON_ID] # all activations for val dataset
        # Construct dataset with non-class activations and synthetic class activations
        activ_class = A_F[:,NEURON_ID]
        # add synthetic class activations
        A_F_synthetic = torch.cat((activ_non_class, A_F[:,NEURON_ID]), 0) # [50000] with 50 synthetic class images
        class_labels_synthetic = torch.cat((torch.zeros([activ_non_class.shape[0]]), torch.ones([A_F[:,NEURON_ID].shape[0]])), 0) # [50000]
        # Psi synthetic true class
        auc_synthetic = roc_auc_score(class_labels_synthetic.to("cpu"), A_F_synthetic.to("cpu"))
        statistic, m_pvalue, median, table = median_test(activ_non_class.to("cpu"), activ_class.to("cpu"))
        avg_activations = activ_class.mean().item()
        avg_activations_norm = minmax_scale(activ_class.to("cpu"), axis=0).mean().item()

        # get Psi for neuron n for random synthetic class c
        A_F_random_syn = torch.cat((activ_non_class, A_F_rnd_syn[:,NEURON_ID]), 0) # [50000] with 50 random synthetic class images
        class_labels_random_syn = torch.cat((torch.zeros([activ_non_class.shape[0]]), torch.ones([A_F_rnd_syn[:,NEURON_ID].shape[0]])), 0) # [50000]
        # Psi synthetic random
        auc_random_synthetic = roc_auc_score(class_labels_random_syn.to("cpu"), A_F_random_syn.to("cpu"))
        statistic_rnd, _, _, table = median_test(activ_non_class.to("cpu"), A_F_rnd_syn[:,NEURON_ID].to("cpu"))
        avg_activations_rnd = A_F_rnd_syn[:,NEURON_ID].mean().item()
        avg_activations_rnd_norm = minmax_scale(A_F_rnd_syn[:,NEURON_ID].to("cpu"), axis=0).mean().item()

        new_rows  = [[NEURON_ID, concept_raw, random_concept_syn, auc_synthetic, avg_activations, avg_activations_norm, statistic, auc_random_synthetic, avg_activations_rnd, avg_activations_rnd_norm, statistic_rnd]]
        add_rows_to_csv(csv_filename, new_rows)

end = datetime.now()
print("END: ", end)
print(f"TOTAL TIME: {end - start}")
