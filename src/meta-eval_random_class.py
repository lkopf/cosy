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
                    choices = ["imagenet_train", "imagenet_val", "ade20k_train", "ade20k_val", "coco_train", "coco_val"],
                    help="""Which dataset to use for probing and evaluation.""")
parser.add_argument("--method", type=str, default="INVERT", 
                    help="""Which explanation method to use for analysis.""")
parser.add_argument("--data_type", type=str, default="val",
                    choices = ["val", "probe"])
parser.add_argument("--device", type=str, default="cuda", help="whether to use GPU/which gpu")
parser.add_argument("--batch_size_eval", type=int, default=10, help="Batch size when running evaluation")
parser.add_argument("--num_workers", type=int, default=2, help="Number of workers for dataloader")
parser.add_argument("--activation_dir", type=str, default="activations", help="where to save activations")
parser.add_argument("--result_dir", type=str, default="results", help="where to save results")
parser.add_argument("--gen_images_dir", type=str, default="gen_images", help="where to save generated images")


parser.parse_args()

if __name__ == '__main__':
    args = parser.parse_args()

    PATH = os.getcwd()
    os.makedirs(os.path.join(PATH, args.result_dir), exist_ok=True)
    RESULT_PATH = f"{os.path.join(PATH, args.result_dir)}/meta-eval/random_class/"
    os.makedirs(RESULT_PATH, exist_ok=True)

    layer_name = args.target_layer.split(".")[-1]
    model_layer = f"{args.target_model}-{layer_name}"
    target_model, preprocess = get_target_model(args.target_model, args.device)
    n_neurons = get_n_neurons(model_layer)
    
    print(f"Target: {model_layer}")

    # defined class labels for analysis
    n_class_names = ['leatherback turtle', 'beer bottle', 'china cabinet', 'hard disc', 'bulbul', 'english setter', 'cardigan', 'submarine', 'coffee mug', 'switch', 'egyptian cat']
    # load imagenet labels
    imgnt_labels_num, imgnt_labels_str = get_imgnt_labels("./assets/ILSVRC2012_label_description.json")
    n_classes = [imgnt_labels_str.index(i) for i in n_class_names] 
    with open("./assets/ImageNet_1k_map.json") as json_file:
        imgnt_map = json.load(json_file)

    # Load activations for val dataset
    ACTIVATION_PATH = os.path.join(PATH, args.activation_dir)
    A_F_val = torch.load(f"{ACTIVATION_PATH}/{args.data_type}_{model_layer}.pt").to(args.device)

    # Load ImageNet labels
    df_imgnt = pd.read_csv("./assets/ILSVRC2012_val_labels.csv")

    csv_filename = f"{RESULT_PATH}meta-eval_random_{model_layer}_natural-synthetic.csv"
    csv_headers = ["neuron", "concept", "random natural concept", "random synthetic concept", "auc natural", "auc synthetic", "auc random-natural", "auc random-synthetic"]

    if not csv_file_exists(csv_filename):
        create_csv(csv_filename, csv_headers)

    for NEURON_ID, CONCEPT_NAME in tqdm(zip(n_classes, n_class_names), total=len(n_classes), desc="Processing"):
        NEURON_ID = int(NEURON_ID)

        # Load activations for synthetic concept
        concept_raw = CONCEPT_NAME
        concept = concept_raw.replace(" ", "_")
        CONCEPT_PATH = f"{args.gen_images_dir}{concept}/"

        dataset = ImageDataset(root=CONCEPT_PATH,
                                transform=TRANSFORMS_IMGNT,
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
                                  tensor_path=TENSOR_PATH,
                                  dataset=dataset, dataloader=dataloader,
                                  n_neurons=n_neurons,  device=args.device)

        ############################################################################################################

        # Load activations for random synthetic concept
        all_images = os.listdir(args.gen_images_dir)
        non_concept_images = [image for image in all_images if concept not in image]
        random_concept_syn = random.choice(non_concept_images)
        RANDOM_CONCEPT_PATH = f"{args.gen_images_dir}{random_concept_syn}/"

        dataset_rnd = ImageDataset(root=RANDOM_CONCEPT_PATH,
                                transform=TRANSFORMS_IMGNT,
                                )

        dataloader_rnd = torch.utils.data.DataLoader(dataset_rnd,
                                                batch_size=args.batch_size_eval,
                                                shuffle=False,
                                                num_workers=args.num_workers)

        RANDOM_SYN_TENSOR_PATH = f"{RESULT_PATH}{model_layer}_random_neuron-{NEURON_ID}.pt"

        if os.path.exists(RANDOM_SYN_TENSOR_PATH):
            A_F_rnd_syn = torch.load(RANDOM_SYN_TENSOR_PATH).to(args.device)
        else:
            A_F_rnd_syn = get_activations(model=target_model, model_name=model_layer,
                                  tensor_path=RANDOM_SYN_TENSOR_PATH,
                                  dataset=dataset_rnd, dataloader=dataloader_rnd,
                                  n_neurons=n_neurons,  device=args.device)

        ############################################################################################################

        # Load activations for random natural concept
        data_path = get_data_path(args.dataset)
        all_natural_images = list(imgnt_map.keys())
        imgnt_concept = list(imgnt_map.keys())[list(imgnt_map.values()).index(concept_raw)]
        non_natural_concept_images = [image for image in all_natural_images if imgnt_concept not in image]
        random_natural_concept = random.choice(non_natural_concept_images)
        random_natural_concept_name = imgnt_map[random_natural_concept]

        dataset_rnd_nat = ConceptDataset(root=data_path,
                                        concept=random_natural_concept,
                                        transform=TRANSFORMS_IMGNT,
                                        )

        dataloader_rnd_nat = torch.utils.data.DataLoader(dataset_rnd_nat,
                                                batch_size=args.batch_size_eval,
                                                shuffle=False,
                                                num_workers=args.num_workers)


        RANDOM_NAT_TENSOR_PATH = f"{RESULT_PATH}{model_layer}_random_natural_neuron-{NEURON_ID}.pt"

        if os.path.exists(RANDOM_NAT_TENSOR_PATH):
            A_F_rnd_nat = torch.load(RANDOM_NAT_TENSOR_PATH).to(args.device)
        else:
            A_F_rnd_nat = get_activations(model=target_model, model_name=model_layer,
                                  tensor_path=RANDOM_NAT_TENSOR_PATH,
                                  dataset=dataset_rnd_nat, dataloader=dataloader_rnd_nat,
                                  n_neurons=n_neurons,  device=args.device)

        ############################################################################################################

        # GET AUC
        concept_id = imgnt_labels_num[NEURON_ID]
        # filter for images including classes
        df_class = df_imgnt.loc[df_imgnt[concept_id] == 1]
        # filter for images not including classes
        df_non_class = df_imgnt.loc[df_imgnt[concept_id] == 0]

        # get AUC for neuron n for natural classes c
        class_labels = torch.zeros([len(df_imgnt)]).long()
        class_labels[df_class.index] = 1
        activ = A_F_val[:,NEURON_ID] # all activations for NEURON_ID
        auc_natural = roc_auc_score(class_labels.to("cpu"), activ.to("cpu"))

        # get AUC for neuron n for synthetic class c
        activ_non_class = activ[df_non_class.index] # [49950] without natural class
        # add synthetic class activations
        A_F_synthetic = torch.cat((activ_non_class, A_F[:,NEURON_ID]), 0) # [50000] with 50 synthetic class images
        class_labels_synthetic = torch.cat((torch.zeros([activ_non_class.shape[0]]), torch.ones([A_F[:,NEURON_ID].shape[0]])), 0) # [50000]
        auc_synthetic = roc_auc_score(class_labels_synthetic.to("cpu"), A_F_synthetic.to("cpu"))

        # get AUC for neuron n for random synthetic class c
        A_F_random_syn = torch.cat((activ_non_class, A_F_rnd_syn[:,NEURON_ID]), 0) # [50000] with 50 random synthetic class images
        class_labels_random_syn = torch.cat((torch.zeros([activ_non_class.shape[0]]), torch.ones([A_F_rnd_syn[:,NEURON_ID].shape[0]])), 0) # [50000]
        auc_random_synthetic = roc_auc_score(class_labels_random_syn.to("cpu"), A_F_random_syn.to("cpu"))

        # get AUC for neuron n for random natural class c
        A_F_random_nat = torch.cat((activ_non_class, A_F_rnd_nat[:,NEURON_ID]), 0) # [50000] with 50 random natural class images
        class_labels_random_nat = torch.cat((torch.zeros([activ_non_class.shape[0]]), torch.ones([A_F_rnd_nat[:,NEURON_ID].shape[0]])), 0) # [50000]
        auc_random_natural = roc_auc_score(class_labels_random_nat.to("cpu"), A_F_random_nat.to("cpu"))

        new_rows = [[NEURON_ID, concept_raw, random_natural_concept_name, random_concept_syn, auc_natural, auc_synthetic, auc_random_natural, auc_random_synthetic]]
        add_rows_to_csv(csv_filename, new_rows)

end = datetime.now()
print("END: ", end)
print(f"TOTAL TIME: {end - start}")

print("Done!")