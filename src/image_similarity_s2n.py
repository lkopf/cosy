import os
import json
import torch
import random
import numpy as np
from PIL import Image
from transformers import AutoProcessor, CLIPModel
from utils import *
from datetime import datetime

torch.cuda.empty_cache()

random.seed(42)

start = datetime.now()
print("START: ", start)

ROOT_PATH = "/root/path/"

MODEL_NAME = (# "Stable-Cascade"
              "SDXL1"
            )
PROMPTS = ["a", "painting_of", "photo", "realistic_photo", "realistic_photo_close"]
RESULT_PATH = "./results/prompt_comparison/"
os.makedirs(RESULT_PATH, exist_ok=True)
DISTANCE_PATH = RESULT_PATH + "distance_tensors/s2n/"
os.makedirs(DISTANCE_PATH, exist_ok=True)
IMAGE_PATH = (ROOT_PATH + 
              # "coval/generated_images/prompt_comparison/gen_images_stable_cascade/"
              "coval/generated_images/prompt_comparison/gen_images_sdxl_base/"
              )
IMAGENET_PATH = ROOT_PATH + "ImageNet_1k/train_50k/val/"

# Load ImageNet classes
CLASSES = [
    "leatherback_turtle",
    "beer_bottle",
    "china_cabinet",
    "hard_disc",
    "bulbul",
    "english_setter",
    "cardigan",
    "submarine",
    "coffee_mug",
    "switch",
    "egyptian_cat",
]

# Load imagenet label ids with corresponding strings
with open("./assets/ImageNet_1k_map.json", "r") as f:
    imgnt_map = json.load(f)

name2id = {val: key for key, val in imgnt_map.items()}

# Load the CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)

# Create results CSV file
csv_filename = RESULT_PATH + f"S2N_distance_{MODEL_NAME}.csv"
csv_headers = ["prompt", "CS-mean", "CS-std", "ED-mean", "ED-std"]

# Check if the CSV file exists
if not csv_file_exists(csv_filename):
    # If it doesn't exist, create a new CSV file with headers
    create_csv(csv_filename, csv_headers)

print("Calculating distances...")

# Iterate over the prompts
for prompt in PROMPTS:
    print(prompt)
    # Initialize lists to store average and standard deviation
    cosine_similarity_avg_list = []
    cosine_similarity_std_list = []
    euclidean_distance_avg_list = []
    euclidean_distance_std_list = []

    # Iterate over the classes
    for class_name in CLASSES:
        print(class_name)
        imgnt_name = name2id[class_name.replace("_", " ")]
        syn_class_path = os.path.join(IMAGE_PATH, class_name, prompt)
        nat_class_path = os.path.join(IMAGENET_PATH, imgnt_name)
        tensor_path = os.path.join(DISTANCE_PATH, f"concept_distances_{prompt}_{MODEL_NAME}_{class_name}.pt")

        if os.path.exists(tensor_path):
            concept_distances = torch.load(tensor_path)
        else:
            # Load and preprocess the images using a generator
            images_syn = (processor(images=Image.open(os.path.join(syn_class_path, image_name)), return_tensors="pt").to(device) for image_name in os.listdir(syn_class_path))
            embeddings_syn = torch.cat([model.get_image_features(**image) for image in images_syn])

            images_nat = (processor(images=Image.open(os.path.join(nat_class_path, image_name)), return_tensors="pt").to(device) for image_name in os.listdir(nat_class_path))
            embeddings_nat = torch.cat([model.get_image_features(**image) for image in images_nat])

            # Calculate pairwise cosine similarity
            cosine_similarity = torch.nn.functional.cosine_similarity(embeddings_syn.unsqueeze(1), embeddings_nat.unsqueeze(0), dim=2)

            # Calculate pairwise Euclidean distance
            euclidean_distance = torch.cdist(embeddings_syn, embeddings_nat, p=2)

            # Store distances to file
            concept_distances = torch.stack([cosine_similarity, euclidean_distance])
            torch.save(concept_distances, tensor_path)

        # Calculate average and standard deviation
        cosine_similarity_avg = torch.mean(concept_distances[0])
        cosine_similarity_std = torch.std(concept_distances[0])
        euclidean_distance_avg = torch.mean(concept_distances[1])
        euclidean_distance_std = torch.std(concept_distances[1])

        # Append to lists
        cosine_similarity_avg_list.append(cosine_similarity_avg.item())
        cosine_similarity_std_list.append(cosine_similarity_std.item())
        euclidean_distance_avg_list.append(euclidean_distance_avg.item())
        euclidean_distance_std_list.append(euclidean_distance_std.item())

    # Calculate overall average and standard deviation
    cs_avg = torch.mean(torch.tensor(cosine_similarity_avg_list))
    cs_std = torch.mean(torch.tensor(cosine_similarity_std_list))
    ed_avg = torch.mean(torch.tensor(euclidean_distance_avg_list))
    ed_std = torch.mean(torch.tensor(euclidean_distance_std_list))

    # Data to add to the CSV file
    new_rows = [[prompt, np.round(cs_avg.item(), 2), np.round(cs_std.item(), 2), np.round(ed_avg.item(), 2), np.round(ed_std.item(), 2)]]
    # Add new rows to the CSV file
    add_rows_to_csv(csv_filename, new_rows)

end = datetime.now()
print("END: ", end)
print(f"TOTAL TIME: {end - start}")

print("Done!")
