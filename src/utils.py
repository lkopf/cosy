import os
import cv2
import csv
import json
import pandas as pd
import torch
import torchvision
from torch.utils.data import Dataset

class ImageDataset(Dataset):
    def __init__(self,root,transform):#,image_format):
        self.root=root
        self.transform=transform
        #self.image_format=image_format

        #self.image_names=glob(self.root + f'*.{str(image_format)}')
        self.image_names=[self.root + x for x in os.listdir(self.root)]
        self.image_names.sort()
   
    def __len__(self):
        return len(self.image_names)
 
    def __getitem__(self,index):
        image=cv2.imread(self.image_names[index])
        image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)

        image=self.transform(image)

        return image


transforms = torchvision.transforms.Compose([
                           torchvision.transforms.ToPILImage(),
                           torchvision.transforms.Resize(224),
                           torchvision.transforms.CenterCrop((224,224)),
                           torchvision.transforms.ToTensor(),
                           torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                            std=[0.229, 0.224, 0.225])
                       ])


# Function to load selected explanations as string
def load_explanations(path, name, image_path, neuron_ids): 
    df = pd.read_csv(path)
    if name == "INVERT":
        # Load imagenet label ids with corresponding strings
        with open("./assets/ImageNet_1k_map.json", "r") as f:
            imgnt_map = json.load(f)
        explanations = [] # all explanations
        for neuron_id in neuron_ids:
            explanation_raw = df["formula"][neuron_id]
            explanation_raw = str(explanation_raw).split()
            explanation_mapped = [imgnt_map[i] if i in imgnt_map else i for i in explanation_raw]
            explanation = " ".join(explanation_mapped)
            explanations.append(explanation)

    elif name == "CLIP-Dissect":
        explanations = []
        for neuron_id in neuron_ids:
            explanation = df.loc[df["unit"] == neuron_id, "description"].values[0]
            explanations.append(explanation)

    elif name == "MILAN":
        explanations = []
        for neuron_id in neuron_ids:
            explanation = df.loc[df["unit"] == neuron_id, "description"].values[0]
            explanations.append(explanation.lower())

    elif name == "FALCON":
        falcon_concept_list = []
        for i in range(len(df)):
            falcon_concepts_all = ast.literal_eval(df["concept_set_noun_phrases"][i])
            falcon_concept = falcon_concepts_all[0][0]
            falcon_concept_list.append(falcon_concept)
        falcon_neuron_ids = df["group"].to_list()
        falcon_concept_ids = dict(zip(falcon_neuron_ids, falcon_concept_list))
        filtered_dict = {key: value for key, value in falcon_concept_ids.items() if key in neuron_ids}
        explanations = list(filtered_dict.values())

    # Check which explanation images are already existing and output missing ones
    explanations_set = set(explanations)
    explanations_set = list(explanations_set)    
    image_directories = [i.replace('_', ' ') for i in os.listdir(image_path)]
    missing_items = list(set(explanations_set) - set(image_directories))
    explanations_filtered = missing_items
 
    return explanations, explanations_filtered


# Function to create a new CSV file with headers
def create_csv(filename, headers):
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(headers)

# Function to check if a CSV file exists
def csv_file_exists(filename):
    return os.path.exists(filename)

# Function to add new rows to an existing CSV file
def add_rows_to_csv(filename, rows):
    with open(filename, mode='a', newline='') as file:
        writer = csv.writer(file)
        for row in rows:
            writer.writerow(row)