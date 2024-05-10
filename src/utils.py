import os
import cv2
import csv
import ast
import json
import itertools
import pandas as pd
from tqdm import tqdm
import torch
import torchvision
import torchvision.transforms.functional as F
from torch.utils.data import Dataset
from transformers import DetrImageProcessor,DetrForObjectDetection,BeitFeatureExtractor,BeitForSemanticSegmentation

DATASET_PATH = {"imagenet_val": "/path/to/images/",
                }

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)

TRANSFORMS = {"transform_imagenet" : torchvision.transforms.Compose([
                        torchvision.transforms.ToPILImage(),
                        torchvision.transforms.Resize(224),
                        torchvision.transforms.CenterCrop((224,224)),
                        torchvision.transforms.ToTensor(),
                        torchvision.transforms.Normalize(mean=MEAN,
                                                        std=STD)
                        ]),
            "transform_ade20k_val" : torchvision.transforms.Compose([
                        torchvision.transforms.Resize(256, interpolation=3),
                        torchvision.transforms.CenterCrop(224),
                        torchvision.transforms.ToTensor(),
                        torchvision.transforms.Normalize(mean=MEAN,
                                                        std=STD)
                        ]),
            "transform_ade20k_train" : torchvision.transforms.Compose([
                        torchvision.transforms.RandomResizedCrop(224),
                        torchvision.transforms.RandomHorizontalFlip(),
                        torchvision.transforms.ToTensor(),
                        torchvision.transforms.Normalize(mean=MEAN,
                                                        std=STD)
                        ])
            }
class ImageDataset(Dataset):
    def __init__(self,root,transform):
        self.root=root
        self.transform=transform

        self.image_names=[self.root + x for x in os.listdir(self.root)]
        self.image_names.sort()
   
    def __len__(self):
        return len(self.image_names)
 
    def __getitem__(self,index):
        image=cv2.imread(self.image_names[index])
        image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)

        image=self.transform(image)

        return image

class ConceptDataset(Dataset):
    def __init__(self,root,concept,transform):
        self.root=root
        self.concept=concept
        self.transform=transform

        # Load ImageNet concept images
        self.df = pd.read_csv("./assets/ILSVRC2012_val_labels.csv")
        self.concept_images = self.df[self.df[self.concept] == 1]["image_name"].tolist()
        self.image_names=[f"{self.root}{x}.JPEG" for x in self.concept_images]
        self.image_names.sort()
   
    def __len__(self):
        return len(self.image_names)
 
    def __getitem__(self,index):
        image=cv2.imread(self.image_names[index])
        image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)

        image=self.transform(image)

        return image
    

# Function to load pre-trained models
def get_target_model(target_name, device):
    """
    returns target model in eval mode and its preprocess function
    target_name: supported options - {resnet18, vit_b_16, detr, beit}
    """
    if target_name == "resnet18":
        weights = torchvision.models.ResNet18_Weights.IMAGENET1K_V1
        preprocess = None
        target_model = torchvision.models.resnet18(weights=weights).to(device)
    elif target_name == "vit_b_16":
        weights = torchvision.models.ViT_B_16_Weights.IMAGENET1K_V1
        preprocess = None
        target_model = torchvision.models.vit_b_16(weights=weights).to(device)
    elif target_name == "detr":
        preprocess = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50", revision="no_timm")
        target_model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50", revision="no_timm", output_hidden_states=True).to(device)
    elif target_name == "beit":
        preprocess = BeitFeatureExtractor.from_pretrained("microsoft/beit-base-finetuned-ade-640-640")
        target_model = BeitForSemanticSegmentation.from_pretrained("microsoft/beit-base-finetuned-ade-640-640", output_hidden_states=True).to(device)
    target_model.eval()
    return target_model, preprocess


def get_data_path(dataset_name): 
    if dataset_name in DATASET_PATH.keys():
        path = DATASET_PATH[dataset_name]
    else:
        raise ValueError("Unsupported dataset_name")
    return path


def get_transform(transform_type):
    if transform_type in TRANSFORMS.keys():
        path = TRANSFORMS[transform_type]
    else:
        raise ValueError("Unsupported transform_type")
    return path


def get_dataset(dataset_name):
    if dataset_name == "imagenet_val" or dataset_name == "imagenet_train":
        data_path = DATASET_PATH[dataset_name]
        data_transform = TRANSFORMS["transform_imagenet"]
        dataset = ImageDataset(root=data_path,
                        transform=data_transform)
    elif dataset_name == "ade20k_val":
        data_path = DATASET_PATH[dataset_name]
        data_transform = TRANSFORMS["transform_ade20k_val"]
        dataset = torchvision.datasets.ImageFolder(root=data_path,
                        transform=data_transform)
    elif dataset_name == "ade20k_train":
        data_path = DATASET_PATH[dataset_name]
        data_transform = TRANSFORMS["transform_ade20k_train"]
        dataset = torchvision.datasets.ImageFolder(root=data_path,
                        transform=data_transform)    
    else:
        raise ValueError("Unsupported dataset_name")
    
    return dataset


def get_n_neurons(model_layer):
    if model_layer == "resnet18-fc" or model_layer == "vit_b_16-head":
        neurons = 1000
    elif model_layer == "resnet18-layer4" or model_layer == "resnet18-avgpool":
        neurons = 512
    elif model_layer == "vit_b_16-layer11" or model_layer == "vit_b_16-ln" or model_layer == "beit-layer13":
        neurons = 768
    elif model_layer == "detr-layer7":
        neurons = 256
    else:
        raise ValueError("Unsupported model_layer")
    return neurons


def get_activations(model, model_name, tensor_path, dataset, dataloader,
                    n_neurons, device, preprocess=None):
    # collect activations
    activation = {}
    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output
        return hook

    if model_name == "resnet18-fc":
        model.fc.register_forward_hook(get_activation("fc"))
    elif model_name == "resnet18-layer4":
        model.layer4.register_forward_hook(get_activation("layer4"))
    elif model_name == "vit_b_16-head":
        model.heads.head.register_forward_hook(get_activation("head"))
    elif model_name == "vit_b_16-layer11":
        model.encoder.layers[11].register_forward_hook(get_activation("layer11"))
    elif model_name == "vit_b_16-ln":
        model.encoder.ln.register_forward_hook(get_activation("ln"))
    
    MODEL_FEATURES = torch.zeros([len(dataset), n_neurons]).to(device)

    counter = 0
    flag = True
    with torch.no_grad():
        for i, x in tqdm(enumerate(dataloader)):
            torch.cuda.empty_cache()
            if model_name == "beit-layer13" or model_name == "detr-layer7":
                x = x[0].float().data.to(device)
                pil_images = [F.to_pil_image(img) for img in x]
                inputs = preprocess(images=pil_images, return_tensors="pt").to(device)         
                outputs = model(**inputs)
                # get last hidden state
                if model_name == "beit-layer13":
                    outputs = outputs.hidden_states[len(outputs.hidden_states)-1]
                elif model_name == "detr-layer7":
                    outputs = outputs.last_hidden_state
            else:
                x = x.float().data.to(device)
                outputs = model(x).data
            if flag:
                if model_name == "resnet18-fc":
                    print(activation["fc"].shape)
                elif model_name == "resnet18-avgpool":
                    print(activation["avgpool"].shape)
                elif model_name == "resnet18-layer4":
                    print(activation["layer4"].shape)
                elif model_name == "vit_b_16-head":
                    print(activation["head"].shape)                        
                elif model_name == "vit_b_16-ln":
                    print(activation["ln"].shape)
                flag = False

            if model_name == "resnet18-fc" or model_name == "vit_b_16-head":
                MODEL_FEATURES[counter:counter + x.shape[0],:] = outputs
            elif model_name == "resnet18-avgpool":
                MODEL_FEATURES[counter:counter + x.shape[0],:] = activation["avgpool"][:, :, 0, 0].data.to(device)
            elif model_name == "resnet18-layer4":
                MODEL_FEATURES[counter:counter + x.shape[0],:] = activation["layer4"].mean(axis =[2,3]).data.to(device)
            elif model_name == "vit_b_16-ln":
                MODEL_FEATURES[counter:counter + x.shape[0],:] = activation["ln"][:,0,:].data.to(device)
            elif model_name == "beit-layer13" or "detr-layer7":
                MODEL_FEATURES[counter:counter + x.shape[0],:] = outputs[:,0,:].data.to(device)
            counter += x.shape[0]

    # Save activations
    torch.save(MODEL_FEATURES, tensor_path)
    
    return MODEL_FEATURES


def clean_label(raw_label):
    label = raw_label.split(".",1)[0]
    label = label.replace("_", " ")
    return label


def get_imgnt_labels(path):
    with open(path) as json_file:
        imgnt_json_data = json.load(json_file)
    imgnt_json_data = dict(itertools.islice(imgnt_json_data.items(), 1000)) # shorten to 1k labels
    imgnt_labels_num = list(imgnt_json_data.keys()) # "n01440764"
    imgnt_labels_str = [clean_label(imgnt_json_data[i]["name"]) for i in imgnt_labels_num] # "tench"
    return imgnt_labels_num, imgnt_labels_str 


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
    image_directories = [i.replace("_", " ") for i in os.listdir(image_path)]
    missing_items = list(set(explanations_set) - set(image_directories))
    explanations_filtered = missing_items
 
    return explanations, explanations_filtered


# Function to create a new CSV file with headers
def create_csv(filename, headers):
    with open(filename, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(headers)

# Function to check if a CSV file exists
def csv_file_exists(filename):
    return os.path.exists(filename)

# Function to add new rows to an existing CSV file
def add_rows_to_csv(filename, rows):
    with open(filename, mode="a", newline="") as file:
        writer = csv.writer(file)
        for row in rows:
            writer.writerow(row)