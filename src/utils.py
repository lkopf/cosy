import os
import cv2
import csv
import ast
import json
import types
import itertools
import pandas as pd
from tqdm import tqdm
import torch
import torchvision
import torchvision.transforms.functional as F
from torch.utils.data import Dataset
from transformers import DetrImageProcessor,DetrForObjectDetection,BeitFeatureExtractor,BeitForSemanticSegmentation

DATASET_PATH = {# dataset : "/path/to/images/",
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
            "transform_places365" : torchvision.transforms.Compose([
                        torchvision.transforms.Resize((256,256)),
                        torchvision.transforms.CenterCrop((224)),
                        torchvision.transforms.ToTensor(),
                        torchvision.transforms.Normalize(mean=MEAN,
                                                        std=STD)
                        ]),
            "transform_train_places365" : torchvision.transforms.Compose([
                        torchvision.transforms.ToPILImage(),
                        torchvision.transforms.Resize((256,256)),
                        torchvision.transforms.CenterCrop((224)),
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
        print((self.image_names[index]))
        image=cv2.imread(self.image_names[index])
        image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)

        image=self.transform(image)

        return image

class PlacesDataset(Dataset):
    def __init__(self,root):
        self.root=root

        self.image_names=[self.root + x for x in os.listdir(self.root)]
        self.image_names.sort()
   
    def __len__(self):
        return len(self.image_names)
 
    def __getitem__(self,index):
        # print((self.image_names[index]))
        return self.image_names[index]

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
    

class Subset(torch.nn.Module):
    indexes: torch.Tensor

    def __init__(self, indexes):
        super().__init__()
        self.register_buffer("indexes", indexes)

    def forward(self, X: torch.Tensor):
        return X[:, self.indexes]

# Function to load pre-trained models
def get_target_model(target_name, device):
    """
    returns target model in eval mode and its preprocess function
    target_name: supported options - {resnet18, vit_b_16, detr, beit}
    """
    if target_name == "resnet18":
        weights = torchvision.models.ResNet18_Weights.IMAGENET1K_V1
        preprocess = weights.transforms()
        target_model = torchvision.models.resnet18(weights=weights).to(device).eval()
        features_layer = target_model.avgpool
    elif target_name == "resnet50_places":
        arch = 'resnet50'
        # load the pre-trained weights
        model_file = '%s_places365.pth.tar' % arch
        if not os.access(model_file, os.W_OK):
            weight_url = 'http://places2.csail.mit.edu/models_places365/' + model_file
            os.system('wget ' + weight_url)
        target_model = torchvision.models.__dict__[arch](num_classes=365)
        checkpoint = torch.load(model_file, map_location=lambda storage, loc: storage)
        state_dict = {str.replace(k,'module.',''): v for k,v in checkpoint['state_dict'].items()}
        target_model.load_state_dict(state_dict)
        target_model = target_model.eval().to(device)
        preprocess = TRANSFORMS["transform_places365"]   
        features_layer = target_model.avgpool
    # elif target_name == "vit_b_16":
    #     weights = torchvision.models.ViT_B_16_Weights.IMAGENET1K_V1
    #     preprocess = weights.transforms()
    #     target_model = torchvision.models.vit_b_16(weights=weights).to(device).eval()
    # elif "densenet161" in target_name:
    #     weights = torchvision.models.DenseNet161_Weights.IMAGENET1K_V1
    #     preprocess = weights.transforms()
    #     target_model = torchvision.models.densenet161(weights=weights).to(device)
    # elif "googlenet" in target_name:
    #     weights = torchvision.models.GoogLeNet_Weights.IMAGENET1K_V1
    #     preprocess = weights.transforms()
    #     target_model = torchvision.models.googlenet(weights=weights).to(device)
    elif target_name == 'densenet161':
        weights = torchvision.models.DenseNet161_Weights.IMAGENET1K_V1
        target_model = torchvision.models.densenet161(weights=weights).to(device).eval()
        preprocess = weights.transforms()

        target_model.features.add_module("relu", torch.nn.ReLU(inplace=False))
        target_model.features.add_module(
            "adaptive_avgpool", torch.nn.AdaptiveAvgPool2d((1, 1)))
        target_model.features.add_module("flatten", torch.nn.Flatten(1))
        def new_forward(self, x: torch.Tensor):
            features = self.features(x)
            out = self.classifier(features)
            return out
        target_model.forward = types.MethodType(new_forward, target_model)
        features_layer = target_model.features.to(device).eval()
    elif "densenet161_places" in target_name:
        arch = 'densenet161'
        # load the pre-trained weights
        model_file = '%s_places365.pth.tar' % arch
        if not os.access(model_file, os.W_OK):
            weight_url = 'http://places2.csail.mit.edu/models_places365/' + model_file
            os.system('wget ' + weight_url)
        target_model = torchvision.models.__dict__[arch](num_classes=365)
        checkpoint = torch.load(model_file, map_location=lambda storage, loc: storage)
        state_dict = {str.replace(k,'module.',''): v for k,v in checkpoint['state_dict'].items()}
        if arch == 'densenet161':
            state_dict = {str.replace(k,'norm.','norm'): v for k,v in state_dict.items()}
            state_dict = {str.replace(k,'conv.','conv'): v for k,v in state_dict.items()}
            state_dict = {str.replace(k,'normweight','norm.weight'): v for k,v in state_dict.items()}
            state_dict = {str.replace(k,'normrunning','norm.running'): v for k,v in state_dict.items()}
            state_dict = {str.replace(k,'normbias','norm.bias'): v for k,v in state_dict.items()}
            state_dict = {str.replace(k,'convweight','conv.weight'): v for k,v in state_dict.items()}
        target_model.load_state_dict(state_dict)
        preprocess = TRANSFORMS["transform_places365"]
        # second to last layer:
        target_model.features.add_module("relu", torch.nn.ReLU(inplace=False))
        target_model.features.add_module(
            "adaptive_avgpool", torch.nn.AdaptiveAvgPool2d((1, 1)))
        target_model.features.add_module("flatten", torch.nn.Flatten(1))
        def new_forward(self, x: torch.Tensor):
            features = self.features(x)
            out = self.classifier(features)
            return out
        target_model.forward = types.MethodType(new_forward, target_model)
        target_model = target_model.to(device).eval()
        features_layer = target_model.features.to(device).eval()
        # target_model = features_layer
    elif target_name == 'vit_b_16':
        weights = torchvision.models.ViT_B_16_Weights.IMAGENET1K_V1
        target_model = torchvision.models.vit_b_16(weights=weights).eval()
        preprocess = weights.transforms()
        # see here https://pytorch.org/vision/stable/_modules/torchvision/models/vision_transformer.html#vit_b_16
        index = torch.zeros([1]).long()
        setattr(target_model, 'subset', torch.nn.Sequential(Subset(index),
                                                     torch.nn.Flatten()))
        def new_forward(self, x: torch.Tensor):
            # Reshape and permute the input tensor
            x = self._process_input(x)
            n = x.shape[0]
            # Expand the class token to the full batch
            batch_class_token = self.class_token.expand(n, -1, -1)
            x = torch.cat([batch_class_token, x], dim=1)
            x = self.encoder(x)
            x = self.subset(x)
            x = self.heads(x)
            return x

        target_model.forward = types.MethodType(new_forward, target_model)
        target_model = target_model.to(device)
        features_layer = target_model.subset
        features_layer = features_layer.eval()

    target_model.eval()
    return target_model, features_layer, preprocess


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
        dataset = torchvision.datasets.Places365(root=data_path,split="train-standard",
                        transform=data_transform)
    if dataset_name == "places365_train_all":
        data_path = DATASET_PATH[dataset_name]
        data_transform = TRANSFORMS["transform_train_places365"]
        dataset = ImageDataset(root=data_path,
                        transform=data_transform)
    elif dataset_name == "places365_train":
        data_path = DATASET_PATH[dataset_name]
        data_transform = TRANSFORMS["transform_places365"]
        dataset = torchvision.datasets.Places365(root=data_path,split="train-standard",
                        transform=data_transform)
    elif dataset_name == "places365_val":
        data_path = DATASET_PATH[dataset_name]
        data_transform = TRANSFORMS["transform_places365"]
        dataset = torchvision.datasets.Places365(root=data_path,split="val",
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
    if model_layer == "resnet18-fc" or model_layer == "densenet161-fc" or model_layer == "googlenet-fc" or model_layer == "vit_b_16-head":
        neurons = 1000
    elif model_layer == "resnet18-layer4" or model_layer == "resnet18-avgpool":
        neurons = 512
    elif model_layer == "resnet18-layer3":
        neurons = 256
    elif model_layer == "resnet18-layer2":
        neurons = 128
    elif model_layer == "resnet18-layer1":
        neurons = 64
    elif model_layer == "resnet50_places-avgpool":
        neurons =2048
    elif model_layer == "densenet161-features" or model_layer == "densenet161_places-features" or model_layer == "densenet161-denseblock4":
        neurons = 2208
    elif model_layer == "googlenet-inception5b":
        neurons = 1024
    elif model_layer == "vit_b_16-features" or model_layer == "vit_b_16-layer11" or model_layer == "vit_b_16-ln" or model_layer == "beit-layer13":
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

    if model_name == "resnet18-fc" or model_name == "googlenet-fc":
        model.fc.register_forward_hook(get_activation("fc"))
    if model_name == "resnet50_places-avgpool" or model_name == "resnet18-avgpool":
        model.avgpool.register_forward_hook(get_activation("avgpool"))
    elif model_name == "resnet18-layer4":
        model.layer4.register_forward_hook(get_activation("layer4"))
    elif model_name == "resnet18-layer3":
        model.layer3.register_forward_hook(get_activation('layer3'))
    elif model_name == "resnet18-layer2":
        model.layer2.register_forward_hook(get_activation('layer2'))
    elif model_name == "resnet18-layer1":
        model.layer1.register_forward_hook(get_activation('layer1'))
    elif model_name == "densenet161-denseblock4":
        model.features.denseblock4.register_forward_hook(get_activation('denseblock4'))
    elif model_name == "densenet161-features" or model_name == "densenet161_places-features":
        model.register_forward_hook(get_activation('features'))
    elif model_name == "densenet161-fc":
        model.classifier.register_forward_hook(get_activation('classifier'))
    elif model_name == "googlenet-inception5b":
        model.inception5b.register_forward_hook(get_activation('inception5b'))   
    elif model_name == "vit_b_16-features":
        model.subset.register_forward_hook(get_activation('heads'))
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
            # FIX THIS FOR ALL DATASETS!!!
            # if model_name == "densenet161_places-features" or  model_name == "resnet50_places-avgpool":
            #     x = x[0].float().data.to(device)
            # else:
            x = x.float().data.to(device)
            outputs = model(x).data
            if flag:
                if model_name == "resnet18-fc" or model_name == "googlenet-fc":
                    print(activation["fc"].shape)
                elif model_name == "resnet18-avgpool" or model_name == "resnet50_places-avgpool":
                    print(activation["avgpool"].shape)
                elif model_name == "resnet18-layer4":
                    print(activation["layer4"].shape)
                elif model_name == "resnet18-layer3":
                    print(activation['layer3'].shape)
                elif model_name == "resnet18-layer2":
                    print(activation['layer2'].shape)
                elif model_name == "resnet18-layer1":
                    print(activation['layer1'].shape)
                elif model_name == "densenet161-features"or model_name == "densenet161_places-features":
                    print(activation['features'].shape)
                elif model_name == "densenet161-fc":
                    print(activation['classifier'].shape)
                elif model_name == "densenet161-denseblock4":
                    print(activation['denseblock4'].shape)
                elif model_name == "googlenet-inception5b":
                    print(activation['inception5b'].shape)
                elif model_name == "vit_b_16-layer11":
                    print(activation['layer11'].shape)
                elif model_name == "vit_b_16-features":
                    print(activation["heads"].shape) 
                elif model_name == "vit_b_16-head":
                    print(activation["head"].shape)                        
                elif model_name == "vit_b_16-ln":
                    print(activation["ln"].shape)
                flag = False

            if model_name == "resnet18-fc" or model_name == "densenet161_places-features" or model_name == "densenet161-fc" or model_name == "googlenet-fc" or model_name == "vit_b_16-head" or model_name == "densenet161-features":
                MODEL_FEATURES[counter:counter + x.shape[0],:] = outputs
            elif model_name == "vit_b_16-features":
                MODEL_FEATURES[counter:counter + x.shape[0],:] = activation["heads"].data.to(device)            
            elif model_name == "resnet18-avgpool" or model_name == "resnet50_places-avgpool":
                MODEL_FEATURES[counter:counter + x.shape[0],:] = activation["avgpool"][:, :, 0, 0].data.to(device)
            elif model_name == "resnet18-layer4":
                MODEL_FEATURES[counter:counter + x.shape[0],:] = activation["layer4"].mean(axis =[2,3]).data.to(device)
            elif model_name == "resnet18-layer3":
                MODEL_FEATURES[counter:counter + x.shape[0],:] = activation['layer3'].mean(axis =[2,3]).data.to(device)
            elif model_name == "resnet18-layer2":
                MODEL_FEATURES[counter:counter + x.shape[0],:] = activation['layer2'].mean(axis =[2,3]).data.to(device)
            elif model_name == "resnet18-layer1":
                MODEL_FEATURES[counter:counter + x.shape[0],:] = activation['layer1'].mean(axis =[2,3]).data.to(device)
            elif model_name == "densenet161-denseblock4":
                MODEL_FEATURES[counter:counter + x.shape[0],:] = activation['denseblock4'].mean(axis =[2,3]).data.to(device)
            elif model_name == "googlenet-inception5b":
                MODEL_FEATURES[counter:counter + x.shape[0],:] = activation['inception5b'].mean(axis=[2,3]).data.to(device)
            elif model_name == "vit_b_16-layer11":
                MODEL_FEATURES[counter:counter + x.shape[0],:] = activation['layer11'][:,0,:].data.to(device)            
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
    if name == "INVERT": # FIX THIS FOR ALL DATASETS!!!
        # with open("/mnt/beegfs/home/lkopf/CLIP-dissect/data/categories_places365_clean.txt", "r") as file:
        #     # Read all lines from the file
        #     lines = file.readlines()
        # places_map = {index: line.strip() for index, line in enumerate(lines)}
        # explanations = [] # all explanations
        # for neuron_id in neuron_ids:
        #     explanation_raw = df["formula"][neuron_id]
        #     explanation = places_map[explanation_raw]
        #     explanations.append(explanation)

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