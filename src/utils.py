import os
import csv
import types
import ast
import cv2
import pandas as pd
from tqdm import tqdm
import torch
import torchvision
from torch.utils.data import Dataset

DATASET_PATH = {  # dataset : "/path/to/images/",
    "imagenet": "/path/to/imagenet_val/",
    "places365": "/path/to/places365_val",
}

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)

TRANSFORMS = {
    "transform_imagenet": torchvision.transforms.Compose(
        [
            torchvision.transforms.ToPILImage(),
            torchvision.transforms.Resize(224),
            torchvision.transforms.CenterCrop((224, 224)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=MEAN, std=STD),
        ]
    ),
    "transform_places365": torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize((256, 256)),
            torchvision.transforms.CenterCrop((224)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=MEAN, std=STD),
        ]
    ),
}


class ImageDataset(Dataset):
    """
    Dataset class for loading images.
    """

    def __init__(self, root, transform):
        """
        Initialize the ImageDataset.

        Args:
            root (str): Root directory of the dataset.
            transform (torchvision.transforms.Compose): Image transformation pipeline.
        """
        self.root = root
        self.transform = transform

        self.image_names = [self.root + x for x in os.listdir(self.root)]
        self.image_names.sort()

    def __len__(self):
        """
        Get the length of the dataset.

        Returns:
            int: Length of the dataset.
        """
        return len(self.image_names)

    def __getitem__(self, index):
        """
        Get an item from the dataset.

        Args:
            index (int): Index of the item.

        Returns:
            torch.Tensor: Transformed image.
        """
        print((self.image_names[index]))
        image = cv2.imread(self.image_names[index])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image = self.transform(image)

        return image


def get_target_model(target_name, device):
    """
    Get the target model in eval mode and its preprocess function.

    Args:
        target_name (str): Name of the target model.
        device (torch.device): Device to load the model on.

    Returns:
        torch.nn.Module: Target model.
        torch.nn.Module: Features layer of the target model.
        torchvision.transforms.Compose: Preprocess function for the target model.
    """
    
    def download_weights(arch, model_file):
        """Helper function to download model weights if not already present."""
        if not os.access(model_file, os.W_OK):
            weight_url = f"http://places2.csail.mit.edu/models_places365/{model_file}"
            os.system(f"wget {weight_url}")

    def load_places_model(arch, model_file, num_classes=365):
        """Helper function to load the Places365 model."""
        download_weights(arch, model_file)
        target_model = torchvision.models.__dict__[arch](num_classes=num_classes)
        checkpoint = torch.load(model_file, map_location=lambda storage, loc: storage)
        state_dict = {
            str.replace(k, "module.", ""): v
            for k, v in checkpoint["state_dict"].items()
        }
        if arch == "densenet161":
            state_dict = {str.replace(k, "norm.", "norm"): v for k, v in state_dict.items()}
            state_dict = {str.replace(k, "conv.", "conv"): v for k, v in state_dict.items()}
            state_dict = {str.replace(k, "normweight", "norm.weight"): v for k, v in state_dict.items()}
            state_dict = {str.replace(k, "normrunning", "norm.running"): v for k, v in state_dict.items()}
            state_dict = {str.replace(k, "normbias", "norm.bias"): v for k, v in state_dict.items()}
            state_dict = {str.replace(k, "convweight", "conv.weight"): v for k, v in state_dict.items()}
        target_model.load_state_dict(state_dict)
        return target_model

    if target_name == "resnet18":
        weights = torchvision.models.ResNet18_Weights.IMAGENET1K_V1
        preprocess = weights.transforms()
        target_model = torchvision.models.resnet18(weights=weights).to(device).eval()
        features_layer = target_model.avgpool

    elif target_name == "vit_b_16":
        weights = torchvision.models.ViT_B_16_Weights.IMAGENET1K_V1
        preprocess = weights.transforms()
        target_model = torchvision.models.vit_b_16(weights=weights).to(device).eval()
        features_layer = target_model.heads.head

    elif "densenet161" in target_name:
        weights = torchvision.models.DenseNet161_Weights.IMAGENET1K_V1
        preprocess = weights.transforms()
        target_model = torchvision.models.densenet161(weights=weights).to(device).eval()
        features_layer = target_model.classifier

    elif "googlenet" in target_name:
        weights = torchvision.models.GoogLeNet_Weights.IMAGENET1K_V1
        preprocess = weights.transforms()
        target_model = torchvision.models.googlenet(weights=weights).to(device).eval()
        features_layer = target_model.fc

    elif target_name == "resnet50_places":
        arch = "resnet50"
        model_file = f"{arch}_places365.pth.tar"
        target_model = load_places_model(arch, model_file).to(device).eval()
        preprocess = TRANSFORMS["transform_places365"]
        features_layer = target_model.avgpool

    elif "densenet161_places" in target_name:
        arch = "densenet161"
        model_file = f"{arch}_places365.pth.tar"
        target_model = load_places_model(arch, model_file).to(device).eval()
        preprocess = TRANSFORMS["transform_places365"]
        
        # Add required layers for Places365 DenseNet161
        target_model.features.add_module("relu", torch.nn.ReLU(inplace=False))
        target_model.features.add_module("adaptive_avgpool", torch.nn.AdaptiveAvgPool2d((1, 1)))
        target_model.features.add_module("flatten", torch.nn.Flatten(1))

        # Redefine the forward method to include the new features
        def new_forward(self, x: torch.Tensor):
            features = self.features(x)
            out = self.classifier(features)
            return out

        target_model.forward = types.MethodType(new_forward, target_model)
        features_layer = target_model.features

    target_model.eval()
    return target_model, features_layer, preprocess



def get_data_path(dataset_name):
    """
    Get the data path for a given dataset name.

    Args:
        dataset_name (str): Name of the dataset.

    Returns:
        str: Data path for the dataset.

    Raises:
        ValueError: If the dataset name is not supported.
    """

    if dataset_name in DATASET_PATH.keys():
        path = DATASET_PATH[dataset_name]
    else:
        raise ValueError("Unsupported dataset_name")
    return path


def get_transform(transform_type):
    """
    Get the transform for a given transform type.

    Args:
        transform_type (str): Type of the transform.

    Returns:
        torchvision.transforms.Compose: Transform for the given type.

    Raises:
        ValueError: If the transform type is not supported.
    """

    if transform_type in TRANSFORMS.keys():
        path = TRANSFORMS[transform_type]
    else:
        raise ValueError("Unsupported transform_type")
    return path


def get_dataset(dataset_name):
    """
    Get the dataset for a given dataset name.

    Args:
        dataset_name (str): Name of the dataset.

    Returns:
        ImageDataset: Dataset for the given name.

    Raises:
        ValueError: If the dataset name is not supported.
    """

    if dataset_name == "imagenet":
        data_path = DATASET_PATH[dataset_name]
        data_transform = TRANSFORMS["transform_imagenet"]
        dataset = ImageDataset(root=data_path, transform=data_transform)
    elif dataset_name == "places365":
        data_path = DATASET_PATH[dataset_name]
        data_transform = TRANSFORMS["transform_places365"]
        dataset = ImageDataset(root=data_path, transform=data_transform)
    else:
        raise ValueError("Unsupported dataset_name")

    return dataset


def get_n_neurons(model_layer):
    """
    Get the number of neurons for a given model layer.

    Args:
        model_layer (str): Name of the model layer.

    Returns:
        int: Number of neurons in the layer.

    Raises:
        ValueError: If the model layer is not supported.
    """

    if (
        model_layer == "resnet18-fc"
        or model_layer == "densenet161-fc"
        or model_layer == "googlenet-fc"
        or model_layer == "vit_b_16-head"
    ):
        neurons = 1000
    elif model_layer == "resnet18-layer4" or model_layer == "resnet18-avgpool":
        neurons = 512
    elif model_layer == "resnet50_places-avgpool":
        neurons = 2048
    elif (
        model_layer == "densenet161-features"
        or model_layer == "densenet161_places-features"
    ):
        neurons = 2208
    elif model_layer == "vit_b_16-features":
        neurons = 768
    else:
        raise ValueError("Unsupported model_layer")
    return neurons


def get_activations(
    model,
    model_name,
    tensor_path,
    dataset,
    dataloader,
    n_neurons,
    device,
    preprocess=None,
):
    """
    Get the activations of a model for a given dataset.

    Args:
        model (torch.nn.Module): Model to get activations from.
        model_name (str): Name of the model.
        tensor_path (str): Path to save the activations tensor.
        dataset (torch.utils.data.Dataset): Dataset to get activations for.
        dataloader (torch.utils.data.DataLoader): DataLoader for the dataset.
        n_neurons (int): Number of neurons in the model layer.
        device (torch.device): Device to load the model on.
        preprocess (torchvision.transforms.Compose, optional): Preprocess function for the model.

    Returns:
        torch.Tensor: Activations tensor of shape [len(dataset), n_neurons].
    """

    activation = {}

    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output
        return hook

    hook_layers = {
        "resnet18-fc": model.fc,
        "googlenet-fc": model.fc,
        "resnet50_places-avgpool": model.avgpool,
        "resnet18-avgpool": model.avgpool,
        "resnet18-layer4": model.layer4,
        "densenet161-features": model.features,
        "densenet161_places-features": model.features,
        "densenet161-fc": model.classifier,
        "vit_b_16-features": model.heads,
        "vit_b_16-head": model.heads.head,
    }

    # Register the forward hook for the specified model layer
    if model_name in hook_layers:
        hook_layers[model_name].register_forward_hook(get_activation(model_name.split('-')[-1]))

    model_features = torch.zeros([len(dataset), n_neurons]).to(device)

    with torch.no_grad():
        for i, x in tqdm(enumerate(dataloader), total=len(dataloader)):
            torch.cuda.empty_cache()
            x = x.float().to(device)
            _ = model(x)  # Forward pass to populate the activation dictionary

            if i == 0:
                # Print the shape of the activation for debugging purposes
                print(f"Activation shape for {model_name}: {activation[model_name.split('-')[-1]].shape}")

            # Assign the appropriate activation to the model_features tensor
            if model_name in ["resnet18-fc", "googlenet-fc", "densenet161-fc", "vit_b_16-head"]:
                model_features[i * x.size(0):(i + 1) * x.size(0), :] = activation[model_name.split('-')[-1]].data
            elif model_name == "vit_b_16-features":
                model_features[i * x.size(0):(i + 1) * x.size(0), :] = activation["heads"].data
            elif model_name in ["resnet18-avgpool", "resnet50_places-avgpool"]:
                model_features[i * x.size(0):(i + 1) * x.size(0), :] = activation["avgpool"][:, :, 0, 0].data
            elif model_name == "resnet18-layer4":
                model_features[i * x.size(0):(i + 1) * x.size(0), :] = activation["layer4"].mean(dim=[2, 3]).data
            elif model_name in ["densenet161-features", "densenet161_places-features"]:
                model_features[i * x.size(0):(i + 1) * x.size(0), :] = activation["features"].data

    torch.save(model_features, tensor_path)
    return model_features


def load_explanations(path, name, image_path, neuron_ids):
    """
    Load explanations based on the given parameters.

    Args:
        path (str): The path to the CSV file containing the explanations.
        name (str): The name of the explanation method.
        image_path (str): The path to the directory containing the explanation images.
        neuron_ids (list): A list of neuron IDs for which explanations are needed.

    Returns:
        tuple: A tuple containing two lists - explanations and explanations_filtered.
            - explanations: A list of explanations for the given neuron IDs.
            - explanations_filtered: A list of explanations that are missing corresponding images.

    Raises:
        FileNotFoundError: If the CSV file or the image directory does not exist.
    """

    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        raise FileNotFoundError("The CSV file does not exist.")
    
    if name == "INVERT":
        explanations = []
        for neuron_id in neuron_ids:
            explanation = df.loc[df["neuron"] == neuron_id, "concept"].values[0]
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
        filtered_dict = {
            key: value for key, value in falcon_concept_ids.items() if key in neuron_ids
        }
        explanations = list(filtered_dict.values())

    # Check which explanation images are already existing and output missing ones
    explanations_set = set(explanations)
    explanations_set = list(explanations_set)
    image_directories = [i.replace("_", " ") for i in os.listdir(image_path)]
    missing_items = list(set(explanations_set) - set(image_directories))
    explanations_filtered = missing_items

    return explanations, explanations_filtered


def create_csv(filename, headers):
    """
    Create a new CSV file with the given filename and write the headers to it.

    Args:
        filename (str): The name of the CSV file to create.
        headers (list): A list of strings representing the column headers.

    Returns:
        None
    """

    with open(filename, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(headers)


def add_rows_to_csv(filename, rows):
    """
    Appends rows to a CSV file.

    Args:
        filename (str): The path to the CSV file.
        rows (list): A list of rows to be added to the CSV file.

    Returns:
        None
    """

    with open(filename, mode="a", newline="") as file:
        writer = csv.writer(file)
        for row in rows:
            writer.writerow(row)
