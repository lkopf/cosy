'''
Goal of this script:
File to use GPT4.o to generalise between images and the activations of neurons.

Expected input:
- Neuron activation per image. Are they all in separate files?
torch.Tensor: Activations tensor of shape [len(dataset), n_neurons].
--> can we trace back the images and the activations? then we would need to not shuffle the dataset in the dataloader

Intermediate step:
- Retrieve the activations of neurons and the corresponding input images, get a top 5 - 10 - 15

Expected output of the script:
- A csv file similar to the assets/explanations 
'''
import os
import time 

import argparse
import asyncio
import torch 

from utils import retrieve_top_x_indices, prompting, response_to_csv, set_openai_api_key


def main(activation_path, top_x, layer, apikey):

    set_openai_api_key(fallback_key=apikey)

    # Load the activations 
    print(f"Loading activations from: {activation_path}")
    activations = torch.load(activation_path)
    print("Activation shape:", activations.shape)

    # Get the top activations and images per neuron in the dataset
    top_x_indices = retrieve_top_x_indices(activations, top_x)
    print("Top-X indices shape:", top_x_indices.shape)

    # Prompting to GPT API
    start_time = time.time()
    responses = asyncio.run(prompting(top_x_indices))
    print("Number of responses:", len(responses))

    # Convert to CSV
    response_to_csv(responses, layer=layer, top_x=top_x)
    print("CSV file created.")
    print("Time taken:", time.time() - start_time)


if __name__ == "__main__":

    try:

        parser = argparse.ArgumentParser(description="Process neuron activations and save responses to CSV.")

        parser.add_argument("--activation_path", type=str, help="Path to the activations .pt file")
        parser.add_argument("--top_x", type=int, default=5, help="Number of top activations to retrieve (default: 5)")
        parser.add_argument("--layer", type=int, default=4, help="Layer number for CSV metadata (default: 4)")
        parser.add_argument("--apikey", type=str, default="APIKEY", help="API key for OpenAI GPT4 (default: API). Not needed if you have the key in your environment")

        args = parser.parse_args()
        main(args.activation_path, args.top_x, args.layer, args.apikey)
    
    except:
        activation_path = "/cosy/src/LLM_activations/activations/val_resnet18-layer4.pt"
        top_x = 1
        layer = 4
        apikey = "YOUR API KEY HERE"
        
        # check activation_path exists
        if not os.path.exists(activation_path):
            raise FileNotFoundError(f"Activation path {activation_path} does not exist.")

        activations = torch.load(activation_path)

        main(activation_path, top_x, layer, apikey)
