
import asyncio
import base64
import csv
import os
import time

import numpy as np
import pandas as pd
import torch

import asyncio
from openai import AsyncOpenAI, RateLimitError


IMAGE_PATH = "/path/to/images/"  
IMAGES = os.listdir(IMAGE_PATH)  

client: AsyncOpenAI | None = None

def set_openai_api_key(fallback_key=None):
    """
    Sets the OpenAI API key from an environment variable if available.
    If not found, uses the provided fallback key.
    """

    global client 
    api_key = os.getenv("OPENAI_API_KEY", fallback_key)

    if not api_key:
        raise ValueError("OpenAI API key is missing! Set OPENAI_API_KEY as an environment variable or provide a fallback key.")
    client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY", fallback_key))


def load_prompt(prompt_path="/cosy/src/LLM_activations/prompt.txt") -> str:
    """
    Loads the default prompt from a text file.
    """
    with open(prompt_path, "r", encoding="utf-8") as f:
        return f.read().strip()

def retrieve_top_x_indices(activations_tensor: torch.Tensor, x: int) -> torch.Tensor:
    '''
    Retrieve the top x indices of the activations tensor for each neuron. Results in (neurons, x) shape.
    '''

    if x <= 0:
        raise ValueError(f"Expected x > 0, but got x={x}")

    if activations_tensor.numel() == 0:
        raise ValueError("activations_tensor is empty, cannot retrieve top-x indices.")

    print("Tensor shape:", activations_tensor.shape)

    #  If tensor is 1D, we need to add a dimension
    if activations_tensor.ndim == 1:
        top_x = torch.argsort(activations_tensor, descending=True)[:x]
        print("Top indices (1D):", top_x)
        return top_x

    top_x = torch.argsort(activations_tensor, dim=0)[-x:, :]
    top_x = top_x.transpose(0, 1) 

    print("Top indices (2D):", top_x)

    return top_x


def retrieve_top_x_activations(activations_tensor: torch.Tensor, indices: list) -> torch.Tensor:
    '''
    Retrieve the top x activations of the activations tensor.
    '''
    if activations_tensor.numel() == 0:
        raise ValueError("activations_tensor is empty, cannot retrieve top-x activations.")
    
    print("Tensor shape:", activations_tensor.shape)

    if activations_tensor.ndim == 1:
        # For 1D tensor, just index directly
        top_x_activations = activations_tensor[indices]
        print("Top activations (1D):", top_x_activations)
        return top_x_activations

    top_x_activations = activations_tensor[indices]
    top_x_activations = top_x_activations.transpose(0, 1) 
    print("Top activations (2D):", top_x_activations)
    return top_x_activations


def encode_image(image_path: str) -> str:
    '''
    Encode the images in format that can be used in the openai API
    '''
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def get_image_from_indices(image_indices_list: list, encode_images: bool = True) -> list[str]:
    '''
    Get the images that belong to the indices found with the highest activations per neuron and decode in format
    suitable for the openai API.
    '''

    if encode_images:
        encoded_images = [[encode_image(os.path.join(IMAGE_PATH, IMAGES[indice]))]for indice in image_indices_list]
    else:
        encoded_images = [os.path.join(IMAGE_PATH, IMAGES[indice]) for indice in image_indices_list]

    #print("Top indices:", image_indices_list)
    #print("Top indices shape:", len(image_indices_list))
    #print("Selected images number:", len(encoded_images))

    return encoded_images


async def prompting(top_x_indices: list, 
                    batch_size=3, 
                    max_tokens_per_minute=29000, 
                    prompt=None):
    '''
    Input: top_x_indices: list of top x indices of the activations tensor for each neuron.
    
    This function prompts the GPT-4o model asynchronously with images and their activations.
    It avoids rate limiting by batching requests and tracking token usage, inserting a pause when necessary.
    '''

    if prompt == None:
        prompt = load_prompt()  

    async def query_gpt(indices):
        '''Helper function to send a request to GPT-4o asynchronously.'''
        encoded_images = get_image_from_indices(indices)  

        messages = [
            {"role": "system", "content": "You are an AI designed for academic research. Provide detailed and structured responses based on the images and activations provided."},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    *[
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img[0]}"}}
                        for img in encoded_images 
                    ]
                ]
            }
        ]

        delay = 2  

        while True:  
            try:
                response = await client.chat.completions.create(
                    model="gpt-4o",
                    messages=messages
                )
                return response  
            
            except RateLimitError as e:
                wait_time = float(e.retry_after) if hasattr(e, "retry_after") else delay
                print(f"Rate limit reached. Retrying in {wait_time:.1f} seconds...")
                await asyncio.sleep(wait_time)
                delay = min(delay * 2, 60)  

            except Exception as e:
                print(f"Unexpected error: {e}. Retrying in {delay} seconds...")
                await asyncio.sleep(delay)
                delay = min(delay * 2, 60) 

    responses = []
    estimated_token_usage = 0 

    for i in range(0, len(top_x_indices), batch_size):
        chunk = top_x_indices[i : i + batch_size]

        estimated_token_usage += len(chunk) * 4000  

        if estimated_token_usage >= max_tokens_per_minute:
            wait_time = 60  
            print(f"Approaching rate limit. Sleeping for {wait_time} seconds...")
            await asyncio.sleep(wait_time)
            estimated_token_usage = 0  

        tasks = [query_gpt(j) for j in chunk]
        raw_responses = await asyncio.gather(*tasks)
        texts = [r.choices[0].message.content for r in raw_responses]
        responses.extend(texts)

    return responses



def response_to_csv(responses, layer=1, top_x=1, prompt_number=None):
    '''
    Store the responses from the GPT-4o model in a csv file.
    '''
    count = 0

    if prompt_number == None:
        prompt_number=""
    else:
        prompt_number = f"_{prompt_number}"

    if top_x ==1:
        filename = f"resnet18_layer{layer}{prompt_number}.csv"
    else:
        filename = f"resnet18_layer{layer}_{top_x}{prompt_number}.csv"

    with open(filename, "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["layer", "unit", "description"])  
        
        for response in responses:
            unit = count
            concept = response
            count += 1
            writer.writerow([layer, unit, concept])  










