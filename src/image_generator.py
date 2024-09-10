"""
This script generates realistic images based on given explanations using a stable diffusion model.

Usage:
    torchrun src/image_generator.py --nproc_per_node=3
"""

import os
import random
from datetime import datetime
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from diffusers import DiffusionPipeline

import utils

torch.cuda.empty_cache()

random.seed(42)

start = datetime.now()
print("START: ", start)

IMAGE_PATH = "./gen_images/"

os.makedirs(IMAGE_PATH, exist_ok=True)

METHOD = (
    "INVERT"
    # "MILAN"
    # "CLIP-Dissect"
    # "FALCON"
)
print(METHOD)
MODEL_NAME = (
    "resnet18-avgpool"
    # "resnet18-fc"
    # "resnet18-layer4"
    # "resnet18-layer3"
    # "resnet18-layer2"
    # "resnet18-layer1"
    # "resnet50-avgpool"
    # "vit_b_16-features"
    # "resnet50_places-avgpool"
    # "densenet161_places-features"
)
print(MODEL_NAME)
# prompt_text = "a"
# prompt_text = "a painting of"
# prompt_text = "photo of"
# prompt_text = "realistic photo of"
prompt_text = "realistic photo of a close up of"
print(prompt_text)

N_NEURONS = utils.get_n_neurons(MODEL_NAME)
N_NEURONS_RANDOM = 50
NEURON_IDS = random.sample(range(N_NEURONS), N_NEURONS_RANDOM)

EXPLANATION_PATH = f"./assets/explanations/{METHOD}/{MODEL_NAME}.csv"
_, EXPLANATIONS = utils.load_explanations(
    path=EXPLANATION_PATH, name=METHOD, image_path=IMAGE_PATH, neuron_ids=NEURON_IDS
)
print(EXPLANATIONS)

N_SIZE = 3  # world and batch size
N_IMAGES = 50  # number of generated images

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

# Load stable diffusion model
model_id = "stabilityai/stable-diffusion-xl-base-1.0"
pipe = DiffusionPipeline.from_pretrained(
    model_id, torch_dtype=torch.float16, use_safetensors=True, variant="fp16"
)
pipe = pipe.to("cuda")
# set seed for stable diffusion
generator = torch.Generator("cuda").manual_seed(0)


def run_inference(
    rank, world_size, batch_size, image_path=IMAGE_PATH, num_images=N_IMAGES
):
    """
    Run the inference process for generating images.

    Args:
        rank (int): The rank of the current process.
        world_size (int): The total number of processes.
        batch_size (int): The batch size for each process.
        image_path (str): The path to save the generated images.
        num_images (int): The number of images to generate for each explanation.
    """
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

    pipe.to(rank)

    num_batches = (len(EXPLANATIONS) + batch_size - 1) // batch_size

    for batch_index in range(num_batches):
        torch.cuda.empty_cache()
        start_index = batch_index * batch_size
        end_index = min((batch_index + 1) * batch_size, len(EXPLANATIONS))

        current_batch = EXPLANATIONS[start_index:end_index]
        print(current_batch, len(current_batch))

        for i in range(num_images):
            if torch.distributed.get_rank() == 0:
                prompt = f"{prompt_text} {current_batch[0]}"
                prompt_name = current_batch[0]
            elif torch.distributed.get_rank() == 1 and len(current_batch) > 0:
                prompt = f"{prompt_text} {current_batch[1]}"
                prompt_name = current_batch[1]
            elif torch.distributed.get_rank() == 2 and len(current_batch) > 1:
                prompt = f"{prompt_text} {current_batch[2]}"
                prompt_name = current_batch[2]

            folder_name = image_path + f"{prompt_name.replace(' ', '_')}"
            os.makedirs(folder_name, exist_ok=True)

            image = pipe(prompt=prompt, generator=generator).images[0]
            image.save(f"{folder_name}/{prompt_name.replace(' ', '_')}_{i}.png")


def main():
    """
    Main function to initiate the image generation process.
    """
    world_size = N_SIZE
    batch_size = N_SIZE
    image_path = IMAGE_PATH

    # Spawn processes for inference
    mp.spawn(
        run_inference,
        args=(world_size, batch_size, image_path),
        nprocs=world_size,
        join=True,
    )


if __name__ == "__main__":
    main()

end = datetime.now()
print("END: ", end)
print(f"TOTAL TIME: {end - start}")

print("Done!")
