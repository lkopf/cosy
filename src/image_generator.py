"""
torchrun src/image_generator.py --nproc_per_node=3
"""
import os
import random
import torch
from datetime import datetime
import torch.distributed as dist
import torch.multiprocessing as mp
from diffusers import DiffusionPipeline

from utils import load_explanations

torch.cuda.empty_cache()

random.seed(42)

start = datetime.now()
print("START: ", start)

IMAGE_PATH = "./gen_images/"
os.makedirs(IMAGE_PATH, exist_ok=True)

# Load Explanations
METHOD = (# "MILAN"
          # "INVERT"
          "CLIP-Dissect"
          # "FALCON"
          )
print(METHOD)
MODEL_NAME = ("resnet18-avgpool"
              # "vit16b-ln"
            )
print(MODEL_NAME)
if MODEL_NAME == "resnet18-fc":
    N_NEURONS = 1000
elif MODEL_NAME == "resnet18-layer4" or MODEL_NAME == "resnet18-avgpool":
    N_NEURONS = 512
elif MODEL_NAME == "vit16b-ln":
    N_NEURONS = 768

N_NEURONS_RANDOM = 50
NEURON_IDS = random.sample(range(N_NEURONS), N_NEURONS_RANDOM)
EXPLANATION_PATH = f"./assets/explanations/{METHOD}/{MODEL_NAME}.csv" # path to METHOD csv file with neuron explanations
_, EXPLANATIONS = load_explanations(path=EXPLANATION_PATH, name=METHOD, image_path=IMAGE_PATH, neuron_ids=NEURON_IDS)
print("# EXPLANATIONS", len(EXPLANATIONS))

N_SIZE = 3 # world and batch size
N_IMAGES = 50 # number of generated images

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

# Load stable diffusion model
model_id = "stabilityai/stable-diffusion-xl-base-1.0"
pipe = DiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16, use_safetensors=True, variant="fp16")
pipe = pipe.to("cuda")
# set seed for stable diffusion
generator = torch.Generator("cuda").manual_seed(0)   

def run_inference(rank, world_size, batch_size, image_path=IMAGE_PATH, num_images=N_IMAGES):
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
                prompt = f"realistic photo of a close up of {current_batch[0]}"
                prompt_name = current_batch[0]
            elif torch.distributed.get_rank() == 1 and len(current_batch) > 0:
                prompt = f"realistic photo of a close up of {current_batch[1]}"
                prompt_name = current_batch[1]
            elif torch.distributed.get_rank() == 2 and len(current_batch) > 1:
                prompt = f"realistic photo of a close up of {current_batch[2]}" 
                prompt_name = current_batch[2]
            
            folder_name = image_path+f"{prompt_name.replace(' ', '_')}"
            os.makedirs(folder_name, exist_ok=True)
                
            image = pipe(prompt=prompt, generator=generator).images[0]
            image.save(f"{folder_name}/{prompt_name.replace(' ', '_')}_{i}.png")

def main():
    world_size = N_SIZE  # Number of processes
    batch_size = N_SIZE
    image_path = IMAGE_PATH

    # Spawn processes for inference
    mp.spawn(run_inference, args=(world_size, batch_size, image_path), nprocs=world_size, join=True)

if __name__ == "__main__":
    main()

end = datetime.now()
print("END: ", end)
print(f"TOTAL TIME: {end - start}")

print("Done!")
