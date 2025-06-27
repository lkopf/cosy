# LLM Activations
This folder contains the code to use a LLM to generate concepts for individual neurons. 

Future work:
- try out different prompts and see the effects on the image concept generation
- get the activations and concepts for different layers

### Step 1:
Retrieve the activations for the specified layer. Use the script `src/activation_collector.py`. This
script is part of the origina CoSy implementation. The activations size is [50000, 64].
- Note: for now I used the ImageNet validation set. Had some issues with creating a subset from the original one and wanted to proof that it works. 

### Step 2:
Using these activations, we want the neuron explanations. Given the top x neuron activations, the GPT API is prompted with the top x images that belong to those activations. To make this work, you need to have access to an OpenAI account with sufficient funds to prompt GPT. This will result in an explanation file that is similar to the files already present. 

A prerequisite is that the images have to be available for the model to use. This path can be set in the `src/LLM_activations/utils.py` with the global variable IMAGE_PATH.

The script can be run through the commmand line:
`python llm_generalisatition.py --activation_path ACTIVATIONS --top_x TOPX --layer LAYERNAME --apikey APIKEY

The top_x parameter does not show up in the name of the output file if top_x=1. Otherwise, the name will be adjusted with the top_x number. In future steps, take this into account when generating synthetic images or evaluating them. 

For now it is just testing with ResNet model. In the future, the option of --model can be added. 

### Step 3 and 4:
Explanations are generated in the previous step. These are in a similar format as the explanations from the other models. Now you can use the other scripts, such as `src/image_generator.py` and `src/evaluation.py` in a similar way as with the other models, by just selecting `GPT` as the model. 