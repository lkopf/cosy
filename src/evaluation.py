import os
import torch
import random
from tqdm import tqdm
from datetime import datetime
from sklearn.metrics import roc_auc_score
from scipy.stats import pointbiserialr,mannwhitneyu
from torchvision.models import resnet18, ResNet18_Weights

from utils import *

torch.cuda.empty_cache()

start = datetime.now()
print("START: ", start)

METHOD = "INVERT"
MODEL_NAME = "A50k_resnet18-fc" # "A50k_resnet18-layer4"
RESULT_PATH = "./results/"
os.makedirs(RESULT_PATH, exist_ok=True)
EXPLANATION_PATH = # path to METHOD csv file with neuron explanations
ACTIVATION_PATH = "./activations/"
IMAGE_PATH = "./gen_images/"

# Select n random neurons
N_NEURONS  = 512 # 1000 # neurons in selected layer
N_NEURONS_RANDOM = 50
NEURON_IDS = random.sample(range(N_NEURONS), N_NEURONS_RANDOM)

# Load Explanations
METHOD = "INVERT" # "MILAN" # "FALCON"
EXPLANATIONS, _ = load_explanations(path=EXPLANATION_PATH,name=METHOD,image_path=IMAGE_PATH,neuron_ids=NEURON_IDS)

print("Evaluate explanations...")

# load model
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1).to(device)
model.eval()

# Load activations for val dataset
A_F_val = torch.load(ACTIVATION_PATH+f"{MODEL_NAME}.pt").to(device)

for NEURON_ID, CONCEPT_NAME in tqdm(zip(NEURON_IDS, EXPLANATIONS), total=len(NEURON_IDS), desc="Processing"):
    NEURON_ID = int(NEURON_ID)
    concept_raw = CONCEPT_NAME
    concept = concept_raw.replace(" ", "_")
    CONCEPT_PATH = IMAGE_PATH+concept+"/"

    dataset = ImageDataset(root=CONCEPT_PATH,
                            transform=transforms,
                            )
                            #image_format=FORMAT)

    dataloader = torch.utils.data.DataLoader(dataset,
                                            batch_size=2,
                                            shuffle=False,
                                            num_workers=2)

    csv_filename = RESULT_PATH+f"auc_p_rpb_{METHOD}_{MODEL_NAME}.csv"
    csv_headers = ["neuron", "concept", "auc", "U1", "p", "rpb_r", "rpb_p"]

    if not csv_file_exists(csv_filename):
        create_csv(csv_filename, csv_headers)

    TENSOR_PATH = ACTIVATION_PATH + MODEL_NAME + f"_neuron-{NEURON_ID}_{concept}.pt"

    if os.path.exists(TENSOR_PATH):
        A_F = torch.load(TENSOR_PATH)
    else:
        # collect activations
        activation = {}
        def get_activation(name):
            def hook(model, input, output):
                activation[name] = output
            return hook

        model.layer4.register_forward_hook(get_activation("layer4"))
        # model.fc.register_forward_hook(get_activation('fc'))

        # Save activations for new concept
        A_F = torch.zeros([len(dataset), N_NEURONS]).to(device)

        counter = 0
        flag = True
        with torch.no_grad():
            for i, x in tqdm(enumerate(dataloader)):
                x = x.float().data.to(device)

                outputs = model(x).data
                if flag:
                    print(activation['layer4'].shape)
                    #print(activation['fc'].shape)
                    flag = False

                A_F[counter:counter + x.shape[0],:] = activation['layer4'].mean(axis =[2,3]).data.to(device)
                # A_F[counter:counter + x.shape[0],:] = outputs
                counter += x.shape[0]

        torch.save(A_F, TENSOR_PATH)

    # Get binary classification scores for neuron explanation
    activ_non_class = A_F_val[:,NEURON_ID] # all activations for val dataset
    # Construct dataset D with non-class activations and synthetic class activations
    activ_class = A_F[:,NEURON_ID]
    A_D = torch.cat((activ_non_class, activ_class), 0) # 
    class_labels = torch.cat((torch.zeros([activ_non_class.shape[0]]), torch.ones([activ_class.shape[0]])), 0)
    # Binary classification scores
    auc_synthetic = roc_auc_score(class_labels.to("cpu"), A_D.to("cpu"))
    U1, p = mannwhitneyu(class_labels.to("cpu"), A_D.to("cpu"))
    rpb_r, rpb_p = pointbiserialr(class_labels.to("cpu"), A_D.to("cpu"))

    new_rows  = [[NEURON_ID, concept_raw, round(auc_synthetic, 4), round(U1, 4), round(p, 4), round(rpb_r, 4), round(rpb_p, 4)]]
    add_rows_to_csv(csv_filename, new_rows)

end = datetime.now()
print("END: ", end)
print(f"TOTAL TIME: {end - start}")

print("Done!")