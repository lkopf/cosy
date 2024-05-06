import os
import torch
from tqdm import tqdm
from torchvision.models import resnet18, ResNet18_Weights, vit_b_16, ViT_B_16_Weights

from utils import *

torch.cuda.empty_cache()

IMAGE_PATH = "/path/to/images/"
RESULT_PATH = "./activations/"
os.makedirs(RESULT_PATH, exist_ok=True)

MODEL_NAME = (# "A50k-train_resnet18-fc"
              "A50k-train_resnet18-avgpool"    
              # "A50k-train_resnet18-layer4"
              # "A50k-train_vit16b-head"
              # "A50k-train_vit16b-layer11"
            )
print(MODEL_NAME)

if MODEL_NAME == "A50k-train_resnet18-fc" or MODEL_NAME == "A50k-train_vit16b-head":
    N_NEURONS = 1000
elif MODEL_NAME == "A50k-train_resnet18-layer4" or MODEL_NAME == "A50k-train_resnet18-avgpool":
    N_NEURONS = 512
elif MODEL_NAME == "A50k-train_vit16b-layer11":
    N_NEURONS = 768

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
print("Collect activations...")

dataset = ImageDataset(root=IMAGE_PATH,
                            transform=TRANSFORMS_IMGNT,
                            )

testloader = torch.utils.data.DataLoader(dataset,
                                          batch_size=256,
                                          shuffle=False,
                                          num_workers=2,
                                          )

if MODEL_NAME == "A50k-train_resnet18-fc" or MODEL_NAME == "A50k-train_resnet18-avgpool" or MODEL_NAME == "A50k-train_resnet18-layer4":
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1).to(device)
elif MODEL_NAME == "A50k-train_vit16b-head" or MODEL_NAME == "A50k-train_vit16b-layer11":
    model = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1).to(device)
model.eval()

activation = {}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.data
    return hook

if MODEL_NAME == "A50k-train_resnet18-fc":
    model.fc.register_forward_hook(get_activation('fc'))
elif MODEL_NAME == "A50k-train_resnet18-avgpool":
    model.avgpool.register_forward_hook(get_activation('avgpool'))
elif MODEL_NAME == "A50k-train_resnet18-layer4":
    model.layer4.register_forward_hook(get_activation('layer4'))
elif MODEL_NAME == "A50k-train_vit16b-head":
    model.heads.head.register_forward_hook(get_activation('head'))
elif MODEL_NAME == "A50k-train_vit16b-layer11":
    model.encoder.layers[11].register_forward_hook(get_activation('layer11'))

MODEL_FEATURES = torch.zeros([len(dataset), N_NEURONS])

counter = 0
flag = True
with torch.no_grad():
    for i, x in tqdm(enumerate(testloader)):
        x = x.float().data.to(device)

        outputs = model(x).data
        if flag:
            if MODEL_NAME == "A50k-train_resnet18-fc":
                print(activation['fc'].shape)
            elif MODEL_NAME == "A50k-train_resnet18-avgpool":
                print(activation['avgpool'].shape)
            elif MODEL_NAME == "A50k-train_resnet18-layer4":
                print(activation['layer4'].shape)
            elif MODEL_NAME == "A50k-train_vit16b-head":
                print(activation['head'].shape)
            elif MODEL_NAME == "A50k-train_vit16b-layer11":
                print(activation['layer11'].shape)
            flag = False


        if MODEL_NAME == "A50k-train_resnet18-fc" or MODEL_NAME == "A50k-train_vit16b-head":
            MODEL_FEATURES[counter:counter + x.shape[0],:] = outputs
        elif MODEL_NAME == "A50k-train_resnet18-avgpool":
            MODEL_FEATURES[counter:counter + x.shape[0],:] = activation['avgpool'][:, :, 0, 0].data.to(device)
        elif MODEL_NAME == "A50k-train_resnet18-layer4":
            MODEL_FEATURES[counter:counter + x.shape[0],:] = activation['layer4'].mean(axis =[2,3]).data.to(device)
        elif MODEL_NAME == "A50k-train_vit16b-layer11":
            MODEL_FEATURES[counter:counter + x.shape[0],:] = activation['layer11'][:,0,:].data.to(device)
        counter += x.shape[0]

torch.save(MODEL_FEATURES, f"{RESULT_PATH}/validation/{MODEL_NAME}.pt")

print("Done!")