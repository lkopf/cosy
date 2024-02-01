import os
import torch
from tqdm import tqdm
from torchvision.models import resnet18, ResNet18_Weights

from utils import transforms,ImageDataset

torch.cuda.empty_cache()

IMAGE_PATH = "/mnt/beegfs/share/atbstaff/ImageNet_1k/ILSVRC/Data/CLS-LOC/val/"
RESULT_PATH = "./activations/"
os.makedirs(RESULT_PATH, exist_ok=True)

N_NEURONS = 512 # 1000
MODEL_NAME = "A50k_resnet18-layer4" # "A50k_resnet18-fc"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
print("Collect activations...")

dataset = ImageDataset(root=IMAGE_PATH,
                            transform=transforms,
                            )

testloader = torch.utils.data.DataLoader(dataset,
                                          batch_size=256,
                                          shuffle=False,
                                          num_workers=2,
                                          )

model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1).to(device)
model.eval()

activation = {}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.data
    return hook

model.layer4.register_forward_hook(get_activation('layer4'))
# model.fc.register_forward_hook(get_activation('fc'))

MODEL_FEATURES = torch.zeros([len(dataset), N_NEURONS])

counter = 0
flag = True
with torch.no_grad():
    for i, x in tqdm(enumerate(testloader)):
        x = x.float().data.to(device)

        outputs = model(x).data
        if flag:
            print(activation['layer4'].shape)
            # print(activation['fc'].shape)
            flag = False

        MODEL_FEATURES[counter:counter + x.shape[0],:] = activation['layer4'].mean(axis =[2,3]).data.to(device)
        # MODEL_FEATURES[counter:counter + x.shape[0],:] = outputs
        counter += x.shape[0]

torch.save(MODEL_FEATURES, f"{RESULT_PATH}{MODEL_NAME}.pt")

print("Done!")