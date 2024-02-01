import os
import cv2
import torchvision
from torch.utils.data import Dataset

class ImageDataset(Dataset):
    def __init__(self,root,transform):#,image_format):
        self.root=root
        self.transform=transform
        #self.image_format=image_format

        #self.image_names=glob(self.root + f'*.{str(image_format)}')
        self.image_names=[self.root + x for x in os.listdir(self.root)]
        self.image_names.sort()
   
    def __len__(self):
        return len(self.image_names)
 
    def __getitem__(self,index):
        image=cv2.imread(self.image_names[index])
        image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)

        image=self.transform(image)

        return image


transforms = torchvision.transforms.Compose([
                           torchvision.transforms.ToPILImage(),
                           torchvision.transforms.Resize(224),
                           torchvision.transforms.CenterCrop((224,224)),
                           torchvision.transforms.ToTensor(),
                           torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                            std=[0.229, 0.224, 0.225])
                       ])