from torchvision import models
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F


resnet = models.resnet18(pretrained=True)
for param in resnet.parameters():
    param.requires_grad = True


class NN(nn.Module):
    def __init__(self, net_pretrained):
        super().__init__()
        self.net_pretrained = net_pretrained
        self.fc1 = nn.Linear(1000, 1)
        
    def forward(self, x):
        x = F.relu(self.net_pretrained(x))
        output = torch.sigmoid(self.fc1(x))

        return output


model = NN(resnet)
model.load_state_dict(torch.load('rubbish_classifier/model4.pth', map_location=torch.device('cpu')))
model.eval()