import torch 
import torch.nn as nn
from models.models_v2l import VQModel_LLaMA
class Classifier(nn.Module):
    def __init__(self, args):
        super(Classifier, self).__init__()
        # self.classifier = nn.Sequential(
        #     nn.Linear(4096, 1024),
        #     nn.ReLU(),
        #     nn.Linear(1024, args.class_num)
        # )
        self.classifier = nn.Linear(4096, args.class_num)

    def forward(self, x):
        return self.classifier(x)