import torch 
import torch.nn as nn
class Classifier(nn.Module):
    def __init__(self, args):
        super(Classifier, self).__init__()
        # self.classifier = nn.Sequential(
        #     nn.Linear(4096, 1024),
        #     nn.ReLU(),
        #     nn.Linear(1024, args.class_num)
        # )
        self.classifier = nn.Linear(2048, args.class_num)

    def forward(self, x):
        return self.classifier(x)
class codebook_projector(nn.Module):
    def __init__(self, args):
        super(codebook_projector, self).__init__()
        self.projector = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.SiLU(),
            nn.Linear(1024, args.embed_dim)
        )
        
    def forward(self, x):
        return self.projector(x)
        