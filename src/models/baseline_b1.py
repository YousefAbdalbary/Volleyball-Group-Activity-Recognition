"""
Baseline B1-tuned
● Don’t try anything that doesn’t finetune (well-proved idea)
● In CVPR paper I used alexnet. You better network (e.g. resnet50)
● For each clip, use the middle image only
○ Fee free to use 5 before and 4 after also
● Fine-tune an image classifier over 8 classes
● Compute the results. This is your first model
"""

import torch.nn as nn
import torchvision.models as models
import torch


class BaselineB1ResNet(nn.Module):
    def __init__(self, num_classes=8, fine_tune_all=True):
        super(BaselineB1ResNet, self).__init__()

        # Load the pre-trained ResNet50
        self.model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

        # Optional: Freeze the backbone
        if not fine_tune_all:
            for param in self.model.parameters():
                param.requires_grad = False

        # Get the number of input features going into the 'fc' layer
        num_ftrs = self.model.fc.in_features

        # Replace the 1000-class layer with an 8-class layer
        self.model.fc = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        return self.model(x)


if __name__ == "__main__":
    model = BaselineB1ResNet()
    dummy_input = torch.randn(1, 3, 224, 224)
    output = model(dummy_input)
    print(output.shape)
