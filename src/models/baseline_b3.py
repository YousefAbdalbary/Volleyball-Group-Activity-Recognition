import torch
import torch.nn as nn
import torchvision.models as models


class BaselineB3PersonModel(nn.Module):
    """
    PHASE A: The 'Scout'
    Looks at a single cropped player and predicts 1 of 9 individual actions.
    """

    def __init__(self, num_classes=9, fine_tune_all=True):
        super(BaselineB3PersonModel, self).__init__()

        # Load the pre-trained ResNet50
        self.model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

        # Optional: Freeze the backbone
        if not fine_tune_all:
            for param in self.model.parameters():
                param.requires_grad = False

        # Get the number of input features going into the 'fc' layer (2048)
        num_ftrs = self.model.fc.in_features

        # Replace the final layer to output 9 classes
        self.model.fc = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        return self.model(x)


class BaselineB3GroupModel(nn.Module):
    """
    PHASE C: The 'Head Coach'
    Takes the max-pooled 2048 feature vector of the whole team
    and predicts 1 of 8 group activities.
    """

    def __init__(self, input_features=2048, num_classes=8, dropout_rate=0.5):
        super(BaselineB3GroupModel, self).__init__()

        self.classifier = nn.Sequential(
            nn.Linear(input_features, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        return self.classifier(x)


if __name__ == "__main__":
    model = BaselineB3PersonModel()
    dummy_input = torch.randn(1, 3, 224, 224)
    output = model(dummy_input)
    print(output.shape)

    model = BaselineB3GroupModel()
    dummy_input = torch.randn(1, 2048)
    output = model(dummy_input)
    print(output.shape)
