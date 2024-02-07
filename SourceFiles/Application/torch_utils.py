from email.mime import image
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import torchvision.models as models
import torch.nn.functional as F

# AlexNet model (modified) | architecture modification (1st)


class CNN1(nn.Module):
    def __init__(self, num_classes=4):
        super(CNN1, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.conv4 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.conv5 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(256)
        self.conv6 = nn.Conv2d(256, 384, kernel_size=3, padding=1)
        self.bn6 = nn.BatchNorm2d(384)

        self.pool = nn.MaxPool2d(kernel_size=3, stride=2)

        self.fc1 = nn.Linear(384 * 6 * 6, 4096)
        self.fc2 = nn.Linear(4096, 1024)
        self.output = nn.Linear(1024, num_classes)

        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(F.leaky_relu(
            self.bn1(self.conv1(x)), negative_slope=0.01))
        x = self.pool(F.leaky_relu(
            self.bn2(self.conv2(x)), negative_slope=0.01))
        x = F.leaky_relu(self.bn3(self.conv3(x)), negative_slope=0.01)
        x = F.leaky_relu(self.bn4(self.conv4(x)), negative_slope=0.01)
        x = self.pool(F.leaky_relu(
            self.bn5(self.conv5(x)), negative_slope=0.01))
        x = F.leaky_relu(self.bn6(self.conv6(x)), negative_slope=0.01)

        x = x.view(x.size(0), -1)

        x = self.dropout(F.leaky_relu(self.fc1(x), negative_slope=0.01))
        x = self.dropout(F.leaky_relu(self.fc2(x), negative_slope=0.01))

        x = self.output(x)

        return x


class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction=32):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class CNN2(nn.Module):
    def __init__(self, num_classes=4):
        super(CNN2, self).__init__()
        self.base_model = models.efficientnet_b0(pretrained=True)

        for param in self.base_model.parameters():
            param.requires_grad = False

        num_ftrs = self.base_model.classifier[1].in_features
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(num_ftrs, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )

        self.base_model.features.add_module("SEBlock", SEBlock(1280))

    def forward(self, x):
        x = self.base_model.features(x)
        x = self.base_model.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

# model = ModifiedAlexNet(num_classes=4)


model1 = CNN1(num_classes=4)
PATH1 = "model1.pth"
model1.load_state_dict(torch.load(PATH1))

model2 = CNN2(num_classes=4)
PATH2 = "model2.pth"
model2.load_state_dict(torch.load(PATH2))


class VotingEnsemble(nn.Module):
    def __init__(self, model1, model2):
        super(VotingEnsemble, self).__init__()
        self.model1 = model1
        self.model2 = model2

    def forward(self, x):
        # Get predictions from model1 and model2
        output1 = self.model1(x)
        output2 = self.model2(x)

        # softmax to convert to probabilities
        prob1 = F.softmax(output1, dim=1)
        prob2 = F.softmax(output2, dim=1)

        # Average the probabilities
        avg_probs = (prob1 + prob2) / 2

        return avg_probs


model1.eval()
model2.eval()

ensemble_model = VotingEnsemble(model1, model2)


def load_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
            0.229, 0.224, 0.225])])
    image = Image.open(image_path).convert('RGB')
    if transform is not None:
        image = transform(image).unsqueeze(0)
    return image


def predict_with_confidence(image):
    model1.eval()
    with torch.no_grad():
        outputs = model1(image)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        predicted = torch.max(probabilities, 1)[1].item()
        confidence = torch.max(probabilities, 1)[0].item()
        return predicted, confidence
