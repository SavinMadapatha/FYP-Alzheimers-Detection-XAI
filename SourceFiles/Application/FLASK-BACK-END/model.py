from email.mime import image
import os
import torch
import torch.nn as nn
import numpy as np
import torchvision.transforms as transforms
from PIL import Image, ImageFilter, ImageOps
import torchvision.models as models
import torch.nn.functional as F
from torchvision.utils import save_image
from gradcam.utils import visualize_cam
from gradcam import GradCAM
from torchvision.transforms.functional import to_pil_image
from torchvision.transforms import Resize, CenterCrop, ToTensor, Normalize, GaussianBlur, Grayscale
from flask import current_app
import cv2
from lime import lime_image
from skimage.segmentation import mark_boundaries
from skimage.color import gray2rgb, label2rgb
from skimage import img_as_ubyte
from skimage import img_as_float
import shap
from captum.attr import visualization as viz
import matplotlib
matplotlib.use('Agg')  
import matplotlib.pyplot as plt

class ContrastStretching:
    def __call__(self, img):
        img_np = np.array(img)
        
        if img_np.ndim == 2:  
            img_np = self.apply_contrast_stretching(img_np)
        elif img_np.ndim == 3:  
            for i in range(img_np.shape[-1]):
                img_np[:, :, i] = self.apply_contrast_stretching(img_np[:, :, i])
        
        return Image.fromarray(img_np.astype('uint8'))
    
    def apply_contrast_stretching(self, channel):
        in_min, in_max = np.percentile(channel, (0, 100))
        out_min, out_max = 0, 255
        channel = np.clip((channel - in_min) * (out_max - out_min) / (in_max - in_min) + out_min, out_min, out_max)
        return channel

class UnsharpMask:
    def __init__(self, radius=1, percent=100, threshold=3):
        self.radius = radius
        self.percent = percent
        self.threshold = threshold

    def __call__(self, img):
        return img.filter(ImageFilter.UnsharpMask(radius=self.radius, 
                                                  percent=self.percent, 
                                                  threshold=self.threshold
                                                 ))

class GaussianBlur:
    def __init__(self, kernel_size, sigma=(0.1, 2.0)):
        self.kernel_size = kernel_size
        self.sigma = sigma

    def __call__(self, img):
        sigma = np.random.uniform(self.sigma[0], self.sigma[1])
        img = img.filter(ImageFilter.GaussianBlur(sigma))
        return img
    
# preprocessing techniques 
preprocess_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    ContrastStretching(),
    UnsharpMask(radius=1, percent=100, threshold=3),  
    GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 0.5)), 
    transforms.ToTensor()
])


# Preprocessing function
def preprocess_image(image_path):
    image = Image.open(image_path)

    if image.mode != 'RGB':
        image = image.convert('RGB')

    # Apply the preprocessing transforms
    image_tensor = preprocess_transform(image).unsqueeze(0)

    return image_tensor

# Modified AlexNet Architecture
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


# Modified EfficientNetB0 Architecture
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


# load the saved state dictionary of the optimal CNN1 (modified AlexNet)
model1 = CNN1(num_classes=4)
PATH1 = "model1_test2.pth"
model1.load_state_dict(torch.load(PATH1))

# load the saved state dictionary of the optimal CNN2 (modified EfficientNetB0)
model2 = CNN2(num_classes=4)
PATH2 = "model2_test1.pth"
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

        probs = prob1 + prob2 

        return probs


model1.eval()
model2.eval()

ensemble_model = VotingEnsemble(model1, model2)


def load_image(image_path):
    transform = transforms.Compose([
    transforms.Resize((224, 224)),
    ContrastStretching(),
    UnsharpMask(radius=1, percent=100, threshold=3),  
    GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 0.5)), 
    transforms.ToTensor()
])
    image = Image.open(image_path).convert('RGB')
    if transform is not None:
        image = transform(image).unsqueeze(0)
    return image

# this function returns the predicted AD stage with confidence percentage
def predict_with_confidence(image):
    ensemble_model.eval()
    with torch.no_grad():
        outputs = ensemble_model(image)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        predicted = torch.max(probabilities, 1)[1].item()
        confidence = torch.max(probabilities, 1)[0].item()
        return predicted, confidence

# Grad-CAM function for AlexNet
def apply_gradcam_AlexNet(model, image_tensor, target_layer, image_path):
    """
    generates the grad-cam output for the classification result of the AlexNet
    """
    try:
        gradcam = GradCAM(model, target_layer)
        mask, _ = gradcam(image_tensor)
        _, result = visualize_cam(mask, image_tensor)

        if isinstance(result, torch.Tensor):
            result_image = to_pil_image(result.cpu())

        filename = os.path.basename(image_path)
        cam_image_filename = filename.replace('.jpg', '_AlexNet_gradcam.jpg')
        cam_image_path = os.path.join(current_app.static_folder, cam_image_filename)
        result_image.save(cam_image_path)

        # Return a web-accessible URL
        return f'/mri_images/{cam_image_filename}'

    except Exception as e:
        print(f"Error applying Grad-CAM and saving: {e}")
        return None
    

# Grad-CAM function for EfficientNetB0
activations = {}
gradients = {}

def save_activation(name):
    def hook(module, input, output):
        activations[name] = output.detach()
    return hook

def save_gradient(name):
    def hook(module, grad_input, grad_output):
        gradients[name] = grad_output[0].detach()
    return hook

def apply_gradcam_efficientnet(model, image_tensor, target_layer_name, image_path):
    """
    Generates the grad-cam output for the classification result of the EfficientNetB0
    """
    model.eval()
    target_layer = dict([*model.base_model.features.named_children()])[target_layer_name]
    target_layer.register_forward_hook(save_activation(target_layer_name))
    target_layer.register_backward_hook(save_gradient(target_layer_name))

    output = model(image_tensor)
    pred_class_idx = output.argmax(dim=1).item()
    score = output[:, pred_class_idx]

    model.zero_grad()
    score.backward()

    gradient = gradients[target_layer_name].mean(dim=[2, 3], keepdim=True)
    weighted_activation = torch.mul(activations[target_layer_name], gradient).sum(dim=1, keepdim=True)
    relu_weighted_activation = torch.relu(weighted_activation)

    saliency_map = relu_weighted_activation.squeeze().cpu().detach().numpy()
    saliency_map = (saliency_map - np.min(saliency_map)) / (np.max(saliency_map) - np.min(saliency_map) + 1e-8)
    saliency_map = cv2.resize(saliency_map, (224, 224))
    heatmap = cv2.applyColorMap(np.uint8(255 * saliency_map), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    return finalize_and_save_image(image_tensor, heatmap, image_path, alpha=0.5, suffix="_gradcam_efficientnet")


# helper function for the apply_gradcam_efficientnet method
def finalize_and_save_image(original_image_tensor, heatmap, image_path, alpha, suffix):
    """
    Blends the original image with the heatmap, converts to RGB, and saves the final image.
    """
    heatmap_pil = Image.fromarray(heatmap)
    original_image_pil = to_pil_image(original_image_tensor.squeeze(0).cpu())
    final_image = Image.blend(original_image_pil, heatmap_pil, alpha=alpha)
    final_image_rgb = final_image.convert('RGB')

    base, _ = os.path.splitext(os.path.basename(image_path))
    new_filename = f"{base}{suffix}.jpg"
    cam_image_path = os.path.join(current_app.static_folder, new_filename)
    final_image_rgb.save(cam_image_path)

    return f'/mri_images/{new_filename}'



# functions related to LIME technique 
def model_prediction(model, image_tensor):
    model.eval()  
    with torch.no_grad():
        logits = model(image_tensor)
        probabilities = torch.nn.functional.softmax(logits, dim=1)
        return probabilities.cpu().numpy()

def apply_highlight(np_image, mask, color=[0, 255, 0]):
    """Applies a mask to the image with the specified highlight color."""
    if np_image.ndim == 2 or np_image.shape[2] == 1:
        np_image = gray2rgb(np_image)

    np_image = img_as_ubyte(np_image)

    colored_mask = np.zeros_like(np_image)
    for i in range(3):  
        colored_mask[:,:,i] = np.where(mask == 1, color[i], 0)

    highlighted_image = np.where(colored_mask, colored_mask, np_image)
    return highlighted_image

def generate_lime_and_highlighted(image_path, model, model_name):
    # Initialize LIME Image Explainer
    explainer = lime_image.LimeImageExplainer()

    # Load and preprocess the image
    image = Image.open(image_path).convert('RGB')
    np_image = np.array(image) / 255.0 

    def batch_predict(images):
        image_tensors = torch.stack([preprocess_transform(Image.fromarray(img.astype(np.uint8))) for img in images])
        return model_prediction(model, image_tensors)

    explanation = explainer.explain_instance(np_image, batch_predict, top_labels=5, hide_color=0, num_samples=1000)

    temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=True, num_features=3, hide_rest=False)

    highlighted_image = apply_highlight(np_image, mask)

    boundaries_path = os.path.join(current_app.static_folder, f"{model_name}_lime_boundaries.jpg")
    plt.imsave(boundaries_path, img_as_ubyte(mark_boundaries(temp, mask)))

    highlighted_path = os.path.join(current_app.static_folder, f"{model_name}_lime_highlighted.jpg")
    plt.imsave(highlighted_path, img_as_ubyte(highlighted_image))

    return f'/mri_images/{os.path.basename(boundaries_path)}', f'/mri_images/{os.path.basename(highlighted_path)}'


def generate_saliency_map(model, image_tensor, image_path, model_name, target_class=None, alpha=0.5):
    model.eval()  # Set the model to evaluation mode
    image_tensor.requires_grad = True

    if target_class is None:
        output = model(image_tensor)
        target_class = output.argmax(dim=1).item()

    output = model(image_tensor)
    model.zero_grad()
    one_hot_output = torch.FloatTensor(1, output.size()[-1]).zero_()
    one_hot_output[0][target_class] = 1
    output.backward(gradient=one_hot_output)

    # Compute saliency map
    saliency = image_tensor.grad.data.abs().squeeze().cpu().numpy()
    saliency = np.max(saliency, axis=0)
    saliency = (saliency - np.min(saliency)) / (np.max(saliency) - np.min(saliency))
    saliency_colored = cv2.applyColorMap(np.uint8(255 * saliency), cv2.COLORMAP_HOT)

    # Convert saliency map to PIL image and resize to match the original image size
    saliency_img = Image.fromarray(saliency_colored).resize(image_tensor.squeeze(0).shape[1:][::-1], Image.LANCZOS)

    # Blend the original image with the saliency map
    original_img = to_pil_image(image_tensor.squeeze(0).cpu())
    blended_img = Image.blend(original_img.convert("RGBA"), saliency_img.convert("RGBA"), alpha=alpha)

    # Save the blended image
    output_filename = f"{os.path.splitext(os.path.basename(image_path))[0]}_{model_name}_saliency_highlighted.png"
    output_path = os.path.join(os.path.dirname(image_path), output_filename)
    blended_img.save(output_path)

    # Return the web-accessible path
    return f'/mri_images/{output_filename}'