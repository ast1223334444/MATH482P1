import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms, models

# --- Configuration ---
# Update these paths based on your screenshots
IMGS_VECS_DIR = './imgs_and_vecs'  # Directory containing seedXXXX.png and seedXXXX.npz
CLASSIFIER_PATH = './celeba_resnet18_whole.pth'  # Path to your trained classifier
TEST_IMAGES_PATH = "./imgs_and_vecs"
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
device = DEVICE
sys.path.append('./stylegan2-ada-pytorch')
import torch_utils

def load_classifier(path):
    print(f"Loading model from {path}...")

    model = models.resnet18(pretrained=False)
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_ftrs, 40),
        nn.Sigmoid()
    )
    # Attempt to load the entire model
    # If this fails, it might be a state_dict (weights only), see "Troubleshooting" below
    model_weights = torch.load(path, map_location=device)
    model.load_state_dict(model_weights)


    model.to(device)
    model.eval()  # Set to evaluation mode (important for BatchNormalization/Dropout)
    return model

def predict_image(image_path, model):
    # 2. Define Image Preprocessing
    # These must match exactly what you used during training!
    # Standard ResNet preprocessing usually involves:
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),  # ResNet expects 224x224
        transforms.ToTensor(),
        # Standard ImageNet normalization (common for ResNet transfer learning)
        # If you didn't use this during training, remove this line.
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # 3. Load and Transform Image
    image = Image.open(image_path).convert('RGB')
    image_tensor = preprocess(image)

    # Add batch dimension (Shape becomes [1, 3, 224, 224])
    image_tensor = image_tensor.unsqueeze(0).to(device)

    # 4. Inference
    with torch.no_grad():
        output = model(image_tensor)

        # 5. Interpret Result
        # Case A: If your model outputs 1 raw number (logits)
        # > 0 usually means Positive Class (e.g., Smiling), < 0 means Negative
        #print(output)
        #score = output.item()

        # Case B: If your model outputs 2 numbers (probabilities for class 0 and 1)
        # prob = torch.softmax(output, dim=1)
        # score = prob[0][1].item() # Probability of class 1
        # prediction = "Positive" if score > 0.5 else "Negative"

        ret = np.asarray(output.cpu())
        return ret[0]

if __name__ == "__main__":
    classifier = load_classifier(CLASSIFIER_PATH)
    columns = [20, 39, 15]  # glasses
    column_names = ["Male", "Young", "EyeGlasses"]
    predictions = [[] for _ in range(len(columns))]
    file_names = []
    z_vals = []
    w_vals = []
    i = 0
    for file in os.listdir(IMGS_VECS_DIR):
        if file.endswith(".png"):
            if i % 100 == 0 and i > 0:
                print(f"at {i}th iteration")
            i += 1
            raw_score = predict_image(os.path.join(TEST_IMAGES_PATH, file), classifier)
            for j in range(len(columns)):
                predictions[j].append(int(raw_score[columns[j]] > 0.5))
            file_names.append(file.replace(".png", ""))
            npz_file = file.replace("png", "npz")
            npz_data = np.load(os.path.join(TEST_IMAGES_PATH, npz_file))
            z_vals.append(npz_data["z"])
            w_vals.append(npz_data["w"][:, 0, :])

    new_df = {"file_name": file_names}
    for column_name, prediction in zip(column_names, predictions):
        new_df[column_name] = prediction

    new_df = pd.DataFrame(new_df)
    new_df.to_csv("image_attribute_classifications.csv", index=False)

