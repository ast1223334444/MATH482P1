import os
import glob
import pickle
import sys

import numpy as np
import torch
import torch.nn as nn
from sklearn.svm import LinearSVC
from PIL import Image
from matplotlib import pyplot as plt
from tqdm import tqdm
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

def load_dataset(data_dir, classifier, device):
    """
    Loads Z/W vectors from .npz files and generates labels using the classifier on .png images.
    """
    print(f"Loading data from {data_dir}...")

    z_list = []
    w_list = []
    labels_list = []

    # Get list of all .npz files
    npz_files = glob.glob(os.path.join(data_dir, "seed*.npz"))

    # Sort to ensure matching order with images if needed, though glob is usually enough
    npz_files.sort()

    classifier.eval()
    classifier.to(device)

    # transform pipeline for classifier (Modify based on how you trained ResNet18)
    from torchvision import transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        # Add normalization if your classifier expects it, e.g.:
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    for npz_path in tqdm(npz_files):
        # 1. Load Vectors
        data = np.load(npz_path)
        # Assuming keys are 'z' and 'w' inside the npz
        if 'z' not in data or 'w' not in data:
            continue

        z = data['z']  # Shape usually (1, 512)
        w = data['w']  # Shape usually (1, 18, 512) or similar

        # Flatten W: Take the first layer (Global style) for SVM
        if w.ndim == 3:
            w_flat = w[0, 0, :]
        else:
            w_flat = w.flatten()  # Fallback

        # 2. Load Image & Generate Label
        # Construct image path from npz path (e.g., seed4999.npz -> seed4999.png)
        img_path = npz_path.replace('.npz', '.png')

        if not os.path.exists(img_path):
            continue

        try:
            img = Image.open(img_path).convert('RGB')
            img_tensor = transform(img).unsqueeze(0).to(device)

            with torch.no_grad():
                logits = classifier(img_tensor)
                # Assuming binary classification: Logits > 0 = Positive
                # Modify this if your classifier outputs probabilities or uses index 0/1
                label = 1 if logits.item() > 0 else 0

            z_list.append(z.flatten())  # Ensure 1D
            w_list.append(w_flat)
            labels_list.append(label)

        except Exception as e:
            print(f"Error processing {img_path}: {e}")

    return np.array(z_list), np.array(w_list), np.array(labels_list)


def get_mean_difference(vectors, labels):
    """Method 1: Mean Difference"""
    pos_vecs = vectors[labels == 1]
    neg_vecs = vectors[labels == 0]

    if len(pos_vecs) == 0 or len(neg_vecs) == 0:
        raise ValueError("One class has 0 samples. Cannot calculate mean difference.")

    direction = np.mean(pos_vecs, axis=0) - np.mean(neg_vecs, axis=0)
    return direction / np.linalg.norm(direction)


def get_svm_normal(vectors, labels):
    """Method 2: SVM Decision Boundary Normal"""
    clf = LinearSVC(max_iter=10000, dual='auto')
    clf.fit(vectors, labels)

    direction = clf.coef_.flatten()
    return direction / np.linalg.norm(direction), clf.score(vectors, labels)

def alpha_scaling_w(vector, alpha_vals, model_path, seed = 42):

    with open(model_path, 'rb') as f:
        network_dict = pickle.load(f)
        G = network_dict['G_ema'].to(device)

    # Optional: Put into eval mode
    G.eval()
    z_np = np.random.RandomState(seed).randn(1, G.z_dim)
    z = torch.from_numpy(z_np).to(device)
    c = torch.zeros([1, G.c_dim]).to(device)

    fig, axs = plt.subplots(1, len(alpha_vals))
    w = G.mapping(z, c, truncation_psi=0.5, truncation_cutoff=8)
    all_w_vals = torch.from_numpy(vector[None, :] * alpha_vals[:, None]).to(device) + w[:, 0, :]
    for i in range(len(alpha_vals)):
        w_vals = all_w_vals[i][None, :]
        w_vals = w_vals.unsqueeze(1).repeat(1, 18, 1)
        img_tensor = G.synthesis(w_vals, noise_mode='const', force_fp32=True)

        img_tensor = (img_tensor.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
        img_np = img_tensor[0].cpu().numpy()

        axs[i].imshow(img_np)
        axs[i].set_title(f"alpha (w): {alpha_vals[i]}")
    plt.show()

def alpha_scaling_z(vector, alpha_vals, model_path, seed = 42):
    with open(model_path, 'rb') as f:
        network_dict = pickle.load(f)
        G = network_dict['G_ema'].to(device)
    G.eval()
    z = np.random.RandomState(seed).randn(1, G.z_dim)
    all_z_vals = torch.from_numpy(vector[None, :] * alpha_vals[:, None] + z).to(device)
    c = torch.zeros([1, G.c_dim]).to(device)

    fig, axs = plt.subplots(1, len(alpha_vals))

    for i in range(len(alpha_vals)):
        w = G.mapping(all_z_vals[i][None, :], c, truncation_psi=0.5, truncation_cutoff=8)
        img_tensor = G.synthesis(w, noise_mode='const', force_fp32=True)

        img_tensor = (img_tensor.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
        img_np = img_tensor[0].cpu().numpy()

        axs[i].imshow(img_np)
        axs[i].set_title(f"alpha (z): {alpha_vals[i]}")
    plt.show()


# --- Main Execution ---

if __name__ == "__main__":
    # 1. Load Classifier Model structure (You need to define the class or load entire model)
    # Assuming 'celeba_resnet18_whole.pth' is a full model save
    classifier = load_classifier(CLASSIFIER_PATH)
    column = 15 # glasses
    predictions = []
    z_vals = []
    w_vals = []
    i = 0
    for file in os.listdir(IMGS_VECS_DIR):

        if file.endswith(".png"):
            if i % 100 == 0 and i > 0:
                print(f"at {i}th iteration")
                break

            i += 1
            raw_score = predict_image(os.path.join(TEST_IMAGES_PATH, file), classifier)
            predictions.append(int(raw_score[column] > 0.5))
            npz_file = file.replace("png", "npz")
            npz_data = np.load(os.path.join(TEST_IMAGES_PATH, npz_file))
            z_vals.append(npz_data["z"])
            w_vals.append(npz_data["w"][:, 0, :])

    predictions = np.array(predictions)
    z_vals = np.array(z_vals)
    w_vals = np.array(w_vals)

    z_vals = z_vals.squeeze(axis = 1)
    w_vals = w_vals.squeeze( axis = 1)

    mean_difference_z_vals = get_mean_difference(z_vals, predictions)
    mean_difference_w_vals = get_mean_difference(w_vals, predictions)

    svm_z_vals, _ = get_svm_normal(z_vals, predictions)
    svm_w_vals, _ = get_svm_normal(w_vals, predictions)


    model_path = "./ffhq.pkl"
    alpha_vals = np.array([0.2 * i for i in range(-10, 11, 1)])

    alpha_scaling_z(mean_difference_z_vals, alpha_vals, model_path, seed=42)
    alpha_scaling_z(svm_z_vals, alpha_vals, model_path, seed=42)

    alpha_scaling_w(mean_difference_w_vals, alpha_vals, model_path, seed=42)
    alpha_scaling_w(svm_w_vals, alpha_vals, model_path, seed=42)