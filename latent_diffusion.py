import os
import pickle
import sys

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

import torch
from matplotlib import pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

sys.path.append('./stylegan2-ada-pytorch')

import torch_utils

class TimeEmbedding(nn.Module):
    """Embeds scalar timesteps into a vector."""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.linear_1 = nn.Linear(dim, dim)
        self.linear_2 = nn.Linear(dim, dim)

    def forward(self, t):
        # Sinusoidal embedding or simple projection
        # t: (batch_size, 1)
        half_dim = self.dim // 2
        emb = np.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t[:, 0, None] * emb[None, :]
        emb = torch.cat((torch.sin(emb), torch.cos(emb)), dim=1)
        return self.linear_2(F.silu(self.linear_1(emb)))


class LatentDiffusionModel(nn.Module):
    def __init__(self, w_dim=512, num_classes=3, hidden_dim=1024):
        super().__init__()
        # 1. Condition Embedding (Attributes like 'Glasses', 'Gender')
        self.class_emb = nn.Embedding(num_classes, hidden_dim)

        # 2. Time Embedding
        self.time_emb = TimeEmbedding(hidden_dim)

        # 3. Main MLP Backbone (Residual Blocks recommended in PDF Hint 1)
        self.input_proj = nn.Linear(w_dim, hidden_dim)

        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim)
            ) for _ in range(4)  # 4 Residual Blocks
        ])

        self.output_proj = nn.Linear(hidden_dim, w_dim)

    def forward(self, x, t, class_labels=None, dropout_prob=0.1):
        """
        x: Noisy w vector (Batch, 512)
        t: Timestep (Batch, 1)
        class_labels: Target attribute (Batch)
        """
        # Embed Inputs
        t_emb = self.time_emb(t)
        h = self.input_proj(x) + t_emb

        # Classifier-Free Guidance Training Logic
        if class_labels is not None:
            c_emb = self.class_emb(class_labels)

            # Randomly drop labels (CFG training trick)
            if self.training and torch.rand(1).item() < dropout_prob:
                c_emb = torch.zeros_like(c_emb)  # Null condition

            h = h + c_emb

        # Residual MLP Pass
        for layer in self.layers:
            h = h + layer(h)  # Residual connection

        return self.output_proj(h)  # Predicts the noise epsilon


class DiffusionTrainer:
    def __init__(self, model, T=1000, beta_start=1e-4, beta_end=0.02, device='cuda'):
        self.model = model.to(device)
        self.T = T
        self.device = device

        # Define Schedule
        self.betas = torch.linspace(beta_start, beta_end, T).to(device)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

    def add_noise(self, x_0, t):
        """Forward process: q(x_t | x_0)"""
        noise = torch.randn_like(x_0)
        sqrt_alpha_bar = torch.sqrt(self.alphas_cumprod[t])[:, None]
        sqrt_one_minus_alpha_bar = torch.sqrt(1 - self.alphas_cumprod[t])[:, None]

        x_t = sqrt_alpha_bar * x_0 + sqrt_one_minus_alpha_bar * noise
        return x_t, noise

    def train_step(self, w_batch, labels):
        """
        w_batch: Clean latent vectors from StyleGAN (Batch, 512)
        labels: Attribute labels from Classifier (Batch)
        """
        batch_size = w_batch.shape[0]

        # 1. Sample random timesteps
        t = torch.randint(0, self.T, (batch_size,), device=self.device).long()

        # 2. Add noise
        w_t, noise = self.add_noise(w_batch, t)

        # 3. Predict noise
        noise_pred = self.model(w_t, t.unsqueeze(1), labels)

        # 4. Loss
        loss = F.mse_loss(noise_pred, noise)
        return loss


@torch.no_grad()
def manipulate_latent(
        trainer,
        w_source,
        target_label,
        start_step=200,
        guidance_scale=2.0
):
    """
    SDE-Edit: Corrupt source -> Denoise towards target
    """
    model = trainer.model
    model.eval()

    # 1. Destruction: Add noise to source W up to start_step
    # Note: We repeat w_source for batch processing if needed
    t_start = torch.tensor([start_step], device=trainer.device)
    w_noisy, _ = trainer.add_noise(w_source, t_start)

    w_t = w_noisy

    # 2. Reconstruction: Denoise from start_step down to 0
    for t in reversed(range(0, start_step)):
        t_tensor = torch.tensor([t], device=trainer.device).unsqueeze(0).repeat(w_source.shape[0], 1)

        # A. Predict noise (Conditional)
        label_tensor = torch.tensor([target_label], device=trainer.device).repeat(w_source.shape[0])
        noise_cond = model(w_t, t_tensor, label_tensor)

        # B. Predict noise (Unconditional - Null Label)
        # Assuming label 0 or a specific token is 'null' or handled via masking in model
        # For simplicity here, we pass a random incorrect label or zeros if handled
        null_label = torch.ones_like(label_tensor)
        noise_uncond = model(w_t, t_tensor, null_label)

        # C. Classifier-Free Guidance
        # epsilon = epsilon_uncond + s * (epsilon_cond - epsilon_uncond)
        noise_pred = noise_uncond + guidance_scale * (noise_cond - noise_uncond)

        # D. Reverse Step (Langevin Dynamics / DDPM Step)
        beta_t = trainer.betas[t]
        alpha_t = trainer.alphas[t]
        alpha_bar_t = trainer.alphas_cumprod[t]

        # Standard DDPM update
        sigma_t = torch.sqrt(beta_t)
        z = torch.randn_like(w_t) if t > 0 else 0

        # w_{t-1} = 1/sqrt(alpha) * (w_t - (beta / sqrt(1-alpha_bar)) * noise) + sigma * z
        w_t = (1 / torch.sqrt(alpha_t)) * (w_t - (beta_t / torch.sqrt(1 - alpha_bar_t)) * noise_pred)
        w_t = w_t + sigma_t * z  # Don't forget the + sigma*z term

    return w_t  # This is your manipulated vector


def prepare_data(attr="Male", device='cuda'):

    all_ws = []
    all_labels = []

    label_df = pd.read_csv("./image_attribute_classifications.csv")
    with torch.no_grad():
        for i in range(len(label_df)):
            name = label_df.iloc[i]["file_name"]
            label = label_df.iloc[i][attr]
            w_val = np.load(os.path.join("imgs_and_vecs", name + ".npz"))["w"][0, 0]
            all_ws.append(w_val)
            all_labels.append((label * 2))

    all_ws = np.array(all_ws)
    all_labels = np.array(all_labels)
    data_w = torch.from_numpy(all_ws).float().to(device)
    data_y = torch.from_numpy(all_labels).long().to(device)

    print(f"Dataset Ready: {data_w.shape}, {data_y.shape}")
    return TensorDataset(data_w, data_y)

def train_diffusion(
        model,
        dataloader,
        num_epochs=50,
        lr=1e-4,
        device='cuda'
):
    # 1. Setup
    model = model.to(device)
    model.train()

    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Initialize the noise scheduler (Trainer class from previous step)
    diffusion = DiffusionTrainer(model, device=device)

    print("Starting Training...")

    loss_history = []

    for epoch in range(num_epochs):
        epoch_loss = 0.0

        # Wrap loader with tqdm for progress bar
        pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}")

        for batch_w, batch_labels in pbar:
            batch_w = batch_w.to(device)
            batch_labels = batch_labels.to(device)
            current_batch_size = batch_w.shape[0]

            # --- A. Input Preparation ---
            # Sample random timesteps t ~ Uniform(0, T)
            t = torch.randint(0, diffusion.T, (current_batch_size,), device=device).long()

            # --- B. Forward Process (Add Noise) ---
            # w_t = sqrt(alpha_bar)*w_0 + sqrt(1-alpha_bar)*epsilon
            w_noisy, noise_epsilon = diffusion.add_noise(batch_w, t)

            # --- C. Classifier-Free Guidance (CFG) Logic ---
            # Project Requirement: "randomly dropping labels 10% of the time"
            # We use a mask where 1 = keep label, 0 = drop label

            cfg_prob = 0.1
            mask = torch.rand(current_batch_size, device=device) > cfg_prob

            # If mask is False (dropped), we can pass a specific "null" label index
            # OR handle it inside the model (as done in the previous model code with zeros).
            # Here, we keep the labels but tell the model which ones to ignore via masking logic
            # if your model supports it.

            # Assuming the model handles dropouts internally or we pass a null index:
            # Let's assume num_classes is the index for "null" (unconditional)
            # if batch_labels has classes 0 and 1, we set dropped ones to 2.

            labels_in = batch_labels.clone()
            # If you expanded embeddings to num_classes + 1, use this:
            labels_in[~mask] = 1

            # Alternatively, if using the model from previous turn which zeros out embedding:
            # The previous model had `dropout_prob` in forward(). We can pass it there.

            # --- D. Prediction & Loss ---
            optimizer.zero_grad()

            # Predict noise
            # Note: We pass dropout_prob=0.1 explicitly to the model
            # effectively fulfilling the CFG requirement during training.
            noise_pred = diffusion.model(w_noisy, t.unsqueeze(1), labels_in, dropout_prob=0.1)

            # Loss: MSE between actual noise and predicted noise
            # L = || epsilon - epsilon_theta ||^2
            loss = torch.nn.functional.mse_loss(noise_pred, noise_epsilon)

            # --- E. Backprop ---
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            pbar.set_postfix({'Loss': loss.item()})

        avg_loss = epoch_loss / len(dataloader)
        loss_history.append(avg_loss)
        print(f"Epoch {epoch + 1} Average Loss: {avg_loss:.6f}")

    print("Training Complete.")
    return diffusion, loss_history

# --- RUNNING THE LOOP ---
# 1. Instantiate Model
# diffusion_model = LatentDiffusionModel(w_dim=512, num_classes=2).to('cuda')

# 2. Create DataLoader (from Step 1)
# train_loader = DataLoader(dataset, batch_size=64, shuffle=True)

# 3. Train
# trained_model, history = train_diffusion(diffusion_model, train_loader)

if __name__ == "__main__":
    dataset = prepare_data()

    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

    diffusion_model = LatentDiffusionModel(w_dim=512, num_classes=3).to('cuda')

    trainer, history = train_diffusion(diffusion_model, dataloader)

    batch_w, batch_labels = next(iter(dataloader))

    for i in range(10):
        w = batch_w[i]
        labels = batch_labels[i]

        w_diff = manipulate_latent(trainer, w[None, :], 2-labels)
        print(w.shape)
        print(w_diff.shape)

        with open("/home/ahmet/PycharmProjects/MATH482P1/ffhq.pkl", 'rb') as f:
            network_dict = pickle.load(f)
            G = network_dict['G_ema'].to(trainer.device)

        # Optional: Put into eval mode
        G.eval()

        img_tensor_orig = G.synthesis(w.view(1, 1, 512).repeat(1, 18, 1), noise_mode='const', force_fp32=True)
        img_tensor_orig = (img_tensor_orig.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
        img_orig_np = img_tensor_orig[0].cpu().numpy()

        img_tensor_manip = G.synthesis(w_diff.view(1, 1, 512).repeat(1, 18, 1), noise_mode='const', force_fp32=True)
        img_tensor_manip = (img_tensor_manip.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
        img_manip_np = img_tensor_manip[0].cpu().numpy()

        ig, axes = plt.subplots(1, 2, figsize=(10, 5))

        axes[0].imshow(img_orig_np)
        axes[0].set_title("original image")
        axes[0].axis('off')

        axes[1].imshow(img_manip_np)
        axes[1].set_title("manipulated image")
        axes[1].axis('off')

    plt.show()

    a = input("Press enter to exit")