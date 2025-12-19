import sys
import pickle
import torch
import numpy as np
from PIL import Image
import os
import warnings

# Filter out the specific StyleGAN2 compilation warnings
warnings.filterwarnings("ignore", message="Failed to build CUDA kernels")

# -------------------------------------------------------------------------
# SETUP: Add the repository to sys.path so pickle can find the classes.
# Change 'stylegan2-ada-pytorch' to the actual path of your cloned repo.
# -------------------------------------------------------------------------
sys.path.append('./stylegan2-ada-pytorch')

import torch_utils

def generate_image(seed, n_images, to_dir):
    device = torch.device('cuda')

    # 1. Load the Network
    # We use dnnlib's open_url to handle caching/downloading automatically.
    with open('./ffhq.pkl', 'rb') as f:
        # The pickle contains a dict; we want the 'G_ema' (Exponential Moving Average)
        # version for the best quality inference.
        network_dict = pickle.load(f)
        G = network_dict['G_ema'].to(device)


    for i in range(n_images):
        # 2. define the latent vector (z)
        # StyleGAN2 takes a (1, 512) vector of Gaussian noise.
        z = torch.from_numpy(np.random.RandomState(seed + i).randn(1, G.z_dim)).to(device)

        # 3. Define the class label (c)
        # FFHQ is unconditional, so we pass a zero-dimension tensor or None.
        # However, the model expects a specific format.
        c = torch.zeros([1, G.c_dim]).to(device)

        # 4. Run Inference
        # noise_mode='const' makes the output deterministic (fixed hair/texture noise).
        if i % 100 == 0:
            print(f'Generating image {i}...')
        img = G(z, c, truncation_psi=0.5, noise_mode='const')

        # 5. Convert Tensor to Image
        # Output is (N, C, H, W) in range [-1, 1]. We need (H, W, C) in range [0, 255].
        img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)

        # Move to CPU and numpy
        img_np = img[0].cpu().numpy()

        Image.fromarray(img_np, 'RGB').save(f"{i}.png")


def generate_and_save(base_seed, n_images, out_dir, network_pkl):
    # 0. Setup output directory and device
    os.makedirs(out_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    with open(network_pkl, 'rb') as f:
        network_dict = pickle.load(f)
        G = network_dict['G_ema'].to(device)

    # Optional: Put into eval mode
    G.eval()

    print(f'Generating {n_images} images to "{out_dir}"...')

    for i in range(n_images):
        # Calculate distinct seed for this iteration
        current_seed = base_seed + i

        # 2. Define the latent vector (z)
        # We recreate the RandomState per seed to ensure reproducibility per file
        z = torch.from_numpy(np.random.RandomState(current_seed).randn(1, G.z_dim)).to(device)

        # 3. Define the class label (c)
        c = torch.zeros([1, G.c_dim]).to(device)

        # 4. Run Inference
        # We add `return_latents=True` to get the 'w' vector back from the model
        w = G.mapping(z, c, truncation_psi=0.5, truncation_cutoff=8)
        img_tensor = G.synthesis(w, noise_mode='const', force_fp32=True)
        #img_tensor, w = G(z, c, truncation_psi=0.5, noise_mode='const', return_latents=True)

        # 5. Convert Tensor to Image
        # (N, C, H, W) -> (H, W, C)
        img_tensor = (img_tensor.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
        img_np = img_tensor[0].cpu().numpy()

        # 6. Save Data
        file_name = f"seed{current_seed:04d}"

        # Save Image
        Image.fromarray(img_np, 'RGB').save(f"{out_dir}/{file_name}.png")

        # Save Vectors (z and w)
        # Move tensors to CPU and convert to numpy before saving
        np.savez(
            f"{out_dir}/{file_name}.npz",
            z=z.cpu().numpy(),
            w=w.cpu().numpy()
        )

        if i % 50 == 0:
            print(f'Generated {i}/{n_images} (saved {file_name})')

    print("Done.")


if __name__ == "__main__":
    generate_and_save(
        base_seed=0,
        n_images=5000,
        out_dir="imgs_and_vecs",
        network_pkl="./ffhq.pkl"
    )
    # Generate an image with a specific seed
    #pil_image = generate_image(seed=42)

    # Save or Display
    #pil_image.save("generated_face.png")
    #print("Saved generated_face.png")