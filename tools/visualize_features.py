import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
import yaml
from collections import OrderedDict
import os

from hat.archs.hat_rcf_arch import HAT_RCF

# --- Global variables to store activations ---
# Using a dictionary is slightly cleaner for matching names to activations
activations_dict = OrderedDict()

def get_activation(name):
    """Hook to store the output of a layer in a dictionary."""
    def hook(model, input, output):
        activations_dict[name] = output.detach().cpu()
    return hook

def load_model(model_path, config_path):
    """Loads the model from a checkpoint and a config file."""
    with open(config_path, 'r') as f:
        opt = yaml.safe_load(f)['network_g']

    model = HAT_RCF(
        img_size=opt.get('img_size', 64), patch_size=opt.get('patch_size', 1),
        in_chans=opt.get('in_chans', 3), embed_dim=opt.get('embed_dim', 180),
        depths=opt.get('depths', [6, 6, 6, 6, 6, 6]), num_heads=opt.get('num_heads', [6, 6, 6, 6, 6, 6]),
        window_size=opt.get('window_size', 16), compress_ratio=opt.get('compress_ratio', 3),
        squeeze_factor=opt.get('squeeze_factor', 30), conv_scale=opt.get('conv_scale', 0.01),
        overlap_ratio=opt.get('overlap_ratio', 0.5), mlp_ratio=opt.get('mlp_ratio', 2),
        qkv_bias=opt.get('qkv_bias', True), qk_scale=opt.get('qk_scale', None),
        drop_rate=opt.get('drop_rate', 0.0), attn_drop_rate=opt.get('attn_drop_rate', 0.0),
        drop_path_rate=opt.get('drop_path_rate', 0.1), norm_layer=torch.nn.LayerNorm,
        ape=opt.get('ape', False), patch_norm=opt.get('patch_norm', True),
        use_checkpoint=opt.get('use_checkpoint', False), upscale=opt.get('upscale', 4),
        img_range=opt.get('img_range', 1.0), upsampler=opt.get('upsampler', 'pixelshuffle'),
        resi_connection=opt.get('resi_connection', '1conv'), rcf_pretrained_path=opt.get('rcf_pretrained_path'),
        rcf_layer_norm=opt.get('rcf_layer_norm', False)
    )

    load_net = torch.load(model_path, map_location=torch.device('cpu'))

    param_key = None
    if 'params' in load_net:
        param_key = 'params'
    elif 'params_ema' in load_net:
        param_key = 'params_ema'

    if param_key:
        model.load_state_dict(load_net[param_key], strict=True)
    else:
        model.load_state_dict(load_net, strict=True)

    print(f"Model loaded from {model_path}")
    return model, opt.get('window_size', 16)

def preprocess_image(image_path, window_size):
    """Loads and preprocesses an image for the model."""
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img = img.astype(np.float32) / 255.

    h, w, c = img.shape

    # Calculate the padding needed
    pad_h = (window_size - h % window_size) % window_size
    pad_w = (window_size - w % window_size) % window_size
    
    img_padded = np.pad(img, ((0, pad_h), (0, pad_w), (0, 0)), 'reflect')

    print(f"Original image size: {h}x{w}")
    print(f"Padded image size: {img_padded.shape[0]}x{img_padded.shape[1]}")

    img_tensor = torch.from_numpy(img_padded).permute(2, 0, 1).unsqueeze(0)
    return img_tensor


# --- IMPROVED: Centralized visualization logic ---
def process_and_save_activations(output_dir):
    """Processes and saves the captured activations."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    print(f"\n--- Visualizing {len(activations_dict)} activation maps ---")

    for layer_name, feat in activations_dict.items():
        if feat.dim() == 3:
            b, seq_len, c = feat.shape
            
            side_length = int(seq_len ** 0.5)

            if side_length * side_length != seq_len:
                print(f"Skipping '{layer_name}' because its sequence length {seq_len} is not a perfect square.")
                continue
            h = w = side_length

            if seq_len == h * w:
                feat = feat.permute(0, 2, 1).view(b, c, h, w)
            else:
                print(f"Skipping '{layer_name}' due to shape mismatch in sequence length ({seq_len} vs {h*w}).")
                continue

        # Take the mean over the channel dimension to get a 2D map
        vis_map = torch.mean(feat.squeeze(0), dim=0)

        # Normalize to 0-1 range for visualization
        min_val = vis_map.min()
        max_val = vis_map.max()
        
        # --- IMPROVED: Added epsilon for safe division ---
        denominator = max_val - min_val
        vis_map = (vis_map - min_val) / (denominator + 1e-6)

        # Save the map
        save_path = os.path.join(output_dir, f"{layer_name}.png")
        plt.imsave(save_path, vis_map.numpy(), cmap='viridis')
        print(f"Saved feature map for '{layer_name}' to {save_path}")


if __name__ == '__main__':
    # --- USER: Please update these paths ---
    model_path = '' 
    config_path = ''
    image_path = ''
    output_dir = 'figs/hat_feature_maps'
    # -----------------------------------------

    # 1. Load the model and set to evaluation mode
    model, window_size = load_model(model_path, config_path)
    model.eval()

    # 2. Register hooks to capture activations
    model.conv_first.register_forward_hook(get_activation('conv_first'))
    for i, layer in enumerate(model.layers):
        layer.register_forward_hook(get_activation(f'RHAG_{i+1}_output'))
    model.norm.register_forward_hook(get_activation('body_norm_output'))

    # 3. Load and preprocess the image
    img_tensor = preprocess_image(image_path, window_size)

    # 4. Perform inference to trigger hooks
    with torch.no_grad():
        _ = model(img_tensor)

    # 5. Process and save the captured activations
    # Pass the model's patch resolution to the function for reshaping
    process_and_save_activations(output_dir)