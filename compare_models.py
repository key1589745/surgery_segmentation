
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from hydra import initialize, compose
from hydra.core.global_hydra import GlobalHydra
from hydra.utils import instantiate
from omegaconf import OmegaConf
import sys

# Add current directory to path just in case
sys.path.append(os.getcwd())

CLASS_COLORS = np.array(
    [
        [0, 0, 0],         # background
        [255, 0, 0],       # cystic_plate
        [255, 165, 0],     # calot_triangle
        [255, 255, 0],     # cystic_artery
        [0, 255, 0],       # cystic_duct
        [0, 0, 255],       # gallbladder
        [255, 255, 255],   # tool
        [0, 0, 0]          # ignore
    ],
    dtype=np.uint8,
)

def overlay_mask_prediction(image, target, prediction, alpha=0.4):
    """
    image: (3, H, W) tensor, normalized
    target: (H, W) tensor, integer class indices
    prediction: (H, W) tensor, integer class indices
    alpha: opacity of the mask
    """
    # Denormalize image
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    
    img = image.permute(1, 2, 0).cpu().numpy()
    img = std * img + mean
    img = np.clip(img, 0, 1)

    # Prepare target mask
    tgt = target.cpu().numpy()
    tgt_viz = tgt.copy()
    tgt_viz[tgt_viz == 255] = 7
    tgt_rgb = CLASS_COLORS[tgt_viz]
    
    # Prepare prediction mask
    pred = prediction.cpu().numpy()
    pred_rgb = CLASS_COLORS[pred]

    # Target Overlay
    tgt_overlay = img.copy()
    mask_indices = (tgt_viz > 0) & (tgt_viz < 7)
    tgt_overlay[mask_indices] = (1 - alpha) * img[mask_indices] + alpha * (tgt_rgb[mask_indices] / 255.0)

    # Prediction Overlay
    pred_overlay = img.copy()
    mask_indices_pred = (pred > 0)
    pred_overlay[mask_indices_pred] = (1 - alpha) * img[mask_indices_pred] + alpha * (pred_rgb[mask_indices_pred] / 255.0)
    
    return img, tgt_overlay, pred_overlay

def save_mask_as_image(mask, filename):
    """
    mask: (H, W) integer tensor or numpy array
    filename: str, path to save
    """
    if torch.is_tensor(mask):
        mask = mask.cpu().numpy()
    
    # Convert indices to RGB
    mask_rgb = CLASS_COLORS[mask]
    from PIL import Image
    Image.fromarray(mask_rgb).save(filename)
    print(f"Saved mask to {filename}")

def load_mask_from_image(filename, device='cuda'):
    """
    Loads an RGB mask image and converts it back to class indices.
    Assumes nearest neighbor color matching to CLASS_COLORS.
    """
    from PIL import Image
    img = Image.open(filename).convert("RGB")
    img_np = np.array(img) # H, W, 3
    
    # Simple Euclidean distance to find nearest class color
    # (This handles slight compression artifacts or anti-aliasing by picking nearest class)
    H, W, C = img_np.shape
    img_flat = img_np.reshape(-1, 3) # (N, 3)
    colors = CLASS_COLORS.astype(np.float32) # (K, 3)
    
    # Vectorized distance calculation
    # dist = sum((x - y)^2)
    # We want argmin over K for each N
    
    # N x 1 x 3 - 1 x K x 3 -> N x K x 3
    # This might be too big for memory if image is large. 
    # Let's do it simpler: using broadcasting might kill memory for 512x512.
    # 512*512 = 262k pixels. 262k * 8 classes * 3 bytes is small (~6MB). It's fine.
    
    img_flat_expanded = img_flat[:, np.newaxis, :]
    colors_expanded = colors[np.newaxis, :, :]
    
    dists = np.sum((img_flat_expanded - colors_expanded) ** 2, axis=2) # (N, K)
    indices_flat = np.argmin(dists, axis=1) # (N,)
    
    indices = indices_flat.reshape(H, W).astype(np.int64)
    return torch.from_numpy(indices).to(device)

def load_model(config_name, checkpoint_path):
    print(f"Loading model config: {config_name}")
    cfg = OmegaConf.load(f"cfgs/{config_name}")
    
    print("Instantiating model...")
    model = instantiate(cfg)
    
    if os.path.exists(checkpoint_path):
        print(f"Loading weights from {checkpoint_path}...")
        state_dict = torch.load(checkpoint_path, map_location='cpu')
        model.load_state_dict(state_dict)
    else:
        print(f"Error: {checkpoint_path} not found.")
        return None
        
    model.cuda()
    model.eval()
    return model

def main():
    # Initialize Hydra
    try:
        GlobalHydra.instance().clear()
        initialize(version_base=None, config_path="cfgs")
        cfg = compose(config_name="experiments")
    except Exception as e:
        print(f"Hydra initialization error: {e}")
        return

    # Instantiate dataloaders
    print("Instantiating dataloaders...")
    # Override batch size to 8
    cfg.dataloaders.batch_size = 8
    dataloaders = instantiate(cfg.dataloaders)
    test_loader = dataloaders.test_loader

    # Load Models
    models = {}
    model_configs = [
        ("model.yaml", "model.pth", "Model_base (model.pth)"),
        ("model_b+.yaml", "model_b+.pth", "Model_b+ (model_b+.pth)"),
        ("model_s.yaml", "model_s.pth", "Model_s (model_s.pth)")
    ]

    for conf_name, ckpt_path, display_name in model_configs:
        model = load_model(conf_name, ckpt_path)
        if model:
            models[display_name] = model

    if not models:
        print("No models loaded.")
        return

    # Run inference
    print("Running inference...")
    import time
    
    # Directory for saving masks
    output_dir = "vis_outputs"
    os.makedirs(output_dir, exist_ok=True)
    
    with torch.no_grad():
        batch = next(iter(test_loader))
        
        if 'image' in batch:
            data = batch['image']
        elif 'video' in batch:
            data = batch['video']
        else:
            print("Unknown data format")
            return
        
        data = data.cuda()
        target = batch['mask']
        video_ids = batch.get('video_id', [])
        frame_indices = batch.get('frame_idx', [])

        # Run inference for all models on the batch
        batch_predictions = {}
        
        # Calculate FPS
        total_frames = data.shape[0] # Batch size
        # If data is video (B, T, C, H, W), total frames is B*T
        # But model typically processes clip as one unit or frame-by-frame?
        # The models seem to take (B, T, C, H, W) input.
        # Let's count "samples" per second for now as batch size is usually small (8).
        
        for name, model in models.items():
            torch.cuda.synchronize()
            start_time = time.time()
            
            output = model(data)
            pred = torch.argmax(output, dim=1) # (B, H, W)
            
            torch.cuda.synchronize()
            end_time = time.time()
            
            elapsed = end_time - start_time
            fps = total_frames / elapsed
            print(f"Model: {name} - Inference time: {elapsed:.4f}s, FPS: {fps:.2f}")
            
            batch_predictions[name] = pred

        # Select samples: 2nd, 3rd, 4th, 5th (indices 1, 2, 3, 4)
        # Ensure we have enough data
        # Change last sample (index 4) to index 5
        indices_to_show = [1, 2, 3, 5]
        valid_indices = [i for i in indices_to_show if i < len(data)]
        
        if not valid_indices:
            print(f"Not enough samples in batch (size {len(data)}) to show indices {indices_to_show}")
            return

        num_samples = len(valid_indices)
        
        # Prepare figure
        # To rotate:
        # Columns = Number of Samples
        # Rows = Ground Truth + Number of Models
        
        num_cols = num_samples
        num_rows = 1 + len(models) # GT, Model1, Model2, Model3
        
        fig, axs = plt.subplots(num_rows, num_cols, figsize=(5 * num_cols, 5 * num_rows))
        
        # Adjust spacing to be narrow
        plt.subplots_adjust(wspace=0.01, hspace=0.01)
        
        # Handle 1D axes if single row/col (though here we likely have multiple)
        if num_rows == 1:
            axs = np.expand_dims(axs, 0)
        if num_cols == 1:
            axs = np.expand_dims(axs, 1)

        for col_idx, sample_idx in enumerate(valid_indices):
            vid = video_ids[sample_idx].item()
            frame = frame_indices[sample_idx].item()
            
            sample_data = data[sample_idx]
            if sample_data.dim() == 4: # Video clip
                img_to_viz = sample_data[-1]
            else:
                img_to_viz = sample_data

            # Get predictions for this sample
            sample_preds = {}
            for name in models.keys():
                safe_name = name.split(' ')[0]
                base_filename = f"{safe_name}_{vid}_{frame}"
                raw_path = os.path.join(output_dir, f"{base_filename}.png")
                edited_path = os.path.join(output_dir, f"{base_filename}_edited.png")
                
                # Always save the raw prediction first
                raw_pred = batch_predictions[name][sample_idx]
                save_mask_as_image(raw_pred, raw_path)
                
                if os.path.exists(edited_path):
                    print(f"Using edited mask for {name}: {edited_path}")
                    sample_preds[name] = load_mask_from_image(edited_path)
                else:
                    sample_preds[name] = raw_pred
            
            # Use first model to get overlays
            first_model_name = list(models.keys())[0]
            img, tgt_overlay, _ = overlay_mask_prediction(img_to_viz, target[sample_idx], sample_preds[first_model_name])
            
            # 1. Plot Ground Truth (Row 0)
            ax_gt = axs[0, col_idx]
            ax_gt.imshow(tgt_overlay)
            ax_gt.axis('off')
            
            # 2. Plot Predictions (Rows 1..N)
            for i, (name, pred_mask) in enumerate(sample_preds.items()):
                _, _, pred_overlay = overlay_mask_prediction(img_to_viz, target[sample_idx], pred_mask)
                ax_pred = axs[i+1, col_idx]
                ax_pred.imshow(pred_overlay)
                ax_pred.axis('off')
            
        # Save
        save_path = "model_comparison.png"
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
        print(f"Saved comparison to {save_path}")
        
if __name__ == "__main__":
    main()
