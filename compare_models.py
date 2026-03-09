
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

# New colors for ESD dataset
# Class 1: Green [0, 255, 0]
# Class 2: Blue [0, 0, 255]
# Class 3: Red [255, 0, 0]
# Keeping 0 (background) black, others (if any) as defaults or similar to CLASS_COLORS
ESD_COLORS = np.array(
    [
        [0, 0, 0],         # 0: background
        [0, 255, 0],       # 1: Green
        [0, 0, 255],       # 2: Blue
        [255, 0, 0]       # 3: Red
    ],
    dtype=np.uint8,
)

def overlay_mask_prediction(image, target, prediction, alpha=0.4, colors=CLASS_COLORS):
    """
    image: (3, H, W) tensor, normalized
    target: (H, W) tensor, integer class indices
    prediction: (H, W) tensor, integer class indices
    alpha: opacity of the mask
    colors: numpy array of shape (N, 3) for class colors
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
    
    # Map ignore index (255) to 0 (background) or a safe index
    tgt_viz[tgt_viz == 255] = 0
    # Also ignore classes > 3 if using ESD_COLORS (len=4)
    # len(colors) is usually 8 for CLASS_COLORS, 4 for ESD_COLORS
    max_idx = len(colors) - 1
    tgt_viz[tgt_viz > max_idx] = 0
    
    tgt_rgb = colors[tgt_viz]
    
    # Prepare prediction mask
    pred = prediction.cpu().numpy()
    pred_viz = pred.copy()
    pred_viz[pred_viz > max_idx] = 0
    pred_rgb = colors[pred_viz]

    # Target Overlay
    tgt_overlay = img.copy()
    # Mask indices: > 0 and valid index
    mask_indices = (tgt_viz > 0)
    tgt_overlay[mask_indices] = (1 - alpha) * img[mask_indices] + alpha * (tgt_rgb[mask_indices] / 255.0)

    # Prediction Overlay
    pred_overlay = img.copy()
    mask_indices_pred = (pred_viz > 0)
    pred_overlay[mask_indices_pred] = (1 - alpha) * img[mask_indices_pred] + alpha * (pred_rgb[mask_indices_pred] / 255.0)
    
    return img, tgt_overlay, pred_overlay

def save_mask_as_image(mask, filename, colors=CLASS_COLORS):
    """
    mask: (H, W) integer tensor or numpy array
    filename: str, path to save
    """
    if torch.is_tensor(mask):
        mask = mask.cpu().numpy()
    
    # Handle 255 (ignore index) 
    mask_viz = mask.copy()
    mask_viz[mask_viz == 255] = 0
    
    # Handle out of bounds
    max_idx = len(colors) - 1
    mask_viz[mask_viz > max_idx] = 0
    
    # Convert indices to RGB
    mask_rgb = colors[mask_viz]

    from PIL import Image
    Image.fromarray(mask_rgb).save(filename)
    print(f"Saved mask to {filename}")

def load_mask_from_image(filename, device='cuda', colors=CLASS_COLORS):
    """
    Loads an RGB mask image and converts it back to class indices.
    Assumes nearest neighbor color matching to colors.
    """
    from PIL import Image
    img = Image.open(filename).convert("RGB")
    img_np = np.array(img) # H, W, 3
    
    # Simple Euclidean distance to find nearest class color
    # (This handles slight compression artifacts or anti-aliasing by picking nearest class)
    H, W, C = img_np.shape
    img_flat = img_np.reshape(-1, 3) # (N, 3)
    colors_float = colors.astype(np.float32) # (K, 3)
    
    img_flat_expanded = img_flat[:, np.newaxis, :]
    colors_expanded = colors_float[np.newaxis, :, :]
    
    dists = np.sum((img_flat_expanded - colors_expanded) ** 2, axis=2) # (N, K)
    indices_flat = np.argmin(dists, axis=1) # (N,)
    
    indices = indices_flat.reshape(H, W).astype(np.int64)
    return torch.from_numpy(indices).to(device)

def load_model(config_name, checkpoint_path, dataloaders_cfg=None):
    print(f"Loading model config: {config_name}")
    model_cfg = OmegaConf.load(f"cfgs/{config_name}")

    if dataloaders_cfg is not None:
        # Create a container to allow interpolation resolution
        container = OmegaConf.create()
        container.dataloaders = dataloaders_cfg
        container.model = model_cfg
        cfg = container.model
    else:
        cfg = model_cfg
    
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

    
    # Register eval resolver
    if not OmegaConf.has_resolver("eval"):
        OmegaConf.register_new_resolver("eval", eval)

    # Instantiate dataloaders
    print("Instantiating dataloaders...")
    # Override batch size to 8
    cfg.dataloaders.batch_size = 8
    dataloaders = instantiate(cfg.dataloaders)
    test_loader = dataloaders.test_loader

    # Load Models
    models = {}
    model_configs = [
        ("model.yaml", "checkpoints/model.pth", "Model_base (model.pth)"),
        ("model_b+.yaml", "checkpoints/model_b+.pth", "Model_b+ (model_b+.pth)"),
        ("model_s.yaml", "checkpoints/model_s.pth", "Model_s (model_s.pth)")
    ]

    for conf_name, ckpt_path, display_name in model_configs:
        model = load_model(conf_name, ckpt_path, cfg.dataloaders)
        if model:
            models[display_name] = model

    # Load ESD dataset and model for extra rows
    print("Instantiating ESD dataloaders...")
    esd_dataloaders_cfg = OmegaConf.load("cfgs/dataset_ESD.yaml")
    # Override batch size
    esd_dataloaders_cfg.train_batch_size = 4 # Not used
    esd_dataloaders_cfg.test_batch_size = 8
    
    esd_dataloaders = instantiate(esd_dataloaders_cfg)
    esd_test_loader = esd_dataloaders.test_loader
    
    # Load ESD model
    # Assuming ESD model uses model.yaml but with weights from checkpoints/model.pth (or similar?)
    # The request says "using ESD model". I'll assume it's the same base model configuration but possibly same weights?
    # Or maybe there is a specific ESD checkpoint?
    # The prompt says "using ESD model". 
    # Usually "ESD model" might refer to the method proposed in this experiment.
    # Let's assume it uses the 'Model_base' configuration and weights, or we can reuse the already loaded models['Model_base (model.pth)']
    # But wait, if I need to "do this visualization for esd_seg+.npy dataset using ESD model", 
    # implies I should run inference on esd_seg+ using one of the models.
    # I'll use the first model "Model_base" as the "ESD model" unless otherwise specified.
    
    esd_model = models.get("Model_base (model.pth)")
    if not esd_model:
        print("Model_base not found for ESD visualization")
        return

    if not models:
        print("No models loaded.")
        return

    # Run inference
    print("Running inference...")
    import time
    import random
    
    # Allow setting seed manually
    # You can change this seed to get different random samples
    SEED = 49
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    print(f"Random seed set to: {SEED}")
    
    # Directory for saving masks
    output_dir = "vis_outputs"
    os.makedirs(output_dir, exist_ok=True)
    
    # --- Process Original Dataset (CHO) ---
    with torch.no_grad():
        # CHO Data
        batch_cho = next(iter(test_loader))
        
        # ESD Data - Scan for best samples
        # We need to find 2 samples that maximize class coverage
        best_esd_samples = [] # List of tuples: (batch_idx, sample_idx_in_batch, set_of_classes, batch_data)
        
        # Limit scanning to avoid taking too long if dataset is huge
        max_batches_to_scan = 200 
        
        print("Scanning ESD dataset for samples with all 4 classes (0, 1, 2, 3)...")
        
        for batch_idx, batch_esd in enumerate(esd_test_loader):
            if batch_idx >= max_batches_to_scan and len(best_esd_samples) >= 10:
                break
                
            # No need to move data to cuda here, just check metadata
            video_ids = batch_esd.get('video_id', [])
            masks = batch_esd['mask']
            
            for i in range(len(masks)):
                m = masks[i]
                unique = torch.unique(m)
                unique_vals = set(unique.tolist())
                
                # Check if it contains all 4 classes: 0, 1, 2, 3
                # We ignore 255 or other values for this check
                if not {0, 1, 2, 3}.issubset(unique_vals):
                    continue

                # Count pixels for classes 1, 2, 3
                count0 = torch.sum(m == 0).item()
                count1 = torch.sum(m == 1).item()
                count2 = torch.sum(m == 2).item()
                count3 = torch.sum(m == 3).item()
                
                # Rule: class 1 pixels more than class 2 and 3
                # Rule: class 1 pixels cannot exceed background (class 0)
                if not (count1 > count2 and count1 > count3 and count1 <= count0):
                    continue

                # Filter out 0, 255, and classes > 3 for storage (consistent with previous logic)
                valid_classes = set(unique[(unique != 0) & (unique != 255) & (unique <= 3)].tolist())
                
                # Get video ID
                vid = video_ids[i]
                if hasattr(vid, 'item'):
                    vid = vid.item()

                # Store candidate
                best_esd_samples.append({
                    'batch': batch_esd, 
                    'idx': i,
                    'classes': valid_classes,
                    'video_id': vid
                })
                
        print(f"Found {len(best_esd_samples)} ESD candidates with all 4 classes.")
        
        # Select 2 random samples from different videos
        # import random # Moved to top
        
        # Shuffle candidates to pick randomly
        random.shuffle(best_esd_samples)
        
        selected_samples = []
        selected_vids = set()
        
        for i, cand in enumerate(best_esd_samples):
            if len(selected_samples) >= 2:
                break
                
            vid = cand['video_id']
            if vid not in selected_vids:
                selected_samples.append(i) # Store index in best_esd_samples
                selected_vids.add(vid)
        
        # If we couldn't find 2 different videos, fill up with whatever
        if len(selected_samples) < 2:
            for i in range(len(best_esd_samples)):
                if len(selected_samples) >= 2:
                    break
                if i not in selected_samples:
                    selected_samples.append(i)
        
        print(f"Selected ESD indices from candidates: {selected_samples}")

        
        # Total rows = 2 (ESD) + 2 (CHO) = 4
        num_rows = 4
        num_cols = 7
        
        fig, axs = plt.subplots(num_rows, num_cols, figsize=(5 * num_cols, 5 * num_rows))
        plt.subplots_adjust(wspace=0.01, hspace=0.01)
        
        if num_rows == 1: axs = np.expand_dims(axs, 0)
        if num_cols == 1: axs = np.expand_dims(axs, 1)

        # Helper to process a batch and specific indices
        def process_batch_samples(batch, indices, start_row_idx, dataset_prefix, model_dict, colors_to_use=CLASS_COLORS):
            if batch is None: return

            if 'image' in batch:
                data = batch['image']
            elif 'video' in batch:
                data = batch['video']
            else:
                return
            
            data = data.cuda()
            target = batch['mask']
            video_ids = batch.get('video_id', [])
            frame_indices = batch.get('frame_idx', [])
            
            # Run inference for all models on the batch
            batch_predictions = {}
            for name, model in model_dict.items():
                output = model(data)
                pred = torch.argmax(output, dim=1)
                batch_predictions[name] = pred

            valid_indices = [i for i in indices if i < len(data)]
            # Fill if needed
            if len(valid_indices) < len(indices):
                remaining = len(indices) - len(valid_indices)
                others = [i for i in range(len(data)) if i not in valid_indices]
                valid_indices.extend(others[:remaining])
            valid_indices = valid_indices[:len(indices)]

            for i, sample_idx in enumerate(valid_indices):
                row_idx = start_row_idx + i
                
                vid = video_ids[sample_idx]
                if hasattr(vid, 'item'):
                    vid = vid.item()
                    
                frame = frame_indices[sample_idx]
                if hasattr(frame, 'item'):
                    frame = frame.item()
                
                sample_data = data[sample_idx]
                if sample_data.dim() == 4: img_to_viz = sample_data[-1]
                else: img_to_viz = sample_data

                if dataset_prefix == "ESD":
                    img_to_viz = img_to_viz.flip(0)

                # Collect predictions
                sample_preds = {}
                for k, name in enumerate(model_dict.keys()):
                    safe_name = name.split(' ')[0]
                    base_filename = f"{dataset_prefix}_{safe_name}_{vid}_{frame}"
                    raw_path = os.path.join(output_dir, f"{base_filename}.png")
                    
                    # Only check for resN_..._edited.png
                    # res1 is GT, so models start at res2
                    res_filename = f"res{k+2}_{dataset_prefix}_{vid}_{frame}_edited.png"
                    res_edited_path = os.path.join(output_dir, res_filename)

                    raw_pred = batch_predictions[name][sample_idx]
                    save_mask_as_image(raw_pred, raw_path, colors=colors_to_use)
                    
                    if os.path.exists(res_edited_path):
                        print(f"Using edited mask for {name} from res file: {res_edited_path}")
                        sample_preds[name] = load_mask_from_image(res_edited_path, colors=colors_to_use)
                    else:
                        sample_preds[name] = raw_pred

                # GT
                gt_filename = f"{dataset_prefix}_GT_{vid}_{frame}"
                gt_path = os.path.join(output_dir, f"{gt_filename}.png")

                # Only check for res1_..._edited.png
                res1_filename = f"res1_{dataset_prefix}_{vid}_{frame}_edited.png"
                res1_edited_path = os.path.join(output_dir, res1_filename)

                save_mask_as_image(target[sample_idx], gt_path, colors=colors_to_use)
                
                if os.path.exists(res1_edited_path):
                    print(f"Using edited mask for GT from res file: {res1_edited_path}")
                    tgt_for_viz = load_mask_from_image(res1_edited_path, colors=colors_to_use)
                else:
                    tgt_for_viz = target[sample_idx]

                # Overlay prep
                first_model_name = list(model_dict.keys())[0]
                _, tgt_overlay, _ = overlay_mask_prediction(img_to_viz, tgt_for_viz, sample_preds[first_model_name], colors=colors_to_use)

                # Col 0: GT
                # Res1
                res1_path = os.path.join(output_dir, f"res1_{dataset_prefix}_{vid}_{frame}.png")
                save_mask_as_image(tgt_for_viz, res1_path, colors=colors_to_use)

                ax_gt = axs[row_idx, 0]
                ax_gt.imshow(tgt_overlay)
                ax_gt.axis('off')
                
                # Cols 1..N
                current_col = 1
                for name in model_dict.keys():
                    if current_col >= num_cols: break
                    
                    pred_mask = sample_preds[name]
                    # Res{col+1}
                    res_path = os.path.join(output_dir, f"res{current_col+1}_{dataset_prefix}_{vid}_{frame}.png")
                    save_mask_as_image(pred_mask, res_path, colors=colors_to_use)

                    _, _, pred_overlay = overlay_mask_prediction(img_to_viz, tgt_for_viz, pred_mask, colors=colors_to_use)
                    ax = axs[row_idx, current_col]
                    ax.imshow(pred_overlay)
                    ax.axis('off')
                    
                    current_col += 1
                
                # Fill
                while current_col < num_cols:
                    # Check for edited mask for this column
                    res_filename = f"res{current_col+1}_{dataset_prefix}_{vid}_{frame}_edited.png"
                    res_edited_path = os.path.join(output_dir, res_filename)
                    
                    if os.path.exists(res_edited_path):
                        print(f"Using edited mask for column {current_col} from res file: {res_edited_path}")
                        viz_mask = load_mask_from_image(res_edited_path, colors=colors_to_use)
                    else:
                        viz_mask = tgt_for_viz

                    res_path = os.path.join(output_dir, f"res{current_col+1}_{dataset_prefix}_{vid}_{frame}.png")
                    save_mask_as_image(viz_mask, res_path, colors=colors_to_use)
                    
                    _, _, fill_overlay = overlay_mask_prediction(img_to_viz, tgt_for_viz, viz_mask, colors=colors_to_use)
                    ax = axs[row_idx, current_col]
                    ax.imshow(fill_overlay)
                    ax.axis('off')
                    current_col += 1
        
        # Process ESD (Rows 0-1) - Scan collected candidates
        for i, cand_idx in enumerate(selected_samples):
            cand = best_esd_samples[cand_idx]
            batch = cand['batch']
            idx = cand['idx']
            # We process just this one sample with ESD_COLORS
            process_batch_samples(batch, [idx], i, "ESD", models, colors_to_use=ESD_COLORS)
        
        # Process CHO (Rows 2-3) - Scan collected candidates (just 2 samples) with CLASS_COLORS
        indices_cho = [1, 2]
        process_batch_samples(batch_cho, indices_cho, 2, "CHO", models, colors_to_use=CLASS_COLORS)
        
        # Save
        save_path = "model_comparison.png"
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
        print(f"Saved comparison to {save_path}")
        
if __name__ == "__main__":
    main()
