import os
import glob
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from sklearn.mixture import GaussianMixture
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

def extract_features(frame_paths, device):
    # Load pretrained ResNet50
    resnet = models.resnet50(pretrained=True)
    # Remove the classification layer (fc)
    modules = list(resnet.children())[:-1]
    model = nn.Sequential(*modules)
    model = model.to(device)
    model.eval()

    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    features = []
    print(f"Extracting features from {len(frame_paths)} frames...")
    
    batch_size = 32
    with torch.no_grad():
        for i in range(0, len(frame_paths), batch_size):
            batch_paths = frame_paths[i:i+batch_size]
            batch_imgs = []
            for p in batch_paths:
                img = Image.open(p).convert('RGB')
                batch_imgs.append(preprocess(img))
            
            batch_tensor = torch.stack(batch_imgs).to(device)
            # shape: (batch, 2048, 1, 1)
            feat = model(batch_tensor)
            feat = feat.flatten(1) # (batch, 2048)
            features.append(feat.cpu().numpy())
            
    return np.concatenate(features, axis=0)

def main():
    # Configuration
    video_id = "185"
    data_dir = "/home/zheyaogao/Experiments/ESD_seg/endoscapes/test_seg"
    output_plot = "cluster_viz.png"
    n_components = 3
    
    # 1. Load frame paths
    pattern = os.path.join(data_dir, f"{video_id}_*.jpg")
    frame_paths = sorted(glob.glob(pattern))
    
    if not frame_paths:
        print(f"No frames found for video {video_id} in {data_dir}")
        # Try checking other directories if needed, but based on ls, they are there
        return

    print(f"Found {len(frame_paths)} frames.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. Extract features
    features = extract_features(frame_paths, device)
    print(f"Features shape: {features.shape}")

    # 3. GMM Clustering
    # Normalize features for better clustering/t-SNE
    scaler = StandardScaler()
    features_norm = scaler.fit_transform(features)

    print("Fitting GMM...")
    gmm = GaussianMixture(n_components=n_components, random_state=42)
    labels = gmm.fit_predict(features_norm)
    
    # Find typical frames (closest to cluster centers)
    # GMM means are in standardized space
    cluster_centers = gmm.means_
    typical_indices = []
    
    for i in range(n_components):
        # features belonging to this cluster
        cluster_mask = (labels == i)
        cluster_features = features_norm[cluster_mask]
        cluster_original_indices = np.where(cluster_mask)[0]
        
        if len(cluster_features) == 0:
            typical_indices.append(None)
            continue
            
        # Distances to the center
        dists = np.linalg.norm(cluster_features - cluster_centers[i], axis=1)
        min_dist_idx = np.argmin(dists)
        original_idx = cluster_original_indices[min_dist_idx]
        typical_indices.append(original_idx)

    # 4. t-SNE Visualization
    print("Running t-SNE...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(frame_paths)-1))
    tsne_results = tsne.fit_transform(features_norm)

    # 5. Plotting
    plt.figure(figsize=(15, 10))
    
    # Scatter plot
    colors = ['r', 'g', 'b']
    for i in range(n_components):
        mask = (labels == i)
        plt.scatter(tsne_results[mask, 0], tsne_results[mask, 1], c=colors[i], label=f'Cluster {i}', alpha=0.6)
        
        # Mark the typical frame
        if typical_indices[i] is not None:
            idx = typical_indices[i]
            plt.scatter(tsne_results[idx, 0], tsne_results[idx, 1], c='k', marker='x', s=100, linewidth=3)
            plt.annotate(f"Typical {i}", (tsne_results[idx, 0], tsne_results[idx, 1]), xytext=(5, 5), textcoords='offset points')

    plt.legend()
    plt.title(f"GMM Clustering of Video {video_id} Frames (t-SNE Projection)")
    plt.xlabel("t-SNE dim 1")
    plt.ylabel("t-SNE dim 2")

    # Add images of typical frames to the plot (as subplots or insets)
    # We will create a grid: top part is scatter, bottom part shows the 3 images
    
    # Save the scatter plot first to a buffer or just redraw
    plt.tight_layout()
    # Let's actually make a combined figure with subplots
    
    fig = plt.figure(figsize=(16, 8))
    gs = fig.add_gridspec(2, 4) # 2 rows, 4 cols. Top row for scatter (spanning all cols), bottom for images

    # Scatter plot
    ax_scatter = fig.add_subplot(gs[0, :])
    for i in range(n_components):
        mask = (labels == i)
        ax_scatter.scatter(tsne_results[mask, 0], tsne_results[mask, 1], c=colors[i], label=f'Cluster {i}', alpha=0.6)
        
        if typical_indices[i] is not None:
            idx = typical_indices[i]
            ax_scatter.scatter(tsne_results[idx, 0], tsne_results[idx, 1], c='k', marker='*', s=200, label=f'Center {i}')

    ax_scatter.legend()
    ax_scatter.set_title(f"t-SNE of Video {video_id} Frames with GMM Clusters")

    # Typical frames
    for i in range(n_components):
        if typical_indices[i] is not None:
            idx = typical_indices[i]
            img_path = frame_paths[idx]
            img = Image.open(img_path)
            
            # Place in the 2nd row, columns 0, 1, 2
            # Or maybe better spacing
            ax_img = fig.add_subplot(gs[1, i])
            ax_img.imshow(img)
            ax_img.set_title(f"Cluster {i} Typical Frame\n{os.path.basename(img_path)}")
            ax_img.axis('off')
            
            print(f"Cluster {i} typical frame: {os.path.basename(img_path)}")

            # Add border color matching cluster
            for spine in ax_img.spines.values():
                spine.set_edgecolor(colors[i])
                spine.set_linewidth(3)

    plt.tight_layout()
    plt.savefig(output_plot)
    print(f"Saved visualization to {output_plot}")

if __name__ == "__main__":
    main()

