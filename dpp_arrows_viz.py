import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
import numpy as np

def get_rotation_matrix(v1, v2):
    """Returns matrix R such that R @ v1 = v2 (v1, v2 are normalized)"""
    v1 = v1 / np.linalg.norm(v1)
    v2 = v2 / np.linalg.norm(v2)
    
    cross = np.cross(v1, v2)
    dot = np.dot(v1, v2)
    
    if np.linalg.norm(cross) < 1e-9:
        # Parallel
        if dot > 0:
            return np.eye(3)
        else:
            # Anti-parallel: rotate 180 deg around any orthogonal axis
            if np.abs(v1[0]) < 0.9:
                axis = np.cross(v1, [1, 0, 0])
            else:
                axis = np.cross(v1, [0, 1, 0])
            axis = axis / np.linalg.norm(axis)
            return 2 * np.outer(axis, axis) - np.eye(3)
            
    v = cross
    s = np.linalg.norm(v)
    c = dot
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    R = np.eye(3) + vx + vx @ vx * ((1 - c) / (s ** 2))
    return R

def draw_3d_cone(ax, start_point, end_point, radius=0.5, length=1.0, color='black', alpha=1.0):
    # Vector direction
    v = end_point - start_point
    v_len = np.linalg.norm(v)
    if v_len == 0:
        return
    v_norm = v / v_len
    
    # Create cone mesh pointing up Z
    theta = np.linspace(0, 2*np.pi, 20)
    z = np.linspace(0, length, 5)
    theta_grid, z_grid = np.meshgrid(theta, z)
    
    # Radius profile: r at z=0, 0 at z=length
    r_grid = radius * (1 - z_grid / length)
    
    x_grid = r_grid * np.cos(theta_grid)
    y_grid = r_grid * np.sin(theta_grid)
    
    # Rotate to align Z with v_norm
    z_axis = np.array([0, 0, 1])
    R = get_rotation_matrix(z_axis, v_norm)
    
    # Rotate points
    points = np.stack([x_grid.flatten(), y_grid.flatten(), z_grid.flatten()])
    rotated_points = R @ points
    
    # Translate so tip ends up at `end_point`.
    # Local tip is (0,0,length) -> Rotated tip is R@(0,0,length) -> We want this at end_point.
    # Shift = end_point - rotated_tip
    
    # R@(0,0,length) is length * v_norm
    shift = end_point - (length * v_norm)
    
    x_final = rotated_points[0].reshape(x_grid.shape) + shift[0]
    y_final = rotated_points[1].reshape(y_grid.shape) + shift[1]
    z_final = rotated_points[2].reshape(z_grid.shape) + shift[2]
    
    ax.plot_surface(x_final, y_final, z_final, color=color, alpha=alpha, shade=True, antialiased=True)
    
    # Close base
    r_base = np.linspace(0, radius, 2)
    t_base = np.linspace(0, 2*np.pi, 20)
    r_b, t_b = np.meshgrid(r_base, t_base)
    x_b = r_b * np.cos(t_b)
    y_b = r_b * np.sin(t_b)
    z_b = np.zeros_like(x_b)
    
    pts_b = np.stack([x_b.flatten(), y_b.flatten(), z_b.flatten()])
    rot_b = R @ pts_b
    x_bf = rot_b[0].reshape(x_b.shape) + shift[0]
    y_bf = rot_b[1].reshape(y_b.shape) + shift[1]
    z_bf = rot_b[2].reshape(z_b.shape) + shift[2]
    
    ax.plot_surface(x_bf, y_bf, z_bf, color=color, alpha=alpha)

def visualize_dpp_arrows_all(features, f_curr, bg_color='lightyellow'):
    """
    Visualize f_curr and features with large 3D dashed arrows (cones) pointing from f_curr to half of the features.
    """
    n = len(features)
    
    # 1. PCA to 3D for visualization
    pca = PCA(n_components=3)
    # Fit on all features + f_curr to ensure common space
    if f_curr.ndim == 1:
         f_curr = f_curr.reshape(1, -1)
    
    all_data = np.vstack([features, f_curr])
    coords_3d_all = pca.fit_transform(all_data)
    
    coords_3d = coords_3d_all[:n]
    f_curr_3d = coords_3d_all[n:]
    start_point = f_curr_3d[0]
    
    fig = plt.figure(figsize=(12, 10))
    fig.patch.set_alpha(0)
    
    ax = fig.add_subplot(111, projection='3d')
    ax.patch.set_alpha(0)
    
    # Plot all features (Uniform color, no highlighting)
    ax.scatter(coords_3d[:, 0], coords_3d[:, 1], coords_3d[:, 2],
                alpha=0.8, label='Features', s=600, color='#1f77b4', depthshade=False)
    
    # Plot f_curr
    ax.scatter(start_point[0], start_point[1], start_point[2],
                color='red', label='f_curr', s=1200, marker='*', depthshade=False)
    
    # Calculate scale for arrows
    span_x = coords_3d_all[:,0].max() - coords_3d_all[:,0].min()
    span_y = coords_3d_all[:,1].max() - coords_3d_all[:,1].min()
    span_z = coords_3d_all[:,2].max() - coords_3d_all[:,2].min()
    max_span = max(span_x, span_y, span_z)
    
    head_len = max_span * 0.02
    head_rad = max_span * 0.006
    
    # Draw dashed arrows from f_curr to half of the features (every 2nd one)
    for i in range(0, n, 3):
        end_point = coords_3d[i]
        
        # Draw Shaft
        ax.plot([start_point[0], end_point[0]], 
                [start_point[1], end_point[1]], 
                [start_point[2], end_point[2]], 
                color='grey', linestyle='--', linewidth=2, alpha=0.8)
        
        # Draw 3D Cone Head
        draw_3d_cone(ax, start_point, end_point, 
                     radius=head_rad, length=head_len, color='grey', alpha=0.8)

    # Configure panes (Style matching previous)
    if bg_color:
        ax.xaxis.pane.fill = True
        ax.yaxis.pane.fill = True
        ax.zaxis.pane.fill = True
        ax.xaxis.pane.set_facecolor(bg_color)
        ax.yaxis.pane.set_facecolor(bg_color)
        ax.zaxis.pane.set_facecolor(bg_color)
        
        ax.xaxis.pane.set_edgecolor('w')
        ax.yaxis.pane.set_edgecolor('w')
        ax.zaxis.pane.set_edgecolor('w')
    else:
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
    
    # Remove tick labels
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])
    
    ax.grid(True, alpha=0.3)
    
    # Set axis range
    x_min, x_max = coords_3d_all[:, 0].min(), coords_3d_all[:, 0].max()
    y_min, y_max = coords_3d_all[:, 1].min(), coords_3d_all[:, 1].max()
    z_min, z_max = coords_3d_all[:, 2].min(), coords_3d_all[:, 2].max()
    
    pad = 0.1
    ax.set_xlim(x_min - pad, x_max + pad)
    ax.set_ylim(y_min - pad, y_max + pad)
    ax.set_zlim(z_min - pad, z_max + pad)
    
    # Save as SVG
    filename = "dpp_all_arrows_3d.svg"
    # To control transparency, change transparent=True to False or set facecolor alpha
    # User requested adjusting transparency factor. 
    # transparent=True makes background fully transparent. 
    # If opaque is desired, set transparent=False.
    plt.savefig(filename, format='svg', transparent=False, bbox_inches='tight')
    print(f"Saved {filename}")
    
    plt.show()
