import torch
import matplotlib.pyplot as plt
import numpy as np
import umap
from genesis.writers.base import BaseWriter
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def visualize_bottleneck(model:torch.nn.Module,dataloader:torch.utils.data.DataLoader, writer:BaseWriter|None = None,epoch: int | None=None):
    """Visualize the autoencoder bottleneck representations using UMAP"""
    epoch = 0 if epoch is None else epoch
    bottlenecks = []
    all_labels = []  # Renamed to avoid confusion

    with torch.no_grad():
        for batch in dataloader:
            images, batch_labels = batch  # Clear naming
            images = images.to(device)
            _, bottleneck = model(images)
            bottlenecks.append(bottleneck.cpu().numpy())
            all_labels.append(batch_labels.cpu().numpy())

            # Optional: Use more batches for better representation
            if len(bottlenecks) >= 5:  # Use first 5 batches instead of just 1
                break

    # Concatenate all collected data
    bottlenecks = np.concatenate(bottlenecks, axis=0)  # Fixed: np.concatenate
    all_labels = np.concatenate(all_labels, axis=0)    # Concatenate labels too

    # Flatten bottleneck if it has more than 2 dimensions
    if len(bottlenecks.shape) > 2:
        bottleneck_flat = bottlenecks.reshape(bottlenecks.shape[0], -1)
    else:
        bottleneck_flat = bottlenecks

    # Create UMAP reducer
    reducer = umap.UMAP(
        n_neighbors=min(15, len(bottleneck_flat) - 1),
        min_dist=0.1,
        n_components=2,
        random_state=42
    )

    # Fit and transform the bottleneck representations
    embedding = reducer.fit_transform(bottleneck_flat)

    # Create figure with subplots
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # Plot 1: UMAP colored by ACTUAL class labels
    scatter1 = axes[0].scatter(
        embedding[:, 0],
        embedding[:, 1],
        c=all_labels,  # Use actual labels, not range
        cmap='tab10',
        s=100,
        alpha=0.8,
        edgecolors='black',
        linewidth=1
    )
    axes[0].set_title(f'UMAP of Bottleneck - Colored by Class\nEpoch {epoch}')
    axes[0].set_xlabel('UMAP 1')
    axes[0].set_ylabel('UMAP 2')

    # Add colorbar for class labels
    cbar1 = plt.colorbar(scatter1, ax=axes[0])
    cbar1.set_label('Class Label')

    # Optional: Add class labels as annotations (only for a subset to avoid clutter)
    # Get unique classes and their first occurrence
    unique_labels = np.unique(all_labels)
    for label in unique_labels:
        # Find first occurrence of this label
        idx = np.where(all_labels == label)[0][0]
        x, y = embedding[idx]
        axes[0].annotate(str(label), (x, y),
                         textcoords="offset points",
                         xytext=(0, 5),
                         ha='center',
                         fontsize=8,
                         fontweight='bold',
                         color='red')

    # Plot 2: UMAP colored by bottleneck magnitude (L2 norm)
    bottleneck_magnitude = np.linalg.norm(bottleneck_flat, axis=1)
    scatter2 = axes[1].scatter(
        embedding[:, 0],
        embedding[:, 1],
        c=bottleneck_magnitude,
        cmap='viridis',
        s=100,
        alpha=0.8,
        edgecolors='black',
        linewidth=1
    )
    axes[1].set_title(f'UMAP of Bottleneck - Colored by Magnitude\nEpoch {epoch}')
    axes[1].set_xlabel('UMAP 1')
    axes[1].set_ylabel('UMAP 2')

    # Add colorbar for magnitude
    cbar2 = plt.colorbar(scatter2, ax=axes[1])
    cbar2.set_label('Bottleneck L2 Norm')

    # Add statistics text
    stats_text = f'Bottleneck dim: {bottleneck_flat.shape[1]}\n'
    stats_text += f'Num samples: {len(bottleneck_flat)}\n'
    stats_text += f'Num classes: {len(unique_labels)}\n'
    stats_text += f'Mean magnitude: {bottleneck_magnitude.mean():.3f}\n'
    stats_text += f'Std magnitude: {bottleneck_magnitude.std():.3f}'

    fig.text(0.5, 0.02, stats_text, ha='center', fontsize=10,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.suptitle(f'Latent Space Visualization - Epoch {epoch}', fontsize=14, fontweight='bold')
    plt.tight_layout()

    # Log the figure
    if writer is not None:
        writer.log_figure('Bottleneck/UMAP_visualization', fig, epoch)
        plt.close(fig)  # Close to free memory
    else:
        return plt

