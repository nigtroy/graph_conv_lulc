import rasterio
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import seaborn as sns

def read_sentinel_image(filepath):
    with rasterio.open(filepath) as src:
        img = src.read()  # shape: (bands, height, width)
        meta = src.meta
        band_names = ['B2', 'B3', 'B4', 'B8', 'NDVI', 'land_cover']
    return img, meta, band_names

def summarize_bands(img, band_names):
    summary = []
    for i in range(img.shape[0]):
        band = img[i]
        summary.append({
            'Band': band_names[i],
            'Min': float(np.nanmin(band)),
            'Max': float(np.nanmax(band)),
            'Mean': float(np.nanmean(band)),
            'Std': float(np.nanstd(band))
        })
    return pd.DataFrame(summary)

def plot_rgb_ndvi(img, band_names, output_prefix="output"):
    rgb = img[[band_names.index('B4'), band_names.index('B3'), band_names.index('B2')]]  # R, G, B
    ndvi = img[band_names.index('NDVI')]

    rgb_norm = np.clip(rgb / np.percentile(rgb, 98), 0, 1)

    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    axs[0].imshow(np.moveaxis(rgb_norm, 0, -1))
    axs[0].set_title("RGB Composite")
    axs[0].axis("off")

    im = axs[1].imshow(ndvi, cmap='RdYlGn', vmin=-1, vmax=1)
    axs[1].set_title("NDVI")
    axs[1].axis("off")
    fig.colorbar(im, ax=axs[1])
    plt.tight_layout()
    plt.savefig(f"{output_prefix}_rgb_ndvi.png", dpi=300)
    plt.close()

def plot_ndvi_histogram(ndvi_band, output_path="ndvi_histogram.png"):
    ndvi_flat = ndvi_band.flatten()
    ndvi_flat = ndvi_flat[~np.isnan(ndvi_flat)]
    sns.histplot(ndvi_flat, bins=50, kde=True, color="green")
    plt.title("NDVI Distribution")
    plt.xlabel("NDVI")
    plt.ylabel("Pixel Count")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

def main(tif_path):
    img, meta, band_names = read_sentinel_image(tif_path)

    # Band Summary
    stats_df = summarize_bands(img, band_names)
    print("Band Summary:\n", stats_df)
    stats_df.to_csv("band_summary.csv", index=False)

    # Visualizations
    plot_rgb_ndvi(img, band_names)
    plot_ndvi_histogram(img[band_names.index('NDVI')])

if __name__ == "__main__":
    import sys
    tif_path = sys.argv[1] if len(sys.argv) > 1 else "sentinel2_clipped.tif"
    main(tif_path)
