import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def plot_individual_bands(image_array, band_names):
    """Plots each individual band in grayscale."""
    num_bands = image_array.shape[-1]
    plt.figure(figsize=(15, 4))
    for i in range(num_bands):
        plt.subplot(1, num_bands, i + 1)
        plt.imshow(image_array[:, :, i], cmap='gray')
        plt.title(band_names[i])
        plt.axis('off')
    plt.suptitle('Individual Sentinel-2 Bands')
    plt.tight_layout()
    plt.show()


def plot_pixel_histograms(image_array, band_names):
    """Plots the histogram of pixel intensities for each band."""
    plt.figure(figsize=(10, 6))
    for i in range(image_array.shape[-1]):
        plt.hist(image_array[:, :, i].ravel(), bins=100, alpha=0.5, label=band_names[i])
    plt.title("Pixel Value Distribution per Band")
    plt.xlabel("Pixel Value")
    plt.ylabel("Frequency")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_class_distribution(label_array, class_mapping):
    """Plots the class distribution based on pixel counts."""
    unique, counts = np.unique(label_array, return_counts=True)
    class_labels = [class_mapping.get(cls, f"Class {cls}") for cls in unique]

    plt.figure(figsize=(10, 6))
    plt.bar(class_labels, counts, color='skyblue')
    plt.title("Land Cover Class Distribution")
    plt.xlabel("Land Cover Class")
    plt.ylabel("Pixel Count")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def plot_ndvi_histogram(ndvi_band):
    plt.figure(figsize=(8, 5))
    plt.hist(ndvi_band.ravel(), bins=100, color='green')
    plt.title("NDVI Distribution")
    plt.xlabel("NDVI Value")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def show_rgb(image_array, bands_indices=(2, 1, 0)):
    """Displays an RGB image from the band stack."""
    rgb = image_array[:, :, bands_indices]
    rgb = (rgb - np.min(rgb)) / (np.max(rgb) - np.min(rgb))  # normalize
    plt.figure(figsize=(8, 8))
    plt.imshow(rgb)
    plt.title("RGB Composite")
    plt.axis("off")
    plt.tight_layout()
    plt.show()


def plot_band_correlation(image_array, band_names):
    """Plots a correlation matrix between bands."""
    reshaped = image_array.reshape(-1, image_array.shape[-1])
    df = pd.DataFrame(reshaped, columns=band_names)
    corr = df.corr()
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr, annot=True, cmap='coolwarm')
    plt.title("Band Correlation Matrix")
    plt.tight_layout()
    plt.show()