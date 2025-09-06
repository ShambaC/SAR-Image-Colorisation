import os
import sys
import cv2
import numpy as np

from skimage.metrics import structural_similarity as ssim

def mse(imageA, imageB):
    # Ensure images have same size
    assert imageA.shape == imageB.shape, "Images must have the same dimensions for MSE calculation."

    # Calculate the Mean Squared Error between two images
    diff = (imageA.astype("float") - imageB.astype("float")) ** 2
    err = np.sum(diff)
    # divide by total number of elements (height * width * channels)
    err /= float(np.prod(imageA.shape))

    return err

def compare_ssim(imageA, imageB):
    # Ensure images have same size
    assert imageA.shape == imageB.shape, "Images must have the same dimensions for SSIM calculation."

    # Convert to grayscale
    imageA = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
    imageB = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)

    # Calculate the Structural Similarity Index between two images.
    score, diff = ssim(imageA, imageB, full=True)

    return score

def compare_histogram(imageA, imageB, method='correlation'):
    # Ensure images have same size
    assert imageA.shape == imageB.shape, "Images must have the same dimensions for histogram comparison."

    # Convert images to HSV color space
    hsvA = cv2.cvtColor(imageA, cv2.COLOR_BGR2HSV)
    hsvB = cv2.cvtColor(imageB, cv2.COLOR_BGR2HSV)

    # Compute the histogram for each image
    histA = cv2.calcHist([hsvA], [0, 1], None, [50, 60], [0, 180, 0, 256])
    histB = cv2.calcHist([hsvB], [0, 1], None, [50, 60], [0, 180, 0, 256])

    # Normalize the histograms
    cv2.normalize(histA, histA, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    cv2.normalize(histB, histB, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)

    # Compare the histograms using the specified method
    methods = {
        'correlation': cv2.HISTCMP_CORREL,
        'chi-square': cv2.HISTCMP_CHISQR,
        'intersection': cv2.HISTCMP_INTERSECT,
        'bhattacharyya': cv2.HISTCMP_BHATTACHARYYA
    }

    if method not in methods:
        raise ValueError(f"Unknown method '{method}'. Available methods: {list(methods.keys())}")

    score = cv2.compareHist(histA, histB, methods[method])

    return score


def _collect_image_names():
    # images are named img_01_s2.png ... img_04_s2.png
    names = [f"img_{i:02d}_s2.png" for i in range(1, 5)]
    return names


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    orig_dir = os.path.join(base_dir, "images", "original")
    output_dir = os.path.join(base_dir, "images", "output")

    models = ["FT", "LDM", "LDM_VQVAE"]
    image_names = _collect_image_names()

    if not os.path.isdir(orig_dir):
        print(f"Original images folder not found: {orig_dir}")
        sys.exit(1)

    results = {}

    for model in models:
        model_dir = os.path.join(output_dir, model)
        if not os.path.isdir(model_dir):
            print(f"Skipping missing model folder: {model_dir}")
            continue

        mse_vals = []
        ssim_vals = []
        hist_vals = []

        for name in image_names:
            orig_path = os.path.join(orig_dir, name)
            gen_path = os.path.join(model_dir, name)

            if not os.path.isfile(orig_path):
                print(f"Missing original image, skipping: {orig_path}")
                continue
            if not os.path.isfile(gen_path):
                print(f"Missing generated image for model {model}, skipping: {gen_path}")
                continue

            orig = cv2.imread(orig_path, cv2.IMREAD_COLOR)
            gen = cv2.imread(gen_path, cv2.IMREAD_COLOR)

            if orig is None or gen is None:
                print(f"Failed to load image pair: {orig_path}, {gen_path}")
                continue

            # if shapes differ, try to resize generated to original
            if orig.shape != gen.shape:
                gen = cv2.resize(gen, (orig.shape[1], orig.shape[0]), interpolation=cv2.INTER_LINEAR)

            try:
                mse_v = mse(orig, gen)
                ssim_v = compare_ssim(orig, gen)
                hist_v = compare_histogram(orig, gen, method='correlation')
            except AssertionError as e:
                print(f"Skipping pair due to error: {e}")
                continue

            mse_vals.append(mse_v)
            ssim_vals.append(ssim_v)
            hist_vals.append(hist_v)

        # compute averages if we have at least one value
        if len(mse_vals) == 0:
            print(f"No valid image pairs for model {model}, skipping stats.")
            continue

        results[model] = {
            'mse_mean': float(np.mean(mse_vals)),
            'ssim_mean': float(np.mean(ssim_vals)),
            'hist_corr_mean': float(np.mean(hist_vals)),
            'count': len(mse_vals)
        }

    # print a compact report
    print("\nPer-model average metrics:")
    print("Model\tCount\tMSE\tSSIM\tHistCorr")
    for model, stats in results.items():
        print(f"{model}\t{stats['count']}\t{stats['mse_mean']:.6f}\t{stats['ssim_mean']:.6f}\t{stats['hist_corr_mean']:.6f}")


if __name__ == '__main__':
    main()