# Includes
import os
import cv2
import csv
import math
import numpy as np
import matplotlib.pyplot as plt

# read cv2 image as RGB image instead of BGR
def cv2_imread_rgb(fname):
    return cv2.cvtColor(cv2.imread(fname), cv2.COLOR_BGR2RGB)


# write cv2 image as RGB image instead of BGR
def cv2_imwrite_rgb(fname, img):
    cv2.imwrite(fname, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


# plot images
def plot_images(images, titles=None, cols=3, fontsize=12):

    n_imgs = len(images)

    if images is None or n_imgs < 1:
        print("No images to display.")
        return

    img_h, img_w = images[0].shape[:2]
    rows = math.ceil(n_imgs / cols)
    width = 21  # 15
    row_height = math.ceil((width/cols)*(img_h/img_w))  # they are 1280*720

    plt.figure(1, figsize=(width, row_height * rows))

    for i, image in enumerate(images):
        if len(image.shape) > 2:
            cmap = None
        else:
            cmap = 'gray'
        title = ""
        if titles is not None and i < len(titles):
            title = titles[i]
        plt.subplot(rows, cols, i+1)
        plt.title(title, fontsize=fontsize)
        # show the image with type uint8
        plt.imshow(image.astype(np.uint8), cmap=cmap)

    plt.tight_layout()
    plt.show()

