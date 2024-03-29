import numpy as np
import cv2, PIL
import matplotlib.pyplot as plt

def plot_images(img_list, box_list, fig_size, box_color, thickness, rows, cols):
    # This function takes in img list with box list and print image dataset.
    plt.figure(figsize=fig_size)
    for x in range(len(img_list)):
        img = img_list[x]
        box = box_list[x]
        box = [float(x.strip()) for x in box.split(',')]
        img = cv2.imread(img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.rectangle(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 0, 255), thickness=thickness)
        ax = plt.subplot(4, 10, x+1)
        plt.imshow(img)
        plt.axis("off")
