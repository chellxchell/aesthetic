import cv2
import numpy as np
from skimage import io
import matplotlib.pyplot as plt
from matplotlib.colors import rgb2hex

#do in batches so my computer doesn't die
# people = ['chelly','daniella','danny','irene','teddy']
# people = ['akhil','armaan','sarah','sophia','stephen','zach']
# people = ['sakke','jacob','jason','angel','sky','emma']
people = ['adam','ava','david','et','isaac','lou','ravindra','stephanie']

for x in range (0,len(people)):
    #read the image
    img = io.imread(f'./profiles/{people[x]}.jpg')[:, :, :]

    #calculate the mean of each chromatic channel
    average = img.mean(axis=0).mean(axis=0)
    hexColor = rgb2hex(average/255) #convert to hex
    pixels = np.float32(img.reshape(-1, 3))

    #apply k-means clustering to create a palette with the most representative colours of the image
    n_colors = 5
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, .1)
    flags = cv2.KMEANS_RANDOM_CENTERS

    _, labels, palette = cv2.kmeans(pixels, n_colors, None, criteria, 10, flags)
    _, counts = np.unique(labels, return_counts=True)
    dominant = palette[np.argmax(counts)]

    #generate figure
    avg_patch = np.ones(shape=img.shape, dtype=np.uint8)*np.uint8(average)

    indices = np.argsort(counts)[::-1]   
    freqs = np.cumsum(np.hstack([[0], counts[indices]/counts.sum()]))
    rows = np.int_(img.shape[0]*freqs)
 
    dom_patch = np.zeros(shape=img.shape, dtype=np.uint8)
    hexVals = ""
    for i in range(len(rows) - 1):
        dom_patch[rows[i]:rows[i + 1], :, :] += np.uint8(palette[indices[i]])
        hexVals = hexVals + str(rgb2hex((palette[indices[i]]) / 255)) + " "
        if i==2:
            hexVals = hexVals + "\n"

    fig, (ax0, ax1,ax2) = plt.subplots(1, 3, figsize=(12,6))

    ax0.imshow(img)
    ax0.set_title('Insta Feed \n' + people[x])
    ax0.axis('off')
    ax1.imshow(avg_patch)
    ax1.set_title('Average color: \n'+str(hexColor))
    ax1.axis('off')
    ax2.imshow(dom_patch)
    ax2.set_title('Dominant colors \n' + hexVals)
    ax2.axis('off')

    plt.show(fig)