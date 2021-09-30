from skimage.io import imread, imshow
from skimage.util import img_as_ubyte, img_as_float
import matplotlib.pyplot as plt
import numpy as np


def whitePatchAlgorithm(image, percentile = 100):

    whitePatchImage = img_as_ubyte((image*1.0 /
                                      np.percentile(image, percentile, axis=(0, 1))).clip(0, 1))

    return whitePatchImage


def gray_world(image):

   #  print(image.mean(axis=(0, 1)))
    # print(image.mean())
    image_grayworld = ((image * (image.mean() /
                                 image.mean(axis=(0, 1)))).
                       clip(0, 255).astype(int))
    # for images having a transparency channel

    if image.shape[2] == 4:
        image_grayworld[:, :, 3] = 255
    return image_grayworld


def redChannelCompnesation(image):

    floatImage = img_as_float(image)
    print(floatImage.shape[0])
    channelMeans  = floatImage.mean(axis=(0, 1))
    print(channelMeans)
    redChannelCompensatedImage = floatImage
    # print(1 - redChannelCompensatedImage[0])
    print(redChannelCompensatedImage[0].shape)


    for i in range(0, redChannelCompensatedImage.shape[0]):

        for j in range(0, redChannelCompensatedImage.shape[1]):

            redChannelCompensatedImage[i][j][0] = redChannelCompensatedImage[i][j][0] + \
                                                  1*(channelMeans[1] - channelMeans[0])*(1 - redChannelCompensatedImage[i][j][0])*redChannelCompensatedImage[i][j][1]


    print(redChannelCompensatedImage.mean(axis=(0, 1)))
    return img_as_ubyte(redChannelCompensatedImage)


image = imread("images.jfif")
# print(image)
fig, ax = plt.subplots(1, 2)


ax[0].imshow(image)
ax[0].set_title("Orginal Image")
ax[1].imshow(whitePatchAlgorithm(image, 95))
ax[1].set_title("WhitePatch Image")

fig1, ax1 = plt.subplots(1, 2)

ax1[0].imshow(gray_world(image))
ax1[0].set_title("Gray World Image")
ax1[1].imshow(gray_world(redChannelCompnesation(image)))
ax1[1].set_title("Gray World Image(RedChannelCompensated)")
plt.show()