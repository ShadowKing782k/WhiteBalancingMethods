from skimage.io import imread, imshow
from skimage.util import img_as_ubyte, img_as_float, img_as_uint
import skimage.filters
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
                       clip(0, 255).astype('uint8'))
    # for images having a transparency channel

    if image.shape[2] == 4:
        image_grayworld[:, :, 3] = 255
    return image_grayworld


def redChannelCompnesation(image):

    floatImage = img_as_float(image)
    #print(floatImage.shape[0])
    channelMeans  = floatImage.mean(axis=(0, 1))
    print(channelMeans)
    redChannelCompensatedImage = floatImage
    # print(1 - redChannelCompensatedImage[0])
   #  print(redChannelCompensatedImage[0].shape)


    for i in range(0, redChannelCompensatedImage.shape[0]):

        for j in range(0, redChannelCompensatedImage.shape[1]):

            redChannelCompensatedImage[i][j][0] = redChannelCompensatedImage[i][j][0] + \
                                                  1*(channelMeans[1] - channelMeans[0])*(1 - redChannelCompensatedImage[i][j][0])*redChannelCompensatedImage[i][j][1]


    print(redChannelCompensatedImage.mean(axis=(0, 1)))
    return img_as_ubyte(redChannelCompensatedImage)


def gammaCorrection(image, gamma=1.0):
    return (image/255)**(gamma)


image = imread("images.jfif")
redChannelCompnesatedImage = redChannelCompnesation(image)
grayWorldImage = gray_world(redChannelCompnesatedImage)
gammaCorrectedImage = gammaCorrection(grayWorldImage, 2)
filteredImage = 2*img_as_float(grayWorldImage) - skimage.filters.gaussian(grayWorldImage, 1, multichannel=False)
print(filteredImage)
# print(grayWorldImage)
# print(floatgrayWorldImage)

# print(grayWorldImage.shape)
#
# floatgrayWorldImage = img_as_float(grayWorldImage)
# print(floatgrayWorldImage.mean(axis=(0, 1)))
# print(img_as_float(image).mean(axis=(0, 1)))




fig0, ax0 = plt.subplots(3, 3, sharex='all', sharey='all')

ax0[0][0].imshow(image)
ax0[0][1].imshow(whitePatchAlgorithm(image, 95))
ax0[0][2].imshow(redChannelCompnesatedImage)
ax0[1][0].imshow(grayWorldImage)
ax0[1][1].imshow(gammaCorrection(grayWorldImage, 0.5))
ax0[1][2].imshow(gammaCorrection(grayWorldImage, 1))
ax0[2][0].imshow(gammaCorrection(grayWorldImage, 1.5))
ax0[2][1].imshow(gammaCorrection(grayWorldImage, 2))
ax0[2][2].imshow(gammaCorrection(grayWorldImage, 2.5))

fig1, ax1 = plt.subplots(1, 2)

ax1[0].imshow(grayWorldImage)
ax1[1].imshow(filteredImage)

plt.show()
