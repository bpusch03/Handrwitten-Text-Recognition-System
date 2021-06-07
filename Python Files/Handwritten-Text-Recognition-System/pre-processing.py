


import cv2
from scipy import ndimage
import numpy as np
import math


''' THIS IS NOT MY CODE - BLATANTLY COPIED FROM https://medium.com/@o.kroeger/tensorflow-mnist-and-your-own-handwritten-digits-4d1cd32bbab4
'''



input_folder = "C:/Users/Constantin/Documents/Ben's shit/GitHub/Handrwitten-Text-Recognition-System/My Handwriting images/Before pre processing"
output_folder = "C:/Users/Constantin/Documents/Ben's shit/GitHub/Handrwitten-Text-Recognition-System/My Handwriting images/processed images"


def getBestShift(img):
    cy,cx = ndimage.measurements.center_of_mass(img)

    rows,cols = img.shape
    shiftx = np.round(cols/2.0-cx).astype(int)
    shifty = np.round(rows/2.0-cy).astype(int)

    return shiftx,shifty

def shift(img,sx,sy):
    rows,cols = img.shape
    M = np.float32([[1,0,sx],[0,1,sy]])
    shifted = cv2.warpAffine(img,M,(cols,rows))
    return shifted





for j in range(11,32):
    #read the image
    gray = cv2.imread(input_folder + '/' + str(j) + '.jpg',cv2.IMREAD_GRAYSCALE)
    gray = cv2.resize(255-gray, (28,28))
    (thresh, gray) = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    while np.sum(gray[0]) == 0:
        gray = gray[1:]

    while np.sum(gray[:, 0]) == 0:
        gray = np.delete(gray, 0, 1)

    while np.sum(gray[-1]) == 0:
        gray = gray[:-1]

    while np.sum(gray[:, -1]) == 0:
        gray = np.delete(gray, -1, 1)

    rows, cols = gray.shape

    if rows > cols:
        factor = 20.0 / rows
        rows = 20
        cols = int(round(cols * factor))
        gray = cv2.resize(gray, (cols, rows))
    else:
        factor = 20.0 / cols
        cols = 20
        rows = int(round(rows * factor))
        gray = cv2.resize(gray, (cols, rows))

    colsPadding = (int(math.ceil((28 - cols) / 2.0)), int(math.floor((28 - cols) / 2.0)))
    rowsPadding = (int(math.ceil((28 - rows) / 2.0)), int(math.floor((28 - rows) / 2.0)))
    gray = np.lib.pad(gray, (rowsPadding, colsPadding), 'constant')

    shiftx, shifty = getBestShift(gray)
    shifted = shift(gray, shiftx, shifty)
    gray = shifted


    cv2.imwrite(output_folder + '/' + str(j) + '.jpg',gray)























'''images = []
for i in range(1,11):
    images.append(Image.open(input_folder + '/' + str(i) + '.jpg'))



bw_images = []

for im in images:

    bw_images.append(ImageOps.grayscale(im))
    #bw_images.append(im.convert('1'))


for i in bw_images:
    i = ImageEnhance.Contrast(i).enhance(2)

    #plt.imshow(i)
    i.show()

#plt.show()'''