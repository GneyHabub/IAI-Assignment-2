from PIL import Image, ImageDraw
import numpy as np
import random
from statistics import mean
import imagehash
import randomcolor
from numba import njit
rand_color = randomcolor.RandomColor()
import time
start_time = time.time()

@njit(parallel=False)
def mixPixels(img1, img2, seed):
    px1 = img1.load()
    px2 = img2.load()
    res = Image.new('RGB', (512, 512), 'black')
    child_px = res.load()
    if seed == 0:
        for i in range(512):
            for j in range (512):
                if j%2 == 0 and i%2 == 1:
                    child_px[i, j] = px2[i, j]
                else:
                    child_px[i, j] = px1[i, j]
    else:
        for i in range(512):
            for j in range (512):
                if j%2 == 0 and i%2 == 1:
                    child_px[i, j] = px1[i, j]
                else:
                    child_px[i, j] = px2[i, j]
    return res
            
@njit(parallel=False)
def produceChildren(parents):
    nextGen = []
    for k in range(len(parents)):
        for i in range(k, len(parents)):
            seed = np.random.randint(2, size=1)
            nextGen.append(mixPixels(parents[k], parents[i], seed[0]))

    return nextGen

orig = Image.open('img/input2.jpg')
orig_px = orig.load()

@njit(parallel=False)
def estimateImgs(img1, img2):
    """Given two images, find number of pixels
    which differ less then by 10 in each component of RGB. 
    Return number of such puixels"""
    hsh = imagehash.average_hash(img1)
    otherhash = imagehash.average_hash(img2)

    return hsh - otherhash
        
orig = Image.open('img/input2.jpg')
orig_px = orig.load()
images = []
ranks = []
for k in range(120):
    images.append(Image.fromarray((np.random.rand(512,512,3) * 255).astype('uint8')).convert('RGB'))
    ranks.append(estimateImgs(orig, images[k]))


topParents = []
for i in range(5):
    top = ranks.index(min(ranks))
    topParents.append(images[top])
    ranks.pop(top)

nextgen = produceChildren(topParents)
childrank = []
for k in range(120):
    childrank.append(estimateImgs(orig, images[k]))
print("PARENTS", mean(ranks))
print("CHILDREN", mean(childrank))



print("--- %s seconds ---" % (time.time() - start_time))

  # # print("Attempt to draw a random line")
    # l = randint(10, 35)
    # angle = math.radians(uniform(0, 360))
    # x0 = randint(512)
    # y0 = randint(512)
    # x1 = int(l * cos(angle) + x0)
    # y1 = int(l * sin(angle) + y0)
    # while (x1 == x0) or (y1==y0):
    #     angle = math.radians(uniform(0, 360))
    #     x1 = int(l * cos(angle) + x0)
    #     y1 = int(l * sin(angle) + y0)   
    # # if(x1<x0):
    # #     x0, x1 = x1, x0
    # # if(y1<y0):
    # #     y0, y1 = y1, y0
    # if (x1 > 512):
    #     x1 = 511
    # if(y1 > 512):
    #     y1 = 511
    # line = interpolate_pixels_along_line(x0, y0, x1, y1)
    # for i in line:
    #     canvas[i] = col
    # # print("Succeeded at drawing a line")
    return canvas

@njit(parallel=False)
def interpolate_pixels_along_line(x0, y0, x1, y1):
    # this function is taken form stackoverflow
    """Uses Xiaolin Wu's line algorithm to interpolate all of the pixels along a
    straight line, given two points (x0, y0) and (x1, y1)
    Wikipedia article containing pseudo code that function was based off of:
        http://en.wikipedia.org/wiki/Xiaolin_Wu's_line_algorithm
    """
    pixels = []
    steep = abs(y1 - y0) > abs(x1 - x0)

    # Ensure that the path to be interpolated is shallow and from left to right
    if steep:
        t = x0
        x0 = y0
        y0 = t

        t = x1
        x1 = y1
        y1 = t

    if x0 > x1:
        t = x0
        x0 = x1
        x1 = t

        t = y0
        y0 = y1
        y1 = t

    dx = x1 - x0
    dy = y1 - y0
    gradient = dy / dx  # slope

    # Get the first given coordinate and add it to the return list
    x_end = round(x0)
    y_end = y0 + (gradient * (x_end - x0))
    xpxl0 = x_end
    ypxl0 = round(y_end)
    if steep:
        pixels.extend([(ypxl0, xpxl0), (ypxl0 + 1, xpxl0)])
    else:
        pixels.extend([(xpxl0, ypxl0), (xpxl0, ypxl0 + 1)])

    interpolated_y = y_end + gradient

    # Get the second given coordinate to give the main loop a range
    x_end = round(x1)
    y_end = y1 + (gradient * (x_end - x1))
    xpxl1 = x_end
    ypxl1 = round(y_end)

    # Loop between the first x coordinate and the second x coordinate, interpolating the y coordinates
    for x in range(xpxl0 + 1, xpxl1):
        if steep:
            pixels.extend([(math.floor(interpolated_y), x), (math.floor(interpolated_y) + 1, x)])

        else:
            pixels.extend([(x, math.floor(interpolated_y)), (x, math.floor(interpolated_y) + 1)])

        interpolated_y += gradient

    # Add the second given coordinate to the given list
    if steep:
        pixels.extend([(ypxl1, xpxl1), (ypxl1 + 1, xpxl1)])
    else:
        pixels.extend([(xpxl1, ypxl1), (xpxl1, ypxl1 + 1)])

    return pixels