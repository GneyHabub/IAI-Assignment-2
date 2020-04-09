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