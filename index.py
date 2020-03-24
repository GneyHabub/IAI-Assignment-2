from PIL import Image, ImageDraw
import numpy as np

def estimateImgs(img1, img2):
    orig_px = img1.load()
    px = img2.load()
    ranks = 0
    for i in range(512):
        for j in range (512):
            if (abs(px[i, j][0] - orig_px[i, j][0]) < 10 and abs(px[i, j][1] - orig_px[i, j][1]) < 10 and abs(px[i, j][2] - orig_px[i, j][2]) < 10):
                ranks += 1
    return ranks

def produceChildren(parents):
    pass


orig = Image.open('img/input2.jpg')
orig_px = orig.load()
images = []
ranks = []
for k in range(120):
    images.append(Image.fromarray((np.random.rand(512,512,3) * 255).astype('uint8')).convert('RGB'))
    ranks.append(estimateImgs(orig, images[k]))

topParents = []
for i in range(5):
    top = ranks.index(max(ranks))
    topParents.append(images[top])
    ranks.pop(top)
