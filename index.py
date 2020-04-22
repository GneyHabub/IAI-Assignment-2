from PIL import Image, ImageDraw
from numpy import zeros, uint8, cos, sin, array, argsort, where, delete
from numpy.random import randint, uniform
from numba import njit, prange
from random import choice
from sys import maxsize
import math
import os
from colorthief import ColorThief as CT
from time import time

#config
path = 'img/girl.jpg'
populationSize = 35
parentSize = 20
childrenSize = 15
populationFitness = []
color_thief = CT(path)
palette = array(color_thief.get_palette(color_count=40))

# Evolutionary algorithm
@njit(parallel=False)
def generateCrossover(parents, parSize, chSize):
    res = zeros((chSize, 512, 512, 3)) 
    for i in range(chSize):
        res[i, 0:256] = parents[randint(parSize), 0:256]
        res[i, 256:] = parents[randint(parSize), 256:]
    return res

@njit(parallel=False)
def mutate(population, popSize):
    for i in range(popSize):
        ind = randint(len(palette))
        col = palette[ind]
        flag = randint(10)
        if(flag < 9):
            population[i] = drawSquare(population[i], col)
    return population

@njit(parallel=True)
def fitness(population, orig, popSize):
    fitness = zeros(len(population))
    for item in range(len(population)):
        fit = 0
        for i in prange(512):
            for j in range(512):
                for k in range(3):
                    fit += (int(population[item][i, j, k]) - int(orig[i, j, k])) ** 2
        fitness[item] = fit
    return fitness

@njit(parallel=True)
def chooseParents(population, fit, parentSize):
    parents = zeros((parentSize, 512, 512, 3))
    indices = argsort(fit)
    for i in prange(parentSize):
        parents[i] = population[indices[i]]
    return parents

@njit(parallel=False)
def drawSquare(canvas, col):
    l = randint(3, 15)
    x0 = randint(512)
    if(512 - x0 <= l):
        x1 = x0 - l
    else: 
        x1 = x0 + l
    y0 = randint(512)
    if(512 - y0 <= l):
        y1 = y0 - l
    else: 
        y1 = y0 + l

    startX = min(x0, x1)
    endX = max(x0, x1)
    startY = min(y0, y1)
    endY = max(y0, y1)
    for i in range(startX, endX):
        for j in range(startY, endY):
            canvas[i, j] = col
    return canvas
  

@njit(parallel=False)
def get_average_color(x,y, n, image):
    """ Returns a 3-tuple containing the RGB value of the average color of the
    given square bounded area of length = n whose origin (top left corner) 
    is (x, y) in the given image"""
 
    r, g, b = 0, 0, 0
    count = 0
    for s in range(x, x+n+1):
        for t in range(y, y+n+1):
            if(len(image[s, t]) == 3):
                pixlr, pixlg, pixlb = image[s, t]
            else:
                pixlr, pixlg, pixlb, chunk = image[s, t]
            r += pixlr
            g += pixlg
            b += pixlb
            count += 1
    return ((r/count), (g/count), (b/count))

# main body
orig = array(Image.open(path))
avgColor = get_average_color(0, 0, 512, orig)
compColor = [255 - avgColor[0], 255- avgColor[1], 255- avgColor[2]]

population = zeros((populationSize, 512, 512, 3), dtype = uint8) #initialize array with unsigned 8-bit integers
for i in range(populationSize):
    # population[i] = array(Image.open('img/output.jpg'))
    for j in range(512):
        for k in range(512):
            population[i][j, k] = avgColor
gifPath = '/Users/gneyhabub/Documents/GitHub/IAI-Assignment-2/img/gif/' + path[4:len(path) - 4]
os.mkdir(gifPath)
for i in range(100000):
    # print(i)
    populationFitness = fitness(population, orig, populationSize)
    # print("passed fitness")
    parents = chooseParents(population, populationFitness, parentSize)
    # print("passed parents")
    descendants = mutate(generateCrossover(parents, parentSize, childrenSize),childrenSize)
    # print("passed descend")
    population[0:20] = parents
    population[20:] = descendants
    # print("passed population")
    if i % 100 == 0:
        top = where(populationFitness == min(populationFitness))
        Image.fromarray(population[top][0]).save('//Users/gneyhabub/Documents/GitHub/IAI-Assignment-2/img/output.jpg', 'JPEG')
    if i % 500 == 0:
        top = where(populationFitness == min(populationFitness))
        Image.fromarray(population[top][0]).save('/Users/gneyhabub/Documents/GitHub/IAI-Assignment-2/img/gif/'+ path[4:len(path) - 4] + '/output' + str(i) + '.jpg', 'JPEG')
