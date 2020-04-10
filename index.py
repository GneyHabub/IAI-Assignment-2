from PIL import Image, ImageDraw
from numpy import zeros, uint8, cos, sin, array, argsort, where
from numpy.random import randint, uniform
from numba import njit, prange
from random import choice
import math
import config

#config
populationSize = 35
parentSize = 20
childrenSize = 15
lineLengths = [10, 15, 20, 30, 40, 50]
populationFitness = []

# Evolutionary algorithm

@njit(parallel=False)
def generateImage():
    res = zeros((512, 512, 3), dtype = uint8)
    for i in range(512):
        for j in range(512):
            res[i, j, 0], res[i, j, 1], res[i, j, 2] = [236, 236, 228]
    return res

@njit(parallel=False)
def generateCrossover(parents, parSize, chSize):
    res = zeros((chSize, 512, 512, 3)) #initialize array with unsigned 8-bit integers
    for i in range(chSize):
        parent1 = int((parSize - 1) * uniform(0, 1) ** 2)
        parent2 = int((parSize - 1) * uniform(0, 1) ** 2)
        res[i, 0:256] = parents[parent1, 0:256]
        res[i, 256:] = parents[parent2, 256:]
    return res

@njit(parallel=False)
def mutate(population, popSize):
    for i in range(popSize):
        # print("pop size:", i)
        if uniform(0, 1) < 0.9:
            population[i] = drawRandomLine(population[i])
    return population

@njit(parallel=True)
def fitness(population, orig, popSize):
    fitness = zeros(len(population))
    for x in range(len(population)):
        distance = 0
        for i in prange(population[x].shape[0]):
            for j in range(population[x].shape[1]):
                for k in range(3):
                    distance += (int(population[x][i, j, k]) - int(orig[i, j, k])) ** 2
        fitness[x] = distance
    return fitness

@njit(parallel=True)
def chooseParents(population, fit, parentSize):
    parents = zeros((parentSize, 512, 512, 3))
    indices = argsort(populationFitness)
    for i in prange(parentSize):
        parents[i] = population[indices[i]]
    return parents

@njit(parallel=False)
def drawRandomLine(canvas):
    # print("Attempt to draw a random line")
    length =  40 # lineLengths[randint(len(lineLengths))]
    angle = math.radians(uniform(0, 360))
    x0 = randint(512)
    y0 = randint(512)
    x1 = int(length * cos(angle) + x0)
    y1 = int(length * sin(angle) + y0)
    if(x1<x0):
        x0, x1 = x1, x0
    if(y1<y0):
        y0, y1 = y1, y0
    if (x1 > 512):
        x1 = 511
    if(y1 > 512):
        y1 = 511
    line = interpolate_pixels_along_line(x0, y0, x1, y1)
    for i, j in line:
        canvas[i, j] = [0, 0, 0]
    # print("Succeeded at drawing a line")
    return canvas

@njit(parallel=False)
def interpolate_pixels_along_line(x0, y0, x1, y1):
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

# main body
orig = array(Image.open('img/input2.jpg'))
population = zeros((populationSize, 512, 512, 3), dtype = uint8) #initialize array with unsigned 8-bit integers
for i in range(populationSize):
    population[i] = generateImage() #get canvas

chkpoint_fit = 0
for i in range(100000):
    # print(i)
    populationFitness = fitness(population, orig, populationSize)
    # print("passed fitness")
    parents = chooseParents(population, populationFitness, parentSize)
    # print("passed parents")
    descendants = mutate(generateCrossover(parents, parentSize, childrenSize),childrenSize)
    # print("passed descend")
    population[0:parents.shape[0], :] = parents
    population[parents.shape[0]:, :] = descendants
    # print("passed population")
    if abs(chkpoint_fit - min(populationFitness)) >= 100000:
        chkpoint_fit = min(populationFitness)
        top = where(populationFitness == min(populationFitness))
        Image.fromarray(population[top][0]).save('//Users/gneyhabub/Documents/GitHub/IAI-Assignment-2/img/output.jpg', 'JPEG')
