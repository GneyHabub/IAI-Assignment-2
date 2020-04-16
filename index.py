from PIL import Image, ImageDraw
from numpy import zeros, uint8, cos, sin, array, argsort, where, delete
from numpy.random import randint, uniform
from numba import njit, prange
from random import choice
from sys import maxsize
import math
from colorthief import ColorThief as CT
from time import time

#config
path = 'img/input.jpg'
populationSize = 35
parentSize = 20
childrenSize = 15
populationFitness = []
color_thief = CT(path)
palette = array(color_thief.get_palette(color_count=10))

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
            population[i] = drawRandomLine(population[i], col)
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
def drawRandomLine(canvas, col):
    # print("Attempt to draw a random line")
    l = randint(10, 35)
    angle = math.radians(uniform(0, 360))
    x0 = randint(512)
    y0 = randint(512)
    x1 = int(l * cos(angle) + x0)
    y1 = int(l * sin(angle) + y0)
    while (x1 == x0) or (y1==y0):
        angle = math.radians(uniform(0, 360))
        x1 = int(l * cos(angle) + x0)
        y1 = int(l * sin(angle) + y0)   
    # if(x1<x0):
    #     x0, x1 = x1, x0
    # if(y1<y0):
    #     y0, y1 = y1, y0
    if (x1 > 512):
        x1 = 511
    if(y1 > 512):
        y1 = 511
    line = interpolate_pixels_along_line(x0, y0, x1, y1)
    for i in line:
        canvas[i] = col
    # print("Succeeded at drawing a line")
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
    population[i] = array(Image.open('img/output.jpg'))#zeros(shape=(512, 512, 3), dtype = uint8)
    # for j in range(512):
    #     for k in range(512):
    #         population[i][j, k] = avgColor
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
