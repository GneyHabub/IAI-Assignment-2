from numpy import zeros, uint8
from numpy.random import randint, uniform
from numba import njit, prange
import config

# Evolutionary algorithm

def generateImage():
    res = zeros((512, 512, 3), dtype = uint8)
    for i in range(512):
        for j in range(512):
            res[i, j, 0], res[i, j, 1], res[i, j, 2] = [236, 236, 228]
    return res

@njit(parallel=False)
def generateCrossover(parents, parSize, chSize):
    res = zeros(ch_size, 512, 512, 3) #initialize array with unsigned 8-bit integers
    for i in range(ch_size):
        parent1 = randint(1, par_size + 1)
        parent2 = randint(1, par_size + 1)
        res[i, 0:256] = parents[parent1, 0:256]
        res[i, 256:] = parents[parent2, 256:]
    return res

@njit(parallel=False)
def mutate(population, popSize, length):
    for i in range(popSize):
        if uniform() < 0.9:
            population[i] = draw_stroke(population[i], length)
    return population

@njit(parallel=True)
def fitness(population, orig, popSize):
    res = zeros(popSize)
    for i in prange(popSize):
        for j in range(512):
            for n in range(512):
                for m in range(3):
                    res[i] += (int(population[i][j, n, m]) - int(orig[j, n, m])) ** 2
    return res

@njit(parallel=True)
def chooseParents(population, fit, parentSize):
    parents = zeros((parentSize, 512, 512, 3))
    indices = argsort(fitness)
    for i in prange(n_parents):
        parents[i] = population[indices[i]]
    return parents

TODO 
@njit(parallel=False)
def draw_stroke(canvas, length):
    pass 

size = 35
population = zeros((size, 512, 512, 3), dtype = uint8) #initialize array with unsigned 8-bit integers
for i in range(size):
    population[i] = generateImage() #get canvas