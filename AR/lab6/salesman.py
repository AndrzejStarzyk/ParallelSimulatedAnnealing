import random
from time import time

from mpi4py.futures import MPIPoolExecutor
from mpi4py import MPI
import numpy as np

N = 10
cities = []

def random_coordinate_on_circle(R):
    phi = random.random()*2*np.pi
    return R*np.sin(phi), R*np.cos(phi), phi

def dist(p1, p2) -> float:
    return np.sqrt((p1[0]-p2[0])**2+(p1[1]-p2[1])**2)

def routes(depth, start):
    res = []
    curr = []
    visited = [False for _ in range(N)]

    def dfs(d, i):
        nonlocal res, curr, visited
        visited[i] = True
        curr.append(i)
        if d <= 1:
            res.append(curr.copy())
            visited[i] = False
            curr.pop()
            return
        for j in range(N):
            if not visited[j]:
                dfs(d-1, j)
        visited[i] = False
        curr.pop()
    dfs(depth, start)
    return res


def tsp(beginning, cities):
    best_route = []
    best_length = np.inf

    curr_route = beginning.copy()
    curr_length = 0.0
    for i in range(len(beginning)-1):
        curr_length += cities[beginning[i]][beginning[i+1]]

    visited = [False for _ in range(N)]
    for i in beginning:
        visited[i] = True

    def dfs_tsp(v):
        nonlocal curr_length, best_length, curr_route, best_route, visited
        if len(curr_route) == N:
            if curr_length < best_length:
                best_route = curr_route.copy()
                best_length = curr_length
            return
        for u in range(N):
            if not visited[u] and curr_length + cities[v][u] < best_length:
                curr_route.append(u)
                curr_length += cities[v][u]
                visited[u] = True

                dfs_tsp(u)

                curr_route.pop()
                curr_length -= cities[v][u]
                visited[u] = False

    if len(beginning) == 0:
        dfs_tsp(0)
    else:
        dfs_tsp(beginning[-1])

    return best_route, best_length

def bnb(depth, cities, n_workers=None):
    start = time()
    with MPIPoolExecutor(max_workers=n_workers) as executor:
        beginnings = routes(depth, 0)
        paths = executor.map(tsp, beginnings, [cities.copy() for _ in range(len(beginnings))])

        best_route = []
        best_length = np.inf

        for r, l in paths:
            if l < best_length:
                best_route = r.copy()
                best_length = l
    end = time()
    return best_route, best_length, end - start


if __name__ == '__main__':
    R = 1
    coordinates = [random_coordinate_on_circle(R) for i in range(N)]
    cities = [[dist(coordinates[i], coordinates[j]) for j in range(N)] for i in range(N)]

    sorted_cities = [(i, coordinates[i][2]) for i in range(N)]
    sorted_cities.sort(key=lambda tup: tup[1])

    t_proc = [bnb(5, cities, i)[2] if i != 0 else 0 for i in range(17)]

    acc_proc = [t_proc[1] / i if i != 0 else 0 for i in t_proc]
    eff_proc = [t_proc[i] / i if i != 0 else 0 for i in range(len(t_proc))]
    kf_proc = [(1 / t_proc[i] - 1 / i) / (1 - 1 / i)  if i != 0 and i != 1 else 0 for i in range(len(t_proc))]

    print(acc_proc)
    print(eff_proc)
    print(kf_proc)

    t_depth = [bnb(i, cities, 16)[2] if i != 0 else 0 for i in range(N)]

    acc_depth = [t_depth[1] / i if i != 0 else 0 for i in t_depth]
    eff_depth = [t_depth[i] / i if i != 0 else 0 for i in range(len(t_depth))]
    kf_depth = [(1 / t_depth[i] - 1 / i) / (1 - 1 / i) if i != 0 and i != 1 else 0 for i in range(len(t_depth))]

    print(acc_depth)
    print(eff_depth)
    print(kf_depth)




