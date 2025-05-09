import random
from time import time

from mpi4py.futures import MPIPoolExecutor
from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD


x0, x1, w = -2.0, +2.0, 640*2
y0, y1, h = -1.5, +1.5, 480*2
dx = (x1 - x0) / w
dy = (y1 - y0) / h

c = complex(0, 0.65)

lines_per_worker = [0 for _ in range(comm.size)]
time_per_worker = [0 for _ in range(comm.size)]

def julia(x, y):
    z = complex(x, y)
    n = 255
    while abs(z) < 3 and n > 1:
        z = z**2 + c
        n -= 1
    return n

def julia_line(k):
    rank = comm.Get_rank()
    start = time()
    line = bytearray(w)
    y = y1 - k * dy
    for j in range(w):
        x = x0 + j * dx
        line[j] = julia(x, y)


    if k == 950:
        np.fft.fft(np.exp(random.randint(1, 100)*0.01j * np.pi * np.arange(10000000) / 10000000))
    end = time()
    print(f"Rank procesu {rank}, linia nr {k}, czas obliczeń {end - start}")
    return line, end - start, rank

if __name__ == '__main__':
    total_start = time()
    with MPIPoolExecutor() as executor:

        image = executor.map(julia_line, range(h))

        with open('julia.pgm', 'wb') as f:
            f.write(b'P5 %d %d %d\n' % (w, h, 255))
            max_t = 0
            min_t = 1
            for img_line, time_p, pid in image:
                max_t = max(max_t, time_p)
                min_t = min(min_t, time_p)
                lines_per_worker[pid] += 1
                time_per_worker[pid] += time_p
                f.write(img_line)
    total_end = time()
    print(f"Max time - min time: {max_t - min_t}")
    print(lines_per_worker)
    print(time_per_worker)
    print(total_end - total_start)