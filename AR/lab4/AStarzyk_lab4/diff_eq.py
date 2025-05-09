#!/usr/bin/env python
from time import time

from mpi4py import MPI

def scheduler(n_proc, n, humidity, steps, x):
    size = n*n
    M = [[0.0 for _ in range(size)] for _ in range(size)]
    for i in range(size):
        if i+1 < size and (i+1)%n != 0:
            M[i][i+1] = 0.25
        if i+n < size:
            M[i][i+n] = 0.25
        if i-1 >= 0 and i%n != 0:
            M[i][i-1] = 0.25
        if i-n >= 0:
            M[i][i-n] = 0.25
    humidity_step = humidity / (n + 1)
    W = [0.0 for _ in range(size)]

    for i in range(n):
        W[i] += humidity_step * (i+1) * 0.25
        W[size - n + i] += humidity_step * (i+1) * 0.25
        W[n * i + n - 1] += humidity * 0.25

    seq_time = sequential(M, W, x, steps, size)
    par_time = [-1.0 for _ in range(n_proc)]

    for i in range(2, n_proc):
        par_time[i] = parallel(M, W, x, steps, i, size)

    for i in range(1, n_proc):
        Comm.send((0, -1, 0, [], [], 0.0), dest=i)

    acc = [seq_time / par_time[i] if i != 0 and  i != 1 else 0 for i in range(n_proc)]
    eff = [par_time[i]/i if i != 0 and  i != 1 else 0 for i in range(n_proc)]
    KF = [(1/par_time[i] - 1/i)/ (1-1/i) if i != 0 and  i != 1 else 0 for i in range(n_proc)]

    print(acc, eff, KF)
    return seq_time, par_time

def sequential(M, W, x, steps, size):
    start = time()
    for i in range(steps):
        new_x = [0 for _ in range(size)]
        for j in range(size):
            new_x[j] += W[j]
            for k in range(size):
                new_x[j] += M[j][k] * x[k]
        x = new_x.copy()
    end = time()

    return end - start

def parallel(M, W, x, steps, last_proc, size):
    for i in range(1, last_proc+1):
        Comm.send((0, -2, 0, [], [], 0.0), dest=i)
    start = time()
    for i in range(steps):
        new_x = [0.0 for _ in range(size)]
        for j in range(size):
            data = (0, i, j, M[j], x,  W[j])
            msg = Comm.recv(source=MPI.ANY_SOURCE)
            if msg[1] >= 0:
                new_x[msg[2]] = msg[5]
            Comm.send(data, dest=msg[0])
        x = new_x.copy()
    end = time()

    return end - start

def worker(comm, pid, n):
    while True:
        data: tuple = comm.recv(source=0)
        print(data)
        if data[1] == -2:
            break

    size = n * n
    comm.send((pid, -1, 0, [], [], 0.0), dest=0)
    while True:
        data: tuple = comm.recv(source=0)
        step: int = data[1]
        if step == -1:
            break
        if step == -2:
            continue


        row: int = data[2]
        m: list = data[3]
        x: list = data[4]
        b: float = data[5]
        res = b
        for i in range(size):
            res += m[i] * x[i]
        comm.send((pid, step, row, [], [], res), dest=0)




if __name__ == '__main__':
    Comm = MPI.COMM_WORLD
    rank = Comm.Get_rank()
    N_proc = Comm.Get_size()

    N = 9
    Humidity = 60
    Steps = 100

    Size = N*N
    tmp = [[0 for i in range(N)] for j in range(N)]
    x_start = sum(tmp, [])

    if rank == 0:
        scheduler(N_proc, N, Humidity, Steps, x_start)
    else:
        worker(Comm, rank, N)


