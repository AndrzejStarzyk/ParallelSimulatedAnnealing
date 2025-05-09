#!/usr/bin/env python

from mpi4py import MPI


def scheduler(n_proc, n, humidity, steps, x):
    size = n * n
    M = [[0.0 for _ in range(size)] for _ in range(size)]
    for i in range(size):
        if i + 1 < size and (i + 1) % n != 0:
            M[i][i + 1] += 0.25
        if i + n < size:
            M[i][i + n] += 0.25
        if i - 1 >= 0 and i % n != 0:
            M[i][i - 1] += 0.25
        if i - n >= 0:
            M[i][i - n] += 0.25
    humidity_step = humidity / (n + 1)
    W = [0.0 for _ in range(size)]

    for i in range(n):
        W[i] += humidity_step * (i + 1) * 0.25
        W[size - n + i] += humidity_step * (i + 1) * 0.25
        W[n * i + n - 1] += humidity * 0.25

    for i in range(steps):
        new_x = [0.0 for _ in range(size)]
        for j in range(size):
            data = (0, i, j, M[j], x, W[j])
            msg = comm.recv(source=MPI.ANY_SOURCE)
            if msg[1] >= 0:
                new_x[msg[2]] = msg[5]
            comm.send(data, dest=msg[0])
        x = new_x.copy()

    for i in range(1, n_proc):
        comm.send((0, -1, 0, [], [], 0.0), dest=i)

    print(x)


def worker(comm, pid, n):
    size = n * n
    comm.send((pid, -1, 0, [], [], 0.0), dest=0)
    while True:
        data: tuple = comm.recv(source=0)
        step: int = data[1]
        if step == -1:
            break

        row: int = data[2]
        m: list = data[3]
        x: list = data[4]
        b: float = data[5]
        res = b
        for i in range(size):
            res += m[i] * x[i]
        comm.send((pid, step, row, [], [], res), dest=0)


if __name__ == '__main__':
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    n_proc = comm.Get_size()

    n = 19
    humidity = 68.4
    steps = 4000

    size = n * n
    tmp = [[0 for i in range(n)] for j in range(n)]
    x_start = sum(tmp, [])

    if rank == 0:
        scheduler(n_proc, n, humidity, steps, x_start)
    else:
        worker(comm, rank, n)