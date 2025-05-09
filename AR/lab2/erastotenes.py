import math
from math import floor

from mpi4py import MPI

n = 100

comm = MPI.COMM_WORLD
p = comm.Get_size()
pid = comm.Get_rank()
print('size=%d, rank=%d' % (p, pid))

sieve = [False for _ in range(2, n+1)]
divisors = []

for i in range(2, math.floor(math.sqrt(n)) + 1):
    if not sieve[i]:
        divisors.append(i)
        for j in range(i, math.floor(math.sqrt(n)) + 1, i):
            sieve[i] = True

if pid == 0:
    not_finished = p
    primes = [i for i in divisors]

    while not_finished > 0:
        data = comm.recv(source=MPI.ANY_SOURCE, tag=not_finished)
        for el in data:
            primes.append(el)
        not_finished -= 1
    print(sorted(primes))
else:
    b = math.floor(math.sqrt(n))
    q = floor((n - b) / (p-1))
    r = n - b - p*q

    low = b + (pid-1)*q + 1
    high = b + pid*q


    if pid == p-1:
        high = n

    for div in divisors:
        if div < 2: continue
        a = math.ceil(low / div) * div
        while a < high:
            sieve[a] = True
            a += div

    primes = []
    for i in range(low, high+1):
        if not sieve[i]:
            primes.append(i)
    comm.send(primes, 0, tag=pid)
