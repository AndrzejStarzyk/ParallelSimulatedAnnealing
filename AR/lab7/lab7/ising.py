from time import time
import numpy as np

from mpi4py.futures import MPIPoolExecutor
from mpi4py import MPI


class MagneticField:
    def __init__(self, rows, cols, temperature):
        self.rows = rows
        self.cols = cols
        self.temperature = temperature
        self.dipoles = None

        self.initialize()

    def initialize(self):
        self.dipoles = np.random.choice([-1, 1], (self.rows, self.cols))

    def init_copy(self, dipoles):
        self.dipoles = dipoles.copy()

    def is_inside(self, i, j):
        if i >= self.rows or j >= self.cols or i < 0 or j < 0:
            return 0
        return 1

    def get_spin(self, i, j):
        return self.dipoles[i % self.rows, j % self.cols]

    def energy_change(self, i, j):
        if not self.is_inside(i, j):
            return 0
        return -2*self.get_spin(i, j)*(
                self.get_spin(i+1, j)+
                self.get_spin(i-1, j)+
                self.get_spin(i, j+1)+
                self.get_spin(i, j-1))

    def get_mean_spin(self):
        return np.abs(np.sum(self.dipoles) / (self.rows * self.cols))

    def get_total_energy(self):
        return sum(map(lambda x: self.energy_change(x[0], x[1])/2,
                       [(i, j) for i in range(self.rows) for j in range(self.cols)]))

    def disp(self):
        with open('vis.txt', 'a') as f:
            for i in range(self.rows):
                for j in range(self.cols):
                    f.write(f"{(self.dipoles[i][j]+1)//2}")
                f.write('\n')
            f.write('\n')
        """X, Y = np.meshgrid(np.linspace(0, self.rows-1, self.rows), np.linspace(0, self.cols-1, self.cols))
        Z = self.dipoles * 120 + 120

        fig, ax = plt.subplots()

        ax.imshow(Z, origin='lower')

        plt.show()"""

    def glue_fields(self, fields, size, n):

        starts = [i*size//n for i in range(n)]
        ends = [(i+1)*size // n for i in range(n)]
        ends[-1] = size
        dipoles_per_worker = [fields[i][:, starts[i]:ends[i]] for i in range(n)]
        print(dipoles_per_worker)
        self.dipoles = np.concatenate(dipoles_per_worker, axis=1)

def simulate(field_: MagneticField, specs):
    steps = specs[0]
    epochs = specs[1]
    start = specs[2]
    end = specs[3]
    size = specs[4]
    n_workers = specs[5]

    comm = MPI.COMM_WORLD
    energy = 0
    spin = 0
    dE = 2
    ds = 2
    t = 5
    while dE > 10 and ds > 0.02:
        for i in range(epochs):
            for j in range(steps):
                row = np.random.randint(0, field_.rows)
                col = np.random.randint(start, end)

                dE = field_.energy_change(row, col)

                if dE <= 0.0:
                    field_.dipoles[row, col] *= -1
                elif np.exp(-dE / field_.temperature) > np.random.rand():
                    field_.dipoles[row, col] *= -1

            gathered_dipoles = comm.allgather(field_.dipoles[:, start:end])
            field_.glue_fields(gathered_dipoles, size, n_workers)

        dE = np.abs(field_.get_total_energy() - energy)
        ds = np.abs(field_.get_mean_spin() - spin)
        if dE <= 10 and ds <= 0.02:
            break
        energy = field_.get_total_energy()
        spin = field_.get_mean_spin()

    return field_

def montecarlo(size, temperature, steps_per_epoch, epochs, n_workers=4):
    with MPIPoolExecutor(max_workers=n_workers) as executor:
        main_field = MagneticField(size, size, temperature)
        main_field.disp()

        starts = [i*size//n_workers for i in range(n_workers)]
        ends = [(i+1)*size // n_workers for i in range(n_workers)]
        ends[-1] = size

        print(starts, ends)
        start = time()
        simulated = executor.map(simulate, [main_field for _ in range(n_workers)],
                                 [(steps_per_epoch, epochs, starts[i], ends[i], size, n_workers) for i in range(n_workers)])
        new_dipoles = []
        for i, field in enumerate(simulated):
            new_dipoles.append(field.dipoles)
        main_field.glue_fields(new_dipoles, size, n_workers)
        print(main_field.get_total_energy(), main_field.get_mean_spin())
        end = time()
        main_field.disp()
        return end - start


if __name__ == '__main__':
    montecarlo(10, 2.5, 20, 5, 4)
    """
    t_proc = [montecarlo(50, 2.5, 20, 100, i) if i != 0 else 0 for i in range(17)]

    acc_proc = [t_proc[1] / i if i != 0 else 0 for i in t_proc]
    eff_proc = [t_proc[i] / i if i != 0 else 0 for i in range(len(t_proc))]
    kf_proc = [(1 / t_proc[i] - 1 / i) / (1 - 1 / i)  if i != 0 and i != 1 else 0 for i in range(len(t_proc))]

    print(acc_proc)
    print(eff_proc)
    print(kf_proc)"""