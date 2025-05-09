from time import time
import numpy as np

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
                    if j == self.cols-1:
                        f.write(f"{(self.dipoles[i][j] + 1) // 2};")
                    else:
                        f.write(f"{(self.dipoles[i][j]+1)//2},")
                f.write('\n')
            f.write('\n')
        """X, Y = np.meshgrid(np.linspace(0, self.rows-1, self.rows), np.linspace(0, self.cols-1, self.cols))
        Z = self.dipoles * 120 + 120

        fig, ax = plt.subplots()

        ax.imshow(Z, origin='lower')

        plt.show()"""

    def glue_fields(self, fields, size, n):
        self.dipoles = np.concatenate(fields, axis=1)

def simulate(field_: MagneticField, steps, epochs, start, end, size, n_workers):

    drawn = False
    energy = 0
    spin = 0
    dEt = 20
    ds = 2
    while dEt > 5 and ds > 0.001:
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

        dEt = np.abs(field_.get_total_energy() - energy)
        ds = np.abs(field_.get_mean_spin() - spin)
        if dEt <= 5 or ds <= 0.001:
            if pid == 0 and not drawn:
                field_.disp()
            break

        energy = field_.get_total_energy()
        spin = field_.get_mean_spin()

    return field_

if __name__ == '__main__':
    comm = MPI.COMM_WORLD
    pid = comm.rank
    n_workers = comm.size

    size = 500
    temperature = 2.5
    steps_per_epoch = 500
    epochs = 5

    if pid == 0:
        field = MagneticField(size, size, temperature)
    else:
        field = None
    field = comm.bcast(field, root=0)

    starts = [i * size // n_workers for i in range(n_workers)]
    ends = [(i + 1) * size // n_workers for i in range(n_workers)]
    ends[-1] = size

    start = time()

    new_field = simulate(field, steps_per_epoch, epochs, starts[pid], ends[pid], size, n_workers)

    if pid == 0:
        print(new_field.get_total_energy(), new_field.get_mean_spin())
        end = time()
        print(end - start)

    """
    t_proc = [montecarlo(50, 2.5, 20, 100, i) if i != 0 else 0 for i in range(17)]

    acc_proc = [t_proc[1] / i if i != 0 else 0 for i in t_proc]
    eff_proc = [t_proc[i] / i if i != 0 else 0 for i in range(len(t_proc))]
    kf_proc = [(1 / t_proc[i] - 1 / i) / (1 - 1 / i)  if i != 0 and i != 1 else 0 for i in range(len(t_proc))]

    print(acc_proc)
    print(eff_proc)
    print(kf_proc)"""