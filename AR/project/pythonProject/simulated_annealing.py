import math
from random import randint, choice, random
from time import time

from mpi4py import MPI


def delta(a, b):
    if a < b:
        return 1
    return 0

class Pegasus:
    def __init__(self, degree):
        self.degree = degree
        self.n_qubits = 24*degree*(degree-1)

        self.shifts_vertical = [2, 2, 10, 10, 6, 6]
        self.shifts_horizontal = [6, 6, 2, 2, 10, 10]
        self.coordinates = [(orientation, perpendicular_tile_offset, qubit_offset, parallel_tile_offset)
                            for orientation in range(2)
                            for perpendicular_tile_offset in range(self.degree)
                            for qubit_offset in range(12)
                            for parallel_tile_offset in range(self.degree-1)]
        self.graph = [[0 for _ in range(self.n_qubits)]for _ in range(self.n_qubits)]

        for u in range(2):
            for w in range(self.degree):
                for k in range(12):
                    for z in range(self.degree - 2):
                        id1 = self.coordinate_index(u, w, k, z)
                        id2 = self.coordinate_index(u, w, k, z+1)
                        self.graph[id1][id2] = 1
                        self.graph[id2][id1] = 1
        for u in range(2):
            for w in range(self.degree):
                for j in range(6):
                    for z in range(self.degree - 1):
                        id1 = self.coordinate_index(u, w, 2*j, z)
                        id2 = self.coordinate_index(u, w, 2*j+1, z)
                        self.graph[id1][id2] = 1
                        self.graph[id2][id1] = 1
        for w in range(self.degree):
            for k in range(12):
                for j in range(12):
                    for z in range(self.degree - 1):
                            id1 = self.coordinate_index(0, w, k, z)
                            id2 = self.coordinate_index(
                                1,
                                z + delta(j, self.shifts_vertical[k//2]),
                                j,
                                w - delta(k, self.shifts_horizontal[j//2]))
                            self.graph[id1][id2] = 1
                            self.graph[id2][id1] = 1


    def coordinate_index(self, u, w, k, z):
        return z + (self.degree - 1) * (k + 12*(w+self.degree*u))

class Chimera:
    def __init__(self, rows, cols, degree):
        self.rows = rows
        self.cols = cols
        self.degree = degree

        self.n_qubits = rows * cols * 2 * degree

        self.coordinates = [(row, col, side, index)
                            for row in range(self.rows)
                            for col in range(self.cols)
                            for side in range(2)
                            for index in range(self.degree)]
        self.graph = [[0 for _ in range(self.n_qubits)]for _ in range(self.n_qubits)]

        for i in range(self.rows):
            for j in range(self.cols):
                for k in range(self.degree):
                    for k1 in range(self.degree):
                        id1 = self.coordinate_index(i, j, 0, k)
                        id2 = self.coordinate_index(i, j, 1, k1)
                        self.graph[id1][id2] = 1
                        self.graph[id2][id1] = 1
        for i in range(self.rows):
            for j in range(self.cols):
                for k in range(self.degree):
                    id1 = self.coordinate_index(i, j, 0, k)
                    id2 = self.coordinate_index(i + 1, j, 0, k)
                    id3 = self.coordinate_index(i - 1, j, 0, k)
                    if i+1 < 3:
                        self.graph[id1][id2] = 1
                        self.graph[id2][id1] = 1
                    if i-1 > 0:
                        self.graph[id1][id3] = 1
                        self.graph[id3][id1] = 1
        for i in range(self.rows):
            for j in range(self.cols):
                for k in range(self.degree):
                    id1 = self.coordinate_index(i, j, 0, k)
                    id2 = self.coordinate_index(i, j + 1, 1, k)
                    id3 = self.coordinate_index(i, j - 1, 1, k)
                    if j+1 < 3:
                        self.graph[id1][id2] = 1
                        self.graph[id2][id1] = 1
                    if j-1 > 0:
                        self.graph[id1][id3] = 1
                        self.graph[id3][id1] = 1


    def coordinate_index(self, i, j, u, k):
        return i * self.cols * 2 * self.degree + j * 2 * self.degree + u * self.degree + k


def var2row(v):
    if v//4 == 0:
        return 1, 1, 1
    elif v//4 == 1:
        return 2, 1, 2
    elif v//4 == 2:
        return 3, 2, 1

def var2col(v):
    if v//4 == 0:
        return 1, 2, 3
    elif v//4 == 1:
        return 2, 2, 3
    elif v//4 == 2:
        return 3, 3, 3

def var2side(v):
    if v//4 == 0:
        return 1, 1
    elif v//4 == 1:
        return 0, 1
    elif v//4 == 2:
        return 0, 0

very_small_number = -1000000

class ChimeraForFactorisation(Chimera):
    def __init__(self, H:list, J:list[list], T:int, Temp:float, rows=3, cols=3, degree=4, n=12):
        super().__init__(rows, cols, degree)
        self.n = n
        self.H = [h/4 for h in H]
        self.J = J
        self.T = T
        self.Temp = Temp
        self.temp_step = Temp / T

        self.variable_mappings = [[(var2row(v)[0], var2col(v)[0], 0, v%4),
                                   (var2row(v)[0], var2col(v)[0], 1, v%4),
                                   (var2row(v)[1], var2col(v)[1], var2side(v)[0], v%4),
                                   (var2row(v)[2], var2col(v)[2], var2side(v)[1], v % 4)
                                   ] for v in range(n)]
        self.weights = [[0.0 for _ in range(self.n_qubits)] for _ in range(self.n_qubits)]

        for v in range(n):
            for row1, col1, side1, index1 in self.variable_mappings[v]:
                for row2, col2, side2, index2 in self.variable_mappings[v]:
                    id1 = self.coordinate_index(row1-1, col1-1, side1, index1)
                    id2 = self.coordinate_index(row2-1, col2-1, side2, index2)
                    self.graph[id1][id2] = very_small_number
                    self.graph[id2][id1] = very_small_number

        for v in range(n):
            for w in range(n):
                for row1, col1, side1, index1 in self.variable_mappings[v]:
                    for row2, col2, side2, index2 in self.variable_mappings[w]:
                        id1 = self.coordinate_index(row1-1, col1-1, side1, index1)
                        id2 = self.coordinate_index(row2-1, col2-1, side2, index2)
                        if self.graph[id1][id2] == 1:
                            self.weights[id1][id2] = self.J[v][w]
                        if self.graph[id2][id1] == 1:
                            self.weights[id2][id1] = self.J[v][w]

        self.states = [choice([-1, 1]) for _ in range(self.n_qubits)]
        self.init_hamiltonian = sum(self.states)

    def get_energy_change(self, row, col, side, index):
        dE = 0
        v = self.coordinate_index(row, col, side, index)
        for w in range(self.n_qubits):
            if self.graph[v][w] == 1:
                dE += self.weights[v][w] * self.states[w]
        return dE

    def get_hamiltonian(self, t):
        h_p = 0
        for v in range(self.n):
            for i in range(4):
                h_p += self.H[v] * self.states[self.coordinate_index(*self.variable_mappings[v][i])]

        for v in range(self.n_qubits):
            for w in range(v+1, self.n_qubits):
                h_p += self.weights[v][w] * self.states[v] * self.states[w]

        return (1-t/self.T) * self.init_hamiltonian + t * h_p / self.T

    def get_change_probability(self, t):
        return math.exp(-1*self.get_hamiltonian(t)/self.Temp)

    def get_state(self, v):
        return self.states[v]

    def flip_state(self, v):
        self.states[v] *= -1

    def get_cell_side(self, row, col, side):
        return [self.get_state(self.coordinate_index(row, col, side, k)) for k in range(4)]

    def insert_side(self, row, col, side, qubits):
        for k in range(4):
            self.states[self.coordinate_index(row, col, side, k)] = qubits[k]

    def get_mean_state(self):
        return sum(self.states)/len(self.states)


def simulate(chimera: ChimeraForFactorisation, row, col, steps_per_epoch, epochs, init_temp, temp_decay, neighbours):
    temp = init_temp
    for epoch in range(epochs):
        for step in range(steps_per_epoch):
            side = choice([0, 1])
            index = randint(0, 3)

            v = chimera.coordinate_index(row, col, side, index)

            dE = -2 * chimera.get_state(v) * chimera.get_energy_change(row, col, side, index)

            if dE < 0 or math.exp(-dE/temp) < random():
                chimera.flip_state(v)

            temp *= temp_decay
        for i, other in enumerate(neighbours):
            if other == -1:
                continue
            comm.send(chimera.get_cell_side(row, col, (3-i)//2), dest=other)

        received = [[], [], [], []]
        sides = [(-1, 0), (1, 0), (0, 1), (0, -1)]
        for i, other in enumerate(neighbours):
            if other == -1:
                continue
            received[i] = comm.recv(source=other)
            chimera.insert_side(row+sides[i][0], col+sides[i][1], (3-i)//2, received[i])


    return chimera




if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    pid = comm.rank
    n_workers = comm.size

    steps_per_epoch = 3
    epochs = 500
    total_time = steps_per_epoch * epochs

    max_prob = 0.7
    min_prob = 0.001

    max_temp = -1/math.log(max_prob)
    min_temp = -1/math.log(min_prob)
    temp_decay = pow(min_temp / max_temp, 1/(total_time-1))

    temperature = max_temp

    input_h = [130.5, 107.5, 130.0, 107.5, -41, -82, 3, 6, -137, -81, -107, -81]
    input_J = [[0, 2, 79, 47.5, -2, -4, -8, -16, -148, -84, 0, 0],
               [0, 0, 47.5, 71, -6, -16, 1, 2, 6, 6, -124, -84],
               [0, 0, 0, 2, -2, -4, -8, -16, -148, 0, 0, -84],
               [0, 0, 0, 0, -8, -16, 1, 2, 6, -84, -124, 6],
               [0, 0, 0, 0, 0, 34, -4, -8, -8, 1, 2, 1],
               [0, 0, 0, 0, 0, 0, -8, -16, -16, 2, 4, 2],
               [0, 0, 0, 0, 0, 0, 0, 34, 0, -4, -8, -4],
               [0, 0, 0, 0, 0, 0, 0, 0, 0, -8, -16, -8],
               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]

    topology = ChimeraForFactorisation(input_h, input_J, total_time, temperature,3, 3, 4, )
    networking = [(-1, 1, -1, -1), (0, 2, -1, 3), (1, -1, -1, 4), (-1, 4, 1, -1), (3, -1, 2, 5), (-1, -1, 4, -1)] # lewo, prawo, góra, dół

    start = time()

    new_topology = simulate(topology, pid//3, pid%3, steps_per_epoch, epochs, max_temp, temp_decay, networking[pid])

    if pid == 0:
        p = (9 + 2 * (new_topology.get_state(new_topology.coordinate_index(0, 0, 0, 0))+1)//2 +
             4 * (new_topology.get_state(new_topology.coordinate_index(0, 0, 0, 1))+1)//2)
        q = (9 + 2 * (new_topology.get_state(new_topology.coordinate_index(0, 0, 0, 2))+1)//2 +
             4 * (new_topology.get_state(new_topology.coordinate_index(0, 0, 0, 3))+1)//2)
        print("p: ",p )
        print("q: ",q )
        print("pq: ", p*q)
        end = time()
        print(end - start)



    """
    topology = Pegasus(3)

    pegasus_3 = dnx.chimera_graph(3)

    print("Qubits in a full working graph: \n    D-Wave 2000Q Chimera C16: {}\n    Advantage Pegasus P16: {}".format(
        0, len(pegasus_3.nodes)))

    fig, ax = plt.subplots(1, 2, figsize=(14, 6))

    dnx.draw_chimera(pegasus_3, ax=ax[1], node_size=5, node_color='g')
    ax[1].set_title('Pegasus P16', fontsize=18)

    plt.show()
    for edge in pegasus_3.edges:
        if topology.graph[edge[0]][edge[1]] == 0:
            print(edge)
    
    """

