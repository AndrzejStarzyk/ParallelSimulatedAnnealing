def delta(a, b):
    if a < b:
        return 1
    return 0



class Topology:
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
                        id2 = self.coordinate_index(1,
                                                    z+delta(j, self.shifts_vertical[k//2]),
                                                    j,
                                                    w - delta(k, self.shifts_horizontal[j//2]))
                        self.graph[id1][id2] = 1
                        self.graph[id2][id1] = 1


    def coordinate_index(self, u, w, k, z):
        return z + (self.degree - 1) * (k + 12*(w+self.degree*u))