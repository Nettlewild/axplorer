import numpy as np
import argparse
from pathlib import Path

from src.envs.environment import BaseEnvironment, DataPoint
from src.envs.tokenizers import DenseTokenizer, SparseTokenizerSequenceKTokens, SparseTokenizerSingleInteger
from src.envs.utils import random_symmetry_adj_matrix, sort_graph_based_on_degree
from src.utils import bool_flag


def read_optimum(N, sep=","):
    path = Path(__file__).parent / "cycle_optimums.txt"
    with open(path, "r") as f:
        values = [int(v.strip()) for v in f.read().split(sep) if v.strip()]
    # the file starts from N=1: values[i] is ex(i + 1, C4)
    return values[N - 1] if 1 <= N <= len(values) else None


class SquareDataPoint(DataPoint):
    MAKE_OBJECT_CANONICAL = False

    def __init__(self, N, init=False):
        super().__init__()
        self.N = N
        self.data = np.zeros((self.N, self.N), dtype=np.uint8)
        self.cycles = []
        self.optimum = self.read_optimum_from_file()

        if init:
            # here there cannot be any cycles, so _cycles_computation() is useless
            self._add_edges_greedily()
            if self.MAKE_OBJECT_CANONICAL:
                self.data = sort_graph_based_on_degree(self.data)
            self.calc_features()
            self.calc_score() # <-- here self.score is updated
                              # (number of edges)  

    def read_optimum_from_file(self, sep=","):
        optimum = read_optimum(self.N, sep)
        self.exists_optimum = optimum is not None
        return optimum
    
    def calc_score(self):
        if len(self.cycles) > 0:
            self.score = -1
        else:
            self.score = self.data.sum().item() // 2

    def calc_features(self):
        w = []
        for i in range(self.N):
            for j in range(i + 1, self.N):
                w.append(self.data[i, j])
        self.features = ",".join(map(str, w))

    def _add_edges_greedily(self):
        np.random.seed(None)
        adjmat_cycle = self.data @ self.data @ self.data
        allowed_edges = []
        for i in range(self.N):
            for j in range(i + 1, self.N):
                if self.data[i, j] == 0 and adjmat_cycle[i, j] == 0:
                    allowed_edges.append((i, j))

        while allowed_edges:
            i, j = allowed_edges[np.random.randint(len(allowed_edges))]
            self.data[i, j] = 1
            self.data[j, i] = 1
            new_allowed_edges = []
            adjmat_cycle = self.data @ self.data @ self.data
            for a, b in allowed_edges:
                if self.data[a, b] == 0 and adjmat_cycle[a, b] == 0:
                    new_allowed_edges.append((a, b))
            allowed_edges = new_allowed_edges

    def _remove_edges_greedily(self):
        while self.cycles:
            edge_count = {}
            for cycle in self.cycles:
                for edge in cycle:
                    edge_count[edge] = edge_count.get(edge, 0) + 1
            i, j = max(edge_count, key=edge_count.get)
            self.data[i, j] = 0
            self.data[j, i] = 0

            remaining_cycles = []
            for cycle in self.cycles:
                if (i, j) not in cycle:
                    remaining_cycles.append(cycle)
            self.cycles = remaining_cycles

    def _cycles_computation(self):
        cycles = set()
        row_bits = []
        for i in range(self.N):
            mask = 0
            for j in range(self.N):
                if self.data[i, j] == 1:
                    mask |= 1 << j
            row_bits.append(mask)

        for i in range(self.N):
            bits_i = row_bits[i]
            for j in range(i + 1, self.N):
                common = bits_i & row_bits[j]
                x = common
                while x:
                    lsb_u = x & -x
                    u = lsb_u.bit_length() - 1
                    x ^= lsb_u
                    y = x
                    while y:
                        lsb_v = y & -y
                        v = lsb_v.bit_length() - 1
                        y ^= lsb_v

                        # add unique sorting to speed up the future computations
                        elems = [i, u, j, v]
                        a = min(elems)
                        min_idx = elems.index(a)
                        neighbours = [elems[(min_idx + 1) % 4], elems[(min_idx - 1) % 4]]
                        b = min(neighbours)
                        d = max(neighbours)
                        c = sum(elems) - a - b - d
                        cycles.add((a, b, c, d))

        self.cycles = []
        for cycle in cycles:
            a, b, c, d = cycle
            self.cycles.append(((min(a, b), max(a, b)), (min(b, c), max(b, c)), (min(c, d), max(c, d)), (min(d, a), max(d, a))))

    def local_search(self, improve_with_local_search):
        # here I start from a dirty graph, so we need to compute 4-cycles first
        self._cycles_computation()
        # first step of local search: remove edges greedily until there is no 4-cycle
        self._remove_edges_greedily()
        # second step of local search: add edges greedily while avoiding 4-cycles
        if improve_with_local_search:
            self._add_edges_greedily()
        # no need to call _cycles_computation() given removing and adding edges doesn't create any 4-cycles
        self.cycles = []
        if self.MAKE_OBJECT_CANONICAL:
            self.data = sort_graph_based_on_degree(self.data)
        self.calc_features()
        self.calc_score()

    @classmethod
    def _update_class_params(cls, pars):
        cls.MAKE_OBJECT_CANONICAL = pars

    @classmethod
    def _save_class_params(cls):
        return cls.MAKE_OBJECT_CANONICAL


class SquareEnvironment(BaseEnvironment):
    # this problem lives in N^2, therefore k=2
    # (i, j) or (j, i) represents the same edge, therefore are_coordinates_symmetric=True
    k = 2
    are_coordinates_symmetric = True
    data_class = SquareDataPoint

    def __init__(self, params):
        super().__init__(params)
        self.optimum = read_optimum(params.N)
        self.exists_optimum = self.optimum is not None
        self.data_class.MAKE_OBJECT_CANONICAL = params.make_object_canonical
        encoding_augmentation = random_symmetry_adj_matrix if params.augment_data_representation else None
        if params.encoding_tokens == "single_integer":
            self.tokenizer = SparseTokenizerSingleInteger(
                self.data_class, params.N, self.k, self.are_coordinates_symmetric, self.SPECIAL_SYMBOLS, encoding_augmentation=encoding_augmentation
            )
        elif params.encoding_tokens == "sequence_k_tokens":
            self.tokenizer = SparseTokenizerSequenceKTokens(
                self.data_class, params.N, self.k, self.are_coordinates_symmetric, self.SPECIAL_SYMBOLS, encoding_augmentation=encoding_augmentation
            )
        elif params.encoding_tokens == "adjacency":
            self.tokenizer = DenseTokenizer(
                self.data_class,
                params.N,
                self.k,
                self.are_coordinates_symmetric,
                self.SPECIAL_SYMBOLS,
                pow2base=params.pow2base,
                encoding_augmentation=encoding_augmentation,
            )
        else:
            raise ValueError(f"Invalid encoding: {params.encoding_tokens}")

    @staticmethod
    def register_args(parser):
        """
        Register environment parameters.
        """
        parser.add_argument("--N", type=int, default=30, help="Number of vertices in the 4-cycle-free graph")
        parser.add_argument("--encoding_tokens", type=str, default="single_integer", help="single_integer/sequence_k_tokens/adjacency")
        parser.add_argument("--make_object_canonical", type=bool_flag, default="false", help="sort the graph node names based on its indegree")
        parser.add_argument(
            "--augment_data_representation", type=bool_flag, default="false", help="augment the data representation with predefined function"
        )
        parser.add_argument("--pow2base", type=int, default=1, help="Number of adjacency entries to code together")
