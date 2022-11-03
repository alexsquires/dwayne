import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from icet import ClusterSpace, StructureContainer, ClusterExpansion
from icet.tools.structure_generation import occupy_structure_randomly
from trainstation import EnsembleOptimizer
import ase, icet
from dataclasses import dataclass
from ase.build import (
    make_supercell,
    get_deviation_from_optimal_cell_shape,
    find_optimal_cell_shape,
)
from icet.tools.structure_generation import enumerate_supercells
from copy import deepcopy
from random import choice
from mchammer.ensembles import CanonicalEnsemble
from mchammer.calculators import ClusterExpansionCalculator


def concentrations_fit_structure(
    structure: ase.Atoms,
    cluster_space: icet.ClusterSpace,
    concentrations: dict[str, dict[str, float]],
    tol: float = 1e-5,
) -> bool:

    for sublattice in cluster_space.get_sublattices(structure):
        if sublattice.symbol in concentrations:
            sl_conc = concentrations[sublattice.symbol]
            for conc in sl_conc.values():
                n_symbol = conc * len(sublattice.indices)
                if abs(int(round(n_symbol)) - n_symbol) > tol:
                    return False
    return True


@dataclass
class StructureFinder:
    cluster_space: icet.ClusterSpace
    max_size: int = 10
    min_size: int = 1
    concentrations: dict[str, dict[str, float]] = None

    def get_commensurate_sizes(self):
        sizes = []
        for size in range(self.min_size, self.max_size + 1):
            test = make_supercell(
                self.cluster_space.primitive_structure,
                [[size, 0, 0], [0, 1, 0], [0, 0, 1]],
            )
            if concentrations_fit_structure(
                test, self.cluster_space, self.concentrations
            ):
                sizes.append(size)
        return sizes

    def get_best_cell(self, size):
        cell = find_optimal_cell_shape(
            self.cluster_space.primitive_structure.cell, size, "sc"
        )
        structure = make_supercell(self.cluster_space.primitive_structure, cell)
        return structure
