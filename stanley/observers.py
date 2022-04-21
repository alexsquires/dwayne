from ase import Atoms
from mchammer.observers.base_observer import BaseObserver
from localenv import LocalNeiboroughCounter, NTMChannelCounter, PercolationTracker


class LocalEnvObserver(BaseObserver):
    def __init__(
        self,
        atoms: "ase.Atoms",
        background: list[str],
        cutoff: float = 3.0,
        interval: int = None,
        return_type=dict,
    ):
        """Instantiate an BaseLocalEnvObserver object"""
        tag = "LocalNeighoroughObserver"
        super().__init__(tag=tag, interval=interval, return_type=return_type)
        self.background = background
        self.cutoff = cutoff
        self.atoms = atoms
        self.interval = None

    def get_observable(self, structure: Atoms) -> dict:
        lenv = LocalNeiboroughCounter(structure, self.background, self.cutoff)
        return {"environment_counts": lenv.get_environments()}


class NTMObserver(BaseObserver):
    def __init__(
        self,
        atoms: "ase.Atoms",
        tetra_centres: list[list],
        non_tm_symbols=("Li",),
        background_species=("O",),
        interval: int = None,
        return_type=dict,
    ):
        """Instantiate an BaseLocalEnvObserver object"""
        tag = "NTMObserver"
        super().__init__(tag=tag, interval=interval, return_type=return_type)
        self.lenv = NTMChannelCounter(
            atoms, tetra_centres, non_tm_symbols, background_species
        )
        self.atoms = atoms

    def get_observable(self, structure: Atoms) -> dict:
        return {"N-TMs": self.lenv.count_env(structure)}


class PercolationObserver(BaseObserver):
    def __init__(
        self,
        atoms: "ase.Atoms",
        tetra_centres: list[list],
        non_tm_symbols=("Li",),
        background_species=("O",),
        interval: int = None,
        return_type=float,
    ):
        """Instantiate an BaseLocalEnvObserver object"""
        tag = "PercolationObserver"
        super().__init__(tag=tag, interval=interval, return_type=return_type)
        self.lenv = PercolationTracker(
            atoms, tetra_centres, non_tm_symbols, background_species
        )
        self.atoms = atoms

    def get_observable(self, structure: Atoms) -> float:
        return self.lenv.count_percolation(structure)
