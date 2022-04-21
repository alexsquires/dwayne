"""
Module for handling local environment tracking
"""
from collections import Counter
import numpy as np
import warnings

from ase.neighborlist import build_neighbor_list
from ase.data import atomic_numbers
from ase import Atoms
import ase

from .percolate import SiteNode


def build_nn_list(atoms, background_species=None, cutoff=3.0) -> dict[int, list[int]]:
    """
    Build NN mapping for each sites excluding any background species

    Args:
        atoms (ase.Atoms): Input template Atoms
        background_species: A list of background species to be excluded from all considerations
        cutoff: Cutoff distance for neighbor list

    Returns:
        a dictionary with keys being the indices of the cation site and values
        being list of the neighbors.
    """
    if background_species is None:
        background_species = []
    cutoffs = [
        0.01 if atom.symbol in background_species else cutoff / 2 for atom in atoms
    ]

    cutoffs = [1 for atom in atoms]
    nlist = build_neighbor_list(
        atoms, bothways=True, cutoffs=cutoffs, self_interaction=False
    )

    tracked_sites = {}
    for atom in atoms:
        if atom.symbol not in background_species:
            nnlist = nlist.get_neighbors(atom.index)[0].tolist()
            tracked = [n for n in nnlist if atoms[n].symbol if n != atom.index]
            tracked = [
                n
                for n in tracked
                if atoms.get_distance(n, atom.index, mic=True) < cutoff
            ]
            tracked_sites[atom.index] = tracked
    return tracked_sites


class LocalNeiboroughCounter:
    """
    Class for tacking the counts of local environments.

    For a list of species, track and count the number of neighbours
    """

    def __init__(self, atoms, backgrounds, cutoff=3.0):
        """
        initiate the object give a template atoms and background speices
        """
        self.site_mapping = build_nn_list(atoms, backgrounds, cutoff)
        self.atoms = atoms

        # get all the unique speices in the structure
        bkg_numbers = [atomic_numbers[bkg] for bkg in backgrounds]
        self.unique_species = [
            num for num in np.unique(atoms.numbers) if num not in bkg_numbers
        ]
        self.unique_species.sort()

        # store the number of unique species
        self.nspecies = len(self.unique_species)

        # highest number of nearset neighbors any site has
        self.max_nn = max([len(val) for val in self.site_mapping.values()])

        self.env_counts = np.zeros((self.nspecies, self.nspecies, self.max_nn + 1))
        self.ncounted = 0

    def get_environments(self):
        output = {x: [] for x in self.unique_species}
        for center, vertex in self.site_mapping.items():
            center = self.atoms[center].number
            output[center].append([self.atoms[i].number for i in vertex])
        return output


class NTMChannelCounter:
    """
    Counter for N-TM channels
    """

    def __init__(
        self,
        atoms: "ase.Atoms",
        tetra_centres: list,
        non_tm_symbols: tuple[str] = ("Li",),
        background_species: tuple[str] = ("O",),
    ):
        """
        Initialise a counter instance
        """
        self.atoms = atoms
        self.tetra_centres = tetra_centres

        self.non_tm_numbers = [atomic_numbers[tmp] for tmp in non_tm_symbols]
        self.background_numbers = [atomic_numbers[tmp] for tmp in background_species]

        # Build the extended atoms with the tetrahedral sites included
        self.extended_atoms = self._build_extended_atoms()
        # Create the NN links centred on the tetrahedral sites
        self.tetra_centre_nn = self.build_tetra_centre_nn()

        if not all(len(nn) == 4 for nn in self.tetra_centre_nn.values()):
            warnings.warn(
                "Some of the tetrahedral sites does not have four closest neiboroughs."
            )

        self.env_counts = np.zeros(5, dtype=int)
        self.ncounted = 0
        self.n_bouded_sites = 0

    def reset(self):
        """Reset the channel counts"""
        self.env_counts[:] = 0
        self.ncounted = 0
        self.n_bouded_sites = 0

    def _build_extended_atoms(self):
        """
        Build an extended ase.Atoms

        The extended atoms includes the tetrahedral sites appended to the original Atoms.
        This allows the index of the extended atoms to be identical to those in the oringal
        atoms.
        """
        extended_atoms = self.atoms.copy()
        extended_atoms.set_calculator(None)
        tetra_sites = Atoms(
            symbols="X{}".format(len(self.tetra_centres)),
            cell=extended_atoms.cell,
            positions=self.tetra_centres,
        )
        extended_atoms.extend(tetra_sites)
        return extended_atoms

    def build_tetra_centre_nn(self, cutoff=2) -> list[int]:
        """
        Build NN mapping for the surrounding tetrahedral sites

        Args:
            atoms (ase.Atoms): Input template Atoms

        Returns:
            a dictionary with keys being the indices of the cation site and values
            being list of the neighbors.
        """

        extended_atoms = self.extended_atoms

        cutoffs = [
            0.01 if atom.number in self.background_numbers else cutoff / 2
            for atom in extended_atoms
        ]

        # Build the NN list for the extended atoms
        nlist = build_neighbor_list(extended_atoms, cutoffs=cutoffs, bothways=True)

        tetra_sites = {}
        numbers = extended_atoms.numbers
        for atom in extended_atoms:
            if atom.symbol == "X":
                nnlist = nlist.get_neighbors(atom.index)[0].tolist()
                only_cation = []
                for nidx in nnlist:
                    # Add the index of the atoms if it satisfy the condition
                    if (
                        numbers[nidx]
                        not in self.background_numbers  # Not a background ion
                        and nidx != atom.index  # Not myself
                        and extended_atoms.get_distance(nidx, atom.index, mic=True)
                        < cutoff
                    ):  # Lower than the cut off
                        only_cation.append(nidx)
                # Use the position vector of the tetrahedral site as the key for the dictionary
                tetra_sites[atom.index] = only_cation
        return tetra_sites

    def count_env(self, atoms: "ase.Atoms") -> list[int]:
        """
        Count the environment and category them into 0-TM, 1-TM, 2-TM, 3-TM, 4-TM cases

        Increment the internal environment counters accordingly

        Args:
            atoms: the atoms to be counted

        Returns:
            A list of N-TM counts where the N is the index
        """
        tetra_centres = self.tetra_centre_nn
        numbers = atoms.numbers

        site_tm_counts = {}
        for tidx, nindices in tetra_centres.items():
            non_tm_count = 0
            for idx in nindices:
                if numbers[idx] in self.non_tm_numbers:
                    non_tm_count += 1
            # number of TM atoms is 4 - N of non-tm atoms
            site_tm_counts[tidx] = 4 - non_tm_count

        # Count the appearance of N-TM environments
        counter = Counter(site_tm_counts.values())

        # Increment internal counter
        for i in range(5):
            self.env_counts[i] += counter[i]
        self.ncounted += 1

        # locate bound between non-TM atoms through the 0-TM channels
        # The bonds can be used in further analyse for determining the percolation networks

        bonds = []  # Indices of non-TM sites that are connected by 0-TM channels
        bonded_sites = set()
        for tidx, nindices in tetra_centres.items():
            if site_tm_counts[tidx] == 0:
                bonds.append(nindices)
                bonded_sites.update(nindices)
        self.n_bouded_sites += len(bonded_sites)

        return counter


class PercolationTracker(NTMChannelCounter):
    """
    Tracker for the percolation networks
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ratios = []

    def count_env(self, atoms: ase.Atoms) -> float:
        return self.count_percolation(atoms)

    def count_percolation(self, atoms: ase.Atoms) -> float:
        """
        Find the percolation network
        """
        # Typical lengths of bond and lattice vectors
        bond_typical = 5
        lattice_typical = 10

        
        tetra_sites = (
            self.tetra_centre_nn
        )  # Mapping from the tetrahedral sites to the Li atoms
        all_nodes = {
            atom.index: SiteNode(
                atom.position, metadata={"index": atom.index, "atom": atom}
            )
            for atom in atoms
            if atom.number == 3
        }  # Look up dictionary of all the nodes
        # Iterate though all the tetrahedral sites
        numbers = atoms.numbers
        # Check each tetrahedral sites
        for site_index, link_indices in tetra_sites.items():
            # Skip if not a 0-TM Channel
            if not np.all(numbers[link_indices] == 3):
                continue
            # Add bonds between all possible combination of the sites
            for ia, ib in [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]]:
                nodea = all_nodes[link_indices[ia]]
                nodeb = all_nodes[link_indices[ib]]
                # If the two nodes already below to the same network - check if add the bond would make percolation work
                if nodea.root_node == nodeb.root_node:
                    da = nodea.get_displacement_to_root()
                    db = nodeb.get_displacement_to_root()
                    dd = db - da
                    # Longer than typical bond length - check which direction does the percolation occur
                    if np.linalg.norm(dd) > bond_typical:
                        for i, label in zip(range(3), "xyz"):
                            if abs(dd[i]) > lattice_typical:
                                # print(f'percolation in {label}')
                                nodea.network_data[f"percolate_{label}"] = True
                else:
                    # Merge the network - grow the one with larger size
                    if nodea.tot_num_nodes > nodeb.tot_num_nodes:
                        nodea.process_bond_to(nodeb)
                    else:
                        nodeb.process_bond_to(nodea)

        # Analyse the network data
        def is_percolating(node):
            names = ["percolate_x", "percolate_y", "percolate_z"]
            for name in names:
                if name not in node.network_data:
                    return False
            return True

        all_roots = [
            node for node in all_nodes.values() if node.is_root and is_percolating(node)
        ]
        all_roots.sort(key=lambda x: x.tot_num_nodes, reverse=True)

        # Compute the ratio of percolating Li atoms
        num_tot = sum(root_node.tot_num_nodes for root_node in all_roots)
        num_li = Counter(atoms.numbers)[3]  # Number of lithium in the formula
        ratio = num_tot / num_li
        self.ncounted += 1
        self.ratios.append(ratio)
        return ratio

    @property
    def mean_ratio(self):
        """Mean ratio of counted percentage"""
        return np.mean(self.ratios)

    @property
    def std_ratio(self):
        """Mean ratio of counted percentage"""
        return np.std(self.ratios)
