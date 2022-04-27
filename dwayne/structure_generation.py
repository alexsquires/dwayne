import numpy as np
from typing import Optional, Union
from random import sample
from itertools import product
from pymatgen.core import Structure
from pymatgen.core.sites import PeriodicSite
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.analysis.ewald import EwaldSummation
from pymatgen.analysis.graphs import StructureGraph
from pymatgen.analysis.local_env import CutOffDictNN
from ase import Atoms

"""
This module provides tools for generating delithiated structures from pristine (or at least higher lithium-content) DRX structures.

All functions implemented thus far accept either Pymatgen Structure objects or ASE Atoms objects (the latter is simply converted to and from
Pymatgen under the hood).
"""

def delithiate(structure: Union[Structure, Atoms], target: int, method: str='electrostatic', oxi_states: Optional[dict]=None) -> Union[Structure, Atoms]:
    """
    Generate a delithiated structure.

    Args:
        structure (Union[Structure, Atoms]): The structure to delithiate.
        target (int): The number of Li ions to be left in the structure after delithiation.
        method (str, optional): The method by which to delithiate the structure. Defaults to 'electrostatic'.
        oxi_states (dict, optional): A dictionary to be passed to the add_oxidation_states_by_element method of
        the Pymatgen Structure object when utilising the 'electrostatic' method. Defaults to None.

    Returns:
        Union[Structure, Atoms]: The delithiated structure.

    Notes:
        The 'electrostatic' method removes the highest energy Li ions according to a Ewald summation.
        The 'random' method simply selects all of the Li ions to be removed... randomly.

        If oxi_states is not specified when using the 'electrostatic' method, Pymatgen will attempt to guess the
        relevant oxidation states.
    """

    if type(structure) == Atoms:
        type_ase = True
        structure = AseAtomsAdaptor.get_structure(structure)

    else:
        type_ase = False

    if type(target) != int:
        raise TypeError(f'The target number of lithium atoms must be an integer. {target} is not of type int.')

    struct = structure.copy()
    struct.sort()
    num_Li = int(struct.composition.as_dict()['Li'])

    if method == 'electrostatic':
        if oxi_states is not None:
            struct.add_oxidation_state_by_element(oxi_states)

        else:
            struct.add_oxidation_state_by_guess()
            # Ensure that Pymatgen hasn't given up on assigning oxidation states.
            if sum([species.oxi_state for species in struct.types_of_specie]) == 0:
                raise RuntimeError('Pymatgen failed to assign physically-sensible oxidation states. Please provide these manually.')

        ew = EwaldSummation(struct)
        site_energies = np.array([ew.get_site_energy(idx) for idx in range(0, len(struct)) if struct[idx].species.reduced_formula == 'Li'])
        descending_indices = np.argsort(site_energies)[::-1]

        start_idx = next(idx for idx in range(0, len(struct)) if struct[idx].species.reduced_formula == 'Li')
        struct.remove_sites(start_idx + descending_indices[:num_Li - target])

    elif method == 'random':
        li_indices = [idx for idx in range(0, len(struct)) if struct[idx].species.reduced_formula == 'Li']
        struct.remove_sites(sample(li_indices, num_Li - target))

    else:
        raise RuntimeError(f'{method} is not a valid delithiation method.')

    struct.remove_oxidation_states()
    struct.sort()

    if type_ase:
        struct = AseAtomsAdaptor.get_atoms(struct)

    return struct

def insert_tetrahedral_sites(structure: Union[Structure, Atoms], bond_distance: float, anions: list[str], tol: Optional[float]=None) -> Union[Structure, Atoms]:
    """
    Insert tetrahedral sites into a structure explicitly (as 'X' atoms).

    Args:
        structure (Union[Structure, Atoms]): The structure to add tetrahedral sites to.
        bond_distance (float): The octahedral cation-anion bond distance.
        anions (list[str]): The atomic species comprising the anion sublattice.
        tol (float, optional): A tolerance factor corresponding to the minimum possible distance between two
        tetrahedral sites.

    Returns:
        Structure (Union[Structure, Atoms]): The structure with explicit tetrahedral sites inserted.
    """
    if type(structure) == Atoms:
        type_ase = True
        structure = AseAtomsAdaptor.get_structure(structure)

    else:
        type_ase = False

    if tol is None:
        tol = bond_distance * 0.8

    struct = structure.copy()

    for image in product(*((x, -x) for x in [bond_distance / 2, bond_distance / 2, bond_distance / 2])):
        for atom in struct:
            if atom.species_string in anions:
                coords = atom.coords + image
                frac_coords = struct.lattice.get_fractional_coords(coords)
                accept = True
                for coord in frac_coords:
                    if coord < 0 or coord > 1:
                        accept = False

                if accept:
                    # Ensure nothing already occupies the given tetrahedral site.
                    try:
                        new_site = PeriodicSite('X', frac_coords, struct.lattice)
                        for site in struct:
                            if site.species_string == 'X0+':
                                if site.distance(new_site) < tol:
                                    raise

                        struct.insert(len(struct), 'X', frac_coords, validate_proximity=True)

                    except:
                        pass

    struct.sort()

    if type_ase:
        struct = AseAtomsAdaptor.get_atoms(struct)

    return struct

def pre_relax(structure: Union[Structure, Atoms], bond_distance: float, cations: list[str], tol: float=1.2) -> Union[Structure, Atoms]:
    """
    Pre-relax a delithiated structure (with explicit tetrahedral sites) by moving certain cations that face-share with a 0-TM or 1-TM
    site into that site.

    Args:
        structure (Union[Structure, Atoms]): The delithiated structure to pre-relax.
        bond_distance (float): The octahedral cation-anion bond distance.
        cations (list[str]): The cationic species which may relax into tetrahedral sites.
        tol (float, optional): A tolerance factor which scales the spherical cutoff distance when determining
        the coordination environment around each tetrahedral site. Defaults to 1.2.

    Returns:
        Union[Structure, Atoms]: The pre-relaxed structure.
    """
    if type(structure) == Atoms:
        type_ase = True
        structure = AseAtomsAdaptor.get_structure(structure)

    else:
        type_ase = False

    struct = structure.copy()

    cutoff_dist = np.linalg.norm(np.array([bond_distance / 2, bond_distance / 2, bond_distance / 2])) * tol
    cutoff_dict = {}

    for cation in cations:
        cutoff_dict[('X0+', cation)] = cutoff_dist

    graph = StructureGraph.with_local_env_strategy(struct, CutOffDictNN(cutoff_dict))
    oh_indices, td_strings, td_indices = [], [], []
    for idx in [idx for idx in range(0, len(struct)) if struct[idx].species_string == 'X0+']:
        if graph.get_coordination_of_site(idx) == 1:
            connected_site = graph.get_connected_sites(idx)[0]
            cation_idx = connected_site.index
            # Has this cation already 'migrated' elsewhere?
            if cation_idx in oh_indices:
                continue
            cation_string = connected_site.site.species_string
            oh_indices.append(cation_idx)
            td_strings.append(cation_string)
            td_indices.append(idx)

    print(td_strings)

    for idx in oh_indices:
        struct.replace(idx, 'X')

    for idx, string in zip(td_indices, td_strings):
        struct.replace(idx, string)

    struct.sort()

    if type_ase:
        struct = AseAtomsAdaptor.get_atoms(struct)

    return struct
