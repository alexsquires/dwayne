from utilities import read_vasp_calculations
from pymatgen.core import Species
from pymatgen.core import Structure
from pymatgen.entries.computed_entries import ComputedStructureEntry as CSE
from skopt import gp_minimize
import numpy as np
from pprint import pprint

from typing import List
from dataclasses import dataclass

DEFAULT_CALLS = 20
NOISE = 1e-3
ORDERED = 1.0
NEUTRAL = 0

@dataclass
class ChargeNeutrality:
    s_ij: List[float]
    q_tot: List[int] 
    cutoffs: List[float]
    charges: List[int]

    def map(self, mag: float):
        """
        Args:
            mag: magnetic moment for site to map
        Returns:
            charge corresponding to the magnetic moment
        """ 
        if mag < np.min(self.cutoffs):
            return self.charges[0]
        elif mag > np.max(self.cutoffs):
            return 10**10
        for i_uB in range(1, len(self.cutoffs)):
            if self.cutoffs[i_uB-1] < mag < self.cutoffs[i_uB]:
                return self.charges[i_uB]

    def set_parameters(self, current_params):
        """Sets current cutoffs."""
        self.cutoffs = current_params

    def evaluate(self):
        """Calculate number of non-charge-balanced structures."""
        total = 0
        for index in range(len(self.s_ij)):
            struct = self.s_ij[index]
            s_charge = 0
            for mag in struct:
                s_charge += self.map(np.abs(mag))
            if (s_charge + self.q_tot[index]) != 0:
                total += 1
        return total

@dataclass
class BayesianChargeAssigner:
    data: List["pymatgen.entries.ComputedStructureEntry"]
    known_species: List["pymatgen.core.Species"]
    variable_species: List["pymatgen.core.Species"]
    mag_range: List[float]
    domain: List[List[float]]
    shell: str = "d"
    verbose: bool = True


    def __post_init__(self):
        self.n_charge_states = len(self.variable_species)
        self.acq_fncs = ["EI", "LCB", "PI", "gp_hedge"]
        self.acq_optimizer = "sampling"
        self.desired_ox_states = [i.oxi_state for i in self.variable_species]

        assert len(set([i.element for i in self.variable_species])) == 1
        self.variable = self.variable_species[0].element.name
        self.variable_charges = [i.to_pretty_string() for i in self.variable_species]

        self.cn = None
        self.optimization_results = None

        if self.verbose:
            print(f"""
            read in {len(self.data)} training entries. The species to assign are:
            {self.variable} with oxidation states {self.desired_ox_states}.
            """)

    def objective_fnc(self, current_params):
        """Returns number of structures which are not charge-balanced."""
        self.cn.set_parameters(current_params)
        return self.cn.evaluate()

    def optimize_charges(self):
        """
        Returns:
            dict: acquisition function and its number of non-charge-balanced-structures, upper cutoffs
        """
        
        optimization_results = dict()

        # for all data, get a list of magnetic moments and
        # the sum of the fixed charges on each structure

        s_ij = []
        q_ij = []
        for td in self.data:
            mags = []
            for site in td.structure:
                if site.species_string == self.variable:
                   mags.append(site.properties["magmom"])
            s_ij.append(mags)
            q_ij.append(self.add_up_known_charges(td.structure))

        init_cutoffs = [
            self.mag_range[0]
            + self.mag_range[1] / float(self.n_charge_states) * i
            for i in range(self.n_charge_states)
        ]

        self.cn = ChargeNeutrality(s_ij, q_ij, init_cutoffs, self.desired_ox_states)
    
        for fnc in self.acq_fncs:
            residual = gp_minimize(
                self.objective_fnc,
                self.domain,
                acq_func=fnc,
                n_calls=DEFAULT_CALLS,
                acq_optimizer=self.acq_optimizer,
                noise=NOISE,
            )
            optimization_results[fnc] = [residual.fun, residual.x]
        return optimization_results


    def assign_charges(self):
        """Returns dictionary of charge-assigned, charge-balanced structures."""
        self.optimization_results = self.optimize_charges()
        sorted_cutoffs = list(
            sorted(self.optimization_results.items(), key=lambda item: item[1][1])
        )
        optimized_cutoffs = sorted_cutoffs[0][1][1]
        self.optimized_cutoffs = optimized_cutoffs
        return self.do_charge_balancing()

    def do_charge_balancing(self):
        """Takes Bayesian-optimized cutoffs and assigns charges."""
        known_oxi_states = {
            i.element.name: {i.to_pretty_string(): ORDERED} for i in self.known_species
        }
        charge_balanced = []

        for s_data in self.data:
            structure = s_data.structure
            structure.remove_oxidation_states()
            structure.replace_species(known_oxi_states)
            variable_charge_sites = self.get_site_index(structure, self.variable)
            magmoms = [ site.properties["magmom"]
                for site in structure
                if site.species_string == self.variable
            ]

            for i, mag in enumerate(magmoms):
                site = variable_charge_sites[i]
                if mag < np.min(self.optimized_cutoffs):
                    structure[site] = self.variable_charges[0]
                elif mag > np.max(self.optimized_cutoffs):
                    structure[site] = self.variable_charges[-1]
                else:
                    for i_cutoff in range(len(self.optimized_cutoffs) - 1):
                        if self.optimized_cutoffs[i_cutoff] < mag < self.optimized_cutoffs[i_cutoff + 1]:
                            structure[site] = self.variable_charges[i_cutoff + 1]
            if structure.charge == NEUTRAL:
                charge_balanced.append(s_data)
        return charge_balanced

    @staticmethod
    def get_site_index(structure, species):
        """Simple method to get site indices within a structure."""
        sites = [
            i for i in range(len(structure)) if structure[i].species_string == species
        ]
        return sites

    def add_up_known_charges(self, structure):
        """Calculates structure charge for the known species."""
        total_charge = 0
        default_ox = {i.element.name: i.oxi_state for i in self.known_species}
        for i, site in enumerate(structure):
            if site.species_string in default_ox:
                total_charge += default_ox[site.species_string]
        return total_charge

def filter_training_data(training_data):
    new_data = []
    for td in training_data:
        Mns = [site.properties["magmom"] for site in td.structure if site.species_string == "Mn"]
        if all (i > 2.5 for i in Mns):
            new_data.append(td)
    return new_data

import json

with open("pymatgen_entries.json", "r") as f:
    training_data = json.load(f)

training_data = [CSE.from_dict(i) for i in training_data]

parsed_training_data = []
for td in training_data:
    try:
        if td.structure.site_properties["magmom"]:
            parsed_training_data.append(td)
    except:
        None
    

bca = BayesianChargeAssigner(
    data=parsed_training_data,
    known_species=[Species("Li", 1), Species("O", -2), Species("F", -1), Species("Ti", 4), Species("Cr", 3), Species("Fe", 3), Species("Ni", 2)],
    variable_species=[Species("Mn", 4), Species("Mn", 3), Species("Mn", 2)],
    mag_range=(0, 6),
    domain=[(3, 3.5), (3.75, 4.25), (4.01, 6)],
)

cba = bca.assign_charges()
