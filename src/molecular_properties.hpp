#pragma once 

#include "cndo2_helpers.hpp"

#include <armadillo>
#include <vector>

// Calculate CNDO/2 atomic partial charges 
arma::vec calculate_atomic_charges(
    const std::vector<Atom>& atoms,
    const SCFResult& scf_result
);

// Calculate the charge-based molecular dipole moment in Debye 
arma::vec calculate_dipole_moment(
    const std::vector<Atom>& atoms,
    const arma::vec& charges
);

// Calculate the HOMO-LUMO energy gap for one sprin channel.
// Orbital energies must be sorted in ascending order (eV).
// num_electrons is the number of elecrons in this spin channel.
// Returns the orbital-energy gap in eV.
double calculate_homo_lumo_gap(
    const arma::vec& orbital_energies,
    int num_electrons
);

