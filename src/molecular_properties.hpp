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