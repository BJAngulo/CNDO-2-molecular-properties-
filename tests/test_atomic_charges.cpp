#include "molecular_properties.hpp"

#include <armadillo>
#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>

int main() {
    // Two hydrogen atoms
    std::vector<Atom> atoms = {
        {1, 0.0, 0.0, 0.0},
        {1, 1.3984, 0.0, 0.0}
    };

    // Construct a simple test SCF result
    SCFResult scf_result;

    scf_result.P_alpha = arma::mat(2, 2, arma::fill::zeros);
    scf_result.P_beta = arma::mat(2, 2, arma::fill::zeros);

    // One electron assigned to each hydrogen
    scf_result.P_alpha(0, 0) = 0.5;
    scf_result.P_alpha(1, 1) = 0.5;

    scf_result.P_beta(0, 0) = 0.5;
    scf_result.P_beta(1, 1) = 0.5;

    scf_result.basis_to_atom = {0, 1};

    // Calculate atomic charges 
    arma::vec charges = calculate_atomic_charges(atoms, scf_result);

    // Both hydrogen atoms should be neutral
    assert(charges.n_elem == 2);
    assert(std::abs(charges(0)) < 1e-8);
    assert(std::abs(charges(1)) < 1e-8);

    std::cout << "H2 atomic charges test passed\n";

    // Test HF with an assymetric electron distribution
    std::vector<Atom> hf_atoms = {
        {1, 0.0, 0.0, 0.0},
        {9, 1.7, 0.0, 0.0}
    };

    SCFResult hf_result;

    hf_result.P_alpha = arma::mat(5, 5, arma::fill::zeros);
    hf_result.P_beta = arma::mat(5, 5, arma::fill::zeros);

    // H has one basis function; F has four (2s, 2px, 2py, 2pz)
    hf_result.basis_to_atom = {0, 1, 1, 1, 1};

    // Assign 0.7 electrons to H and 7.3 to F 
    hf_result.P_alpha.diag() = 
        arma::vec ({0.35, 1.0, 0.9, 0.9, 0.85});

    hf_result.P_beta.diag() = 
        arma::vec ({0.35, 1.0, 0.9, 0.9, 0.85});

    arma::vec hf_charges = calculate_atomic_charges(hf_atoms, hf_result);

    assert(std::abs(hf_charges(0) - 0.3) < 1e-8);
    assert(std::abs(hf_charges(1) + 0.3) < 1e-8);
    assert(std::abs(arma::sum(hf_charges)) < 1e-8);

    std::cout << "HF charge conservation test passed\n";


    return 0;
}