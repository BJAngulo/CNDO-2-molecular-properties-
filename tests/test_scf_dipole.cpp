#include "molecular_properties.hpp"

#include <stdexcept>
#include <vector>
#include <iostream>
#include <armadillo> 
#include <cmath>

int main() {

    // Test 1: H2 dipole moment from actual SCF results
    std::vector<Atom> h2_atoms = {
        {1, 0.0, 0.0, 0.0},
        {1, 1.3984, 0.0, 0.0}
    };

    // Run CNDO/2 SCF for H2 ( 1 alpha, 1 beta electron)
    SCFResult h2_result = run_scf(h2_atoms, 1, 1);

    // Calculate atomic charges from SCF density matrices
    arma::vec h2_charges = 
        calculate_atomic_charges(h2_atoms, h2_result);

    // Calculate dipole moment from the SCF derived charges 
    arma::vec h2_dipole =
        calculate_dipole_moment(h2_atoms, h2_charges);

    // Verify the calculated dipole is finite 
    if (!h2_dipole.is_finite()) {
        throw std::runtime_error(
            "H2 dipole contains non-finite values"
        );
    }

    // Verify H2 has approximately zero dipole moment 
    const double tolerance = 1e-6;

    if (arma::norm(h2_dipole, 2) > tolerance) {
        throw std::runtime_error(
            "H2 SCF dipole moment should be approximately zero"
        );
    }

    // Verify conservation of total molecular charge 
    if (std::abs(arma::sum(h2_charges)) > tolerance) {
        throw std::runtime_error(
            "H2 total atomic charge is not conserved"
        );
    }

    std::cout << "H2 SCF dipole integration test passed!\n";

    // Test 2: HF dipole moment from actual SCF results 
    std::vector<Atom> hf_atoms = {
        {1, 0.0, 0.0, 0.0},  // Hydrogen
        {9, 1.733, 0.0, 0.0}  // Fluorine
    };

    // HF has 8 valence electrons: 4 alpha and 4 beta 
    SCFResult hf_result = run_scf(hf_atoms, 4, 4);

    // Calculate atomic charges from the SCF density
    arma::vec hf_charges =
        calculate_atomic_charges(hf_atoms, hf_result);

    // Calculate the charge-based molecular dipole
    arma::vec hf_dipole = 
        calculate_dipole_moment(hf_atoms, hf_charges);

    // Display SCF-derived charges and dipole
    std::cout << "\nHF atomic charges (e):\n";
    hf_charges.print();

    std::cout << "\nHF dipole vector (Debye):\n";
    hf_dipole.print();

    std::cout << "HF dipole magnitude (Debye):"
              << arma::norm(hf_dipole, 2) << '\n';

    // Verify HF charge and dipole vector dimensions
    if (hf_charges.n_elem != 2 || hf_dipole.n_elem != 3) {
        throw std::runtime_error(
            "HF charge or dipole vector has incorrect dimensions"
        );
    }

    // Verify numerical values are finite 
    if (!hf_charges.is_finite() || !hf_dipole.is_finite()) {
        throw std::runtime_error(
            "HF calculation contains non-finite values"
        );
    }    

    // Verify the expected polarity 
    if (hf_charges(0) <= 0.0 || hf_charges(1) >= 0.0) {
        throw std::runtime_error(
            "HF atomic charge signs are incorrect"
        );
    }

    // Fluorine lies along positive x, so the dipole moment point towards -x
    if (hf_dipole(0) >= 0.0 ||
        std::abs(hf_dipole(1)) > tolerance ||
        std::abs(hf_dipole(2)) > tolerance) {
        throw std::runtime_error(
            "HF dipole direction is incorrect"
        );
    }

    std::cout << "HF SCF dipole integration test passed!\n";
            
    return 0;
}
