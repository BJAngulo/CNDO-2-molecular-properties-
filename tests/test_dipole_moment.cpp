#include "molecular_properties.hpp"

#include <armadillo>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

int main() {

    // Test 1: H2 should have zero dipole moment 
    std::vector<Atom> h2_atoms = {
        {1, 0.0, 0.0, 0.0},
        {1, 1.3984, 0.0, 0.0} 
    };

    // Neutral atomic charges
    arma::vec h2_charges = {0.0, 0.0};

    // Calculate the molecular dipole 
    arma::vec h2_dipole = 
        calculate_dipole_moment(h2_atoms, h2_charges);

    const double tolerance = 1e-6;

    // Verify all three Cartesian components 
    if (h2_dipole.n_elem != 3 ||
        arma::norm(h2_dipole, 2) > tolerance) {
        throw std::runtime_error(
            "H2 dipole moment should be zero"
        );
    }

    std::cout << "H2 dipole moment test passed!\n";

    // Test 2: HF should have a nonzero dipole moment
    std::vector<Atom> hf_atoms = {
        {1, 0.0, 0.0, 0.0},
        {9, 0.92, 0.0, 0.0}
    };

    // Illustrate partial charges
    arma::vec hf_charges = {0.30, -0.30};

    // Calculate the molecular dipole 
    arma::vec hf_dipole =
        calculate_dipole_moment(hf_atoms, hf_charges);

    // Expected x-component in Debye
    const double expected_mu_x = -0.30 * 0.92 * 2.541746;

    // Verify the x-component
    if (std::abs(hf_dipole(1)) > tolerance ||
        std::abs(hf_dipole(2)) >  tolerance) {
        throw std::runtime_error(
            "HF dipole x-component is incorrect"
        );
    }
        
    std::cout << "HF dipole moment test passed!\n";

    // Test 3: Reject mismatched atomic charges 
    arma::vec invalid_charges = {0.30};

    bool exception_thrown = false;

    try {
        calculate_dipole_moment(hf_atoms, invalid_charges);
    }
    catch (const std::invalid_argument&) {
        exception_thrown = true;
    }

    if (!exception_thrown) {
        throw std::runtime_error(
            "Expected exception for mismatched atomic charges"
        );
    }

    std::cout << "Dipole input validation test passed!\n";
    
    return 0;
}