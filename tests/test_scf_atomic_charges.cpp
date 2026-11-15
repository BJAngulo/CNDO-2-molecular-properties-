#include "molecular_properties.hpp"

#include <armadillo>
#include <cmath>
#include <iostream>
#include <vector>

int main() {
    
    // H2 molecule: two hydrogen atoms
    std::vector<Atom> atoms = {
        {1, 0.0, 0.0, 0.0},
        {1, 1.3984, 0.0, 0.0}
    };

    // Run the actual CNDO/2 SCF calculation 
    SCFResult result = run_scf(atoms, 1, 1);

    // Calculate atomic charges from converged density
    arma::vec charges = calculate_atomic_charges(atoms, result);

    // Verify the number if atomic charges 
    if (charges.n_elem != 2) {
        throw std::runtime_error(
            "Incorrect number of atomic charges"
        );
    }

    // Verify that both hydrogen atoms are neutral 
    const double tolerance = 1e-6;

    if (std::abs(charges(0)) > tolerance ||
        std::abs(charges(1)) > tolerance) {
            throw std::runtime_error(
                "H2 atomic chargers are not approximately zero"
            );
        }

    // Verfify charge conservation 
    if (std::abs(arma::sum(charges)) >  tolerance) {
        throw std::runtime_error(
            "Total molecular charge is not conserved"
        );
    }

    std::cout << "H2 SCF atomic charge integration test passed\n";

    return 0;
}
