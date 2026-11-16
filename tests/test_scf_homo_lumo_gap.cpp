#include "molecular_properties.hpp"

#include <armadillo>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

int main() {
    // HF molecular geometry (Bohr)
    std::vector<Atom> hf_atoms = {
        {1, 0.0, 0.0, 0.0},
        {9, 1.733, 0.0, 0.0} 
    };

    // HF has 8 valence electrons
    const int num_alpha_electrons = 4;
    const int num_beta_electrons = 4;

    // Run the CNDO/2 SCF calculation
    SCFResult hf_result = run_scf(
        hf_atoms,
        num_alpha_electrons,
        num_beta_electrons
    );

    // Retrieve final orbital energies 
    const arma::vec& alpha_energies =
        hf_result.orbital_energies_alpha;

    const arma::vec& beta_energies =
        hf_result.orbital_energies_beta;

    // Calculate spin-resolved HOMO-LUMO gaps
    double alpha_gap = calculate_homo_lumo_gap(
        alpha_energies,
        num_alpha_electrons
    );

    double beta_gap = calculate_homo_lumo_gap(
        beta_energies,
        num_beta_electrons
    );

    // Display results
    std::cout << "HF alpha HOMO-LUMO gap: "
              << alpha_gap << " eV\n";

    std::cout << "HF beta HOMO-LUMO gap: "
              << beta_gap << " eV\n";

    // Verify the gaps are finite and nonnegative 
    if (!std::isfinite(alpha_gap) ||
        !std::isfinite(beta_gap) ||
        alpha_gap < 0.0 ||
        beta_gap < 0.0) {
        throw std::runtime_error(
            "HF HOMO-LUMO gaps are invalid"
        );
    }

    // Closed-shell HF should have matching alpha and beta gaps
    const double tolerance = 1e-6;

    if (std::abs(alpha_gap - beta_gap) > tolerance) {
        throw std::runtime_error(
            "HF alpha and beta HOMO-LUMO gaps do not macth"
        );
    }

    std::cout 
        << "HF SCF HOMO-LUMO integration test passed!\n";

    return 0;
}