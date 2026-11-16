#include "molecular_properties.hpp"

#include <armadillo>
#include <cassert>
#include <cmath>
#include <iostream>
#include <stdexcept>

int main() {
	// Example orbital energies in eV, sorted in ascending order 
	arma::vec orbital_energies = {
		-45.5977,
		-23.4920,
		-21.2479,
		-21.2479,
		  8.8552
  	};

	// Four occupied alpha orbitals
	int num_alpha_electrons = 4;

	// Calculate the HOMO-LUMO gap 
	double gap = calculate_homo_lumo_gap(
		orbital_energies,
		num_alpha_electrons
	);

	// Expected gap = LUMO energy - HOMO energy 
	double expected_gap = 8.8552 - (-21.2479);

	assert(std::abs(gap - expected_gap) < 1e-6);

	std::cout << "HOMO-LUMO gap: "
		  << gap <<" eV" <<  std::endl;

	std::cout << "HOMO-LUMO gao test passed!"
		  << std::endl;

	return 0;
}


