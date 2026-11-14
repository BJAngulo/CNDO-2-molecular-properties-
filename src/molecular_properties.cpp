#include "molecular_properties.hpp"

#include <stdexcept>

arma::vec calculate_atomic_charges(
    const std::vector<Atom>& atoms,
    const SCFResult& scf_result
)
{
    // Number of atoms in the molecule
    const size_t num_atoms = atoms.size();

    // Initialize atomic electron populations to zero
    arma::vec populations(num_atoms, arma::fill::zeros);

    // Total electron density matrix 
    arma::mat P_total = scf_result.P_alpha + scf_result.P_beta;

    // Verify density matrix and basis mapping dimensions
    const size_t num_basis = scf_result.basis_to_atom.size();

    if (P_total.n_rows != num_basis || P_total.n_cols != num_basis){
        throw std::invalid_argument(
            "Density matrix dimensions do not match basis functions"
        );
    }

    // Accumulate electron populations for each atom
    for (size_t mu = 0; mu < num_basis; mu++) {

        // Identify the atom associated with basis function 
        const size_t atom_index = scf_result.basis_to_atom[mu];

        if (atom_index >= num_atoms) {
            throw std::out_of_range("Invalid basis-to-atom mapping");
        }

        // Add orbital electron population to its atom
        populations(atom_index) += P_total(mu, mu);
    }

    // Calculate atomic partial charges
    arma::vec charges(num_atoms, arma::fill::zeros);

    for (size_t atom_index = 0; atom_index < num_atoms; atom_index++) {

        const int atomic_number = atoms[atom_index].element;
        const int valence_charge = valence_electrons(atomic_number);

        charges(atom_index) = 
            static_cast<double>(valence_charge) - populations(atom_index);
    }

    return charges;
}