#ifndef CNDO2_HELPERS_HPP
#define CNDO2_HELPERS_HPP

#include <string>
#include <vector>
#include <armadillo>

// Structs for storing information about atoms and basis functions
struct Atom {
    int element;
    double x;
    double y;
    double z;
};

struct PrimitiveGaussian {
    double exponent;
    double contraction_coefficient;
    double normalization = 0.0;
};

struct BasisFunction {
    int atomic_number;

    double x, y, z; // center of orbital
    int l, m, n;    // Cartesian angular powers

    std::vector<PrimitiveGaussian> primitives;
};

// Store results needed for the analytical gradient calculation
struct SCFResult {
    arma::mat P_alpha;
    arma::mat P_beta;
    arma::mat S;
    arma::mat gamma;
    arma::mat H_core;
    
    arma::vec orbital_energies_alpha;
    arma::vec orbital_energies_beta;

    std::vector<BasisFunction> basis_functions;
    std::vector<size_t> basis_to_atom;

    double E_electronic;
    double E_nuclear;
    double E_total;
};

SCFResult run_scf(
    const std::vector<Atom>& atoms,
    int num_alpha_electrons,
    int num_beta_electrons
);

// Read contracted Gaussian primitives from JSON  basis file
std::vector<PrimitiveGaussian> read_basis_file(const std::string& filename);

// Primitive Gaussian normalization functions
double normalize_s(double alpha);
double normalize_p(double alpha);

// Gaussian overlap functions
double primitive_overlap_3d(
    const BasisFunction& basis_a,
    const BasisFunction& basis_b,
    const PrimitiveGaussian& primitive_a,
    const PrimitiveGaussian& primitive_b
);

arma::vec primitive_overlap_derivative_3d(
    const BasisFunction& basis_a,
    const BasisFunction& basis_b,
    const PrimitiveGaussian& primitive_a,
    const PrimitiveGaussian& primitive_b
);

double contracted_overlap(
    const BasisFunction& basis_a,
    const BasisFunction& basis_b
);

arma::vec contracted_overlap_derivative(
    const BasisFunction& basis_a,
    const BasisFunction& basis_b
);

// Boys function 
double boys_function(double T);

// Primitive electron repulsion integral
double primitive_repulsion(
    double alpha_k,
    double alpha_k_prime,
    double beta_l,
    double beta_l_prime,
    double R_AB_squared
);

// Contracted electron repulsion integral 
double contracted_repulsion(
    const std::vector<PrimitiveGaussian>& primitives_A,
    const std::vector<PrimitiveGaussian>& primitives_B,
    double R_AB_squared 
);

// Contracted electron repulsion derivative 
arma::vec contracted_repulsion_derivative(
    const std::vector<PrimitiveGaussian>& primitives_A,
    const std::vector<PrimitiveGaussian>& primitives_B,
    const arma::vec& R_A,
    const arma::vec& R_B
);

// Atomic valence electron count 
int valence_electrons(int atomic_number);

// CNDO/2 atomic orbital parameter 
double orbital_parameter(
    int atomic_number,
    int l,
    int m,
    int n
);

// CNDO/2 resonance parameter 
double resonance_parameter(int atomic_number);

#endif 