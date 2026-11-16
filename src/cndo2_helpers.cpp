#include "cndo2_helpers.hpp"
#include "gaussian_helpers.hpp"

#include <fstream>
#include <stdexcept>
#include <string>
#include <cmath> 
#include <numbers>

#include <armadillo>
#include <nlohmann/json.hpp>

//  read basis file 
std::vector<PrimitiveGaussian> read_basis_file(const std::string& filename) {

    std::ifstream basis_file(filename);

    if (!basis_file) {
        throw std::runtime_error(
            "Could not open basis file: " + filename
        );
    }

    nlohmann::json basis_data = nlohmann::json::parse(basis_file);

    std::vector<PrimitiveGaussian> primitives;

    for (const auto& primitive_json : basis_data["contracted_gaussians"]) {

        PrimitiveGaussian primitive;

        primitive.exponent = primitive_json["exponent"];
        primitive.contraction_coefficient=
            primitive_json["contraction_coefficient"];

        primitives.push_back(primitive);
    }

    return primitives;
}

// Primitive normalization functions 
double normalize_s(double alpha) {
    const double pi = std::acos(-1.0);

    return std::pow(2.0 * alpha / pi, 0.75);
}

double normalize_p(double alpha) {
    const double pi = std::acos(-1.0);

    return std::pow(
        128.0 * std::pow(alpha, 5.0) / std::pow(pi, 3.0),
        0.25
    );
}

// Overlap functions

// Calculate the unnormalized 3D overlap between two primitive Gaussians
double primitive_overlap_3d(
    const BasisFunction& basis_a,
    const BasisFunction& basis_b,
    const PrimitiveGaussian& primitive_a,
    const PrimitiveGaussian& primitive_b
)
{
    double alpha = primitive_a.exponent;
    double beta = primitive_b.exponent;

    // Overlap in x direction 
    double Sx = overlap_1d(
        basis_a.x, basis_b.x,
        alpha, beta,
        basis_a.l, basis_b.l
    );

    // Overlap in y direction 
    double Sy = overlap_1d(
        basis_a.y, basis_b.y,
        alpha, beta,
        basis_a.m, basis_b.m
    );

    // Overlap in z direction
    double Sz = overlap_1d(
        basis_a.z, basis_b.z,
        alpha, beta,
        basis_a.n, basis_b.n
    );

    return Sx * Sy * Sz;
}

// Calculate the 3D primitive overlap derivatives with respect to the Gaussian ceneter A
arma::vec primitive_overlap_derivative_3d(
    const BasisFunction& basis_a,
    const BasisFunction& basis_b,
    const PrimitiveGaussian& primitive_a,
    const PrimitiveGaussian& primitive_b
)
{
    double alpha = primitive_a.exponent;
    double beta = primitive_b.exponent;

    // Original overlap in x, y, z directions
    double Sx = overlap_1d(
        basis_a.x, basis_b.x,
        alpha, beta,
        basis_a.l, basis_b.l
    );

    double Sy = overlap_1d(
        basis_a.y, basis_b.y,
        alpha, beta,
        basis_a.m, basis_b.m
    );

    double Sz = overlap_1d(
        basis_a.z, basis_b.z,
        alpha, beta,
        basis_a.n, basis_b.n
    );

    // Derivative of each 1D overlap 
    double dSx = overlap_1d_derivative(
        basis_a.x, basis_b.x,
        alpha, beta,
        basis_a.l, basis_b.l
    );

    double dSy = overlap_1d_derivative(
        basis_a.y, basis_b.y,
        alpha, beta,
        basis_a.m, basis_b.m
    );

    double dSz = overlap_1d_derivative(
        basis_a.z, basis_b.z,
        alpha, beta,
        basis_a.n, basis_b.n
    );

    // Combine derivatives into a 3D vector 
    arma::vec derivative(3, arma::fill::zeros);

    derivative(0) = dSx * Sy * Sz; 
    derivative(1) = Sx * dSy * Sz;
    derivative(2) = Sx * Sy * dSz;

    return derivative;
}

// Calculate overlap between two contracted basis functions
double contracted_overlap(
    const BasisFunction& basis_a,
    const BasisFunction& basis_b
)
{
    double overlap = 0.0;

    // Loop over primitives in the first basis function
    for (const auto& primitive_a : basis_a.primitives) {

        // Loop over primitives in the second basis function 
        for (const auto& primitive_b : basis_b.primitives) {    

            // Calculate unnormalized primitive overlap
            double S_ab = primitive_overlap_3d(
                basis_a, basis_b,
                primitive_a, primitive_b
            );

            // Weight by contraction coefficients and normalization 
            double contribution = 
            primitive_a.contraction_coefficient *
            primitive_b.contraction_coefficient *
            primitive_a.normalization *
            primitive_b.normalization *
            S_ab;

            overlap += contribution;
        }
    } 

    return overlap;
}

// Calculate contracted overlap derivatives with respect to the center of basis_a
arma::vec contracted_overlap_derivative(
    const BasisFunction& basis_a,
    const BasisFunction& basis_b
)
{
    // Initialize x, y, z derivatives 
    arma::vec derivative(3, arma::fill::zeros);

    // Loop over primitives in basis_a
    for (const auto& primitive_a : basis_a.primitives) {

        // Loop over primitives in basis_b
        for (const auto& primitive_b : basis_b.primitives) {

            // Calculate the primitive overlap derivative
            arma::vec dS_ab = primitive_overlap_derivative_3d(
                basis_a, basis_b,
                primitive_a, primitive_b
            );

            // Weight by contraction coefficients and normalization 
            double weight = 
                primitive_a.contraction_coefficient *
                primitive_b.contraction_coefficient *
                primitive_a.normalization *
                primitive_b.normalization;

            derivative += weight * dS_ab;
        }
    }
    return derivative;
}

// Boys function zeroth order F0(T)
double boys_function(double T) {
  if (T < 1e-8) {
    return 1.0 - T / 3.0 + T * T / 10.0;
  }

  return 0.5 * std::sqrt(std::numbers::pi / T)
             * std::erf(std::sqrt(T));
}

// First-Order Boys Function F1(T) 
double boys_function_1(double T) {

    // Taylor expansion for small T 
    if (T < 1e-5) {
        return 1.0 / 3.0 - T / 5.0 + T * T / 14.0 - T * T * T / 54.0;
    }

    // For larger T, use the relation F1(T) = (F0(T) - exp(-T)) / (2T)
    return (boys_function(T) - std::exp(-T)) / (2.0 * T);
}

// Shared parameters for electron repulsion integrals 
struct RepulsionParameters {
    double C;
    double V_squared;
    double T;
};


// Calculate shared parameters for electron repulsion integrals
RepulsionParameters calculate_repulsion_parameters(
    double alpha_k, 
    double alpha_k_prime,
    double beta_l,
    double beta_l_prime,
    double R_AB_squared
) {

    // Gaussian width parameters
    double sigma_A = 1.0 / (alpha_k + alpha_k_prime);
    double sigma_B = 1.0 / (beta_l + beta_l_prime);

    // Gaussian integration factors
    double U_A = std::pow(std::numbers::pi * sigma_A, 1.5);
    double U_B = std::pow(std::numbers::pi * sigma_B, 1.5);

    // Combined Gaussian parameter
    double V_squared = 1.0 / (sigma_A + sigma_B);

    // Boys function argument
    double T = V_squared * R_AB_squared;

    //  Gaussian prefactor 
    double C =  U_A * U_B 
                * std::sqrt(2.0 * V_squared)
                * std::sqrt(2.0 / std::numbers::pi);
    
    return {C, V_squared, T};
}


// Primitive electron repulsion integral
double primitive_repulsion(
  double alpha_k, 
  double alpha_k_prime,
  double beta_l,
  double beta_l_prime,
  double R_AB_squared
) {
  // Calculate shared Gaussian parameters
  RepulsionParameters params = calculate_repulsion_parameters(
    alpha_k,
    alpha_k_prime,
    beta_l,
    beta_l_prime,
    R_AB_squared
  );

  // Evaluate the primitive electron repulsion integral
  double integral = params.C * boys_function(params.T);
                
  return integral;
}

// Derivative of primitive electron repulsion integral with respect to the position of atom A
arma::vec primitive_repulsion_derivative(
    double alpha_k, 
    double alpha_k_prime,
    double beta_l,
    double beta_l_prime,
    const arma::vec& R_A,
    const arma::vec& R_B
){
    // Vector between atomic centers 
    arma::vec R_diff = R_A - R_B;

    // Squared distance
    double R_AB_squared = arma::dot(R_diff, R_diff);

    // Reuse the shared Gaussian parameters
    RepulsionParameters params = calculate_repulsion_parameters(
        alpha_k,
        alpha_k_prime,
        beta_l,
        beta_l_prime,
        R_AB_squared
    );

    // Derivative prefactor 
    double factor = -2.0
                    * params.C
                    * params.V_squared
                    * boys_function_1(params.T);
                    
    // Return x, y, z derivatives 
    return factor * R_diff;
}


// Contracted electron repulsion integral 
double contracted_repulsion(
  const std::vector<PrimitiveGaussian>& primitives_A,
  const std::vector<PrimitiveGaussian>& primitives_B,
  double R_AB_squared
) {
  double gamma_AB = 0.0;

  for (const auto& k : primitives_A) {
    for (const auto& kp : primitives_A) {
      for (const auto& l : primitives_B) {
        for (const auto& lp : primitives_B) {

          // Calculate one primitive integral
          double integral = primitive_repulsion(
            k.exponent,
            kp.exponent,
            l.exponent,
            lp.exponent,
            R_AB_squared
          );

          // Normalized contraction coefficients 
          double coefficient_product = 
              k.contraction_coefficient * k.normalization
              * kp.contraction_coefficient * kp.normalization
              * l.contraction_coefficient * l.normalization
              * lp.contraction_coefficient * lp.normalization;
           

            gamma_AB += coefficient_product * integral;
        }
      }
    }
  }

  return gamma_AB;
}

// Derivative of contracted electron repulsion integral with respect to the position of atom A 
arma::vec contracted_repulsion_derivative(
    const std::vector<PrimitiveGaussian>& primitives_A,
    const std::vector<PrimitiveGaussian>& primitives_B,
    const arma::vec& R_A,
    const arma::vec& R_B
){
    // Initialize x, y, z derivatives 
    arma::vec derivative(3, arma::fill::zeros);

    // Loop over primitive Gaussian combinations
    for (const auto& k : primitives_A) {
        for (const auto& kp : primitives_A) {
            for (const auto& l : primitives_B) {
                for (const auto& lp : primitives_B) {

                    // Calculate primitive repulsion derivative
                    arma::vec dI = primitive_repulsion_derivative(
                        k.exponent,
                        kp.exponent,
                        l.exponent,
                        lp.exponent,
                        R_A,
                        R_B
                    );

                    // Normalized contraction coefficients
                    double coefficient_product = 
                        k.contraction_coefficient * k.normalization *
                        kp.contraction_coefficient * kp.normalization *
                        l.contraction_coefficient * l.normalization *
                        lp.contraction_coefficient * lp.normalization;

                    // Accumulate x, y, z derivatives 
                    derivative += coefficient_product * dI;
                }
            }
        }
    }

    return derivative;
}

// Return the number of valence electrons for each element based on its atomic number
int valence_electrons(int atomic_number) {

  switch (atomic_number) {
    case 1: return 1;
    case 6: return 4;
    case 7: return 5;
    case 8: return 6;
    case 9: return 7;

    default:
      throw std::runtime_error("Unsupported atomic number");
  }
}

double orbital_parameter(int atomic_number, int l, int m, int n) {
  bool is_s = (l == 0 && m == 0 && n == 0);

  switch (atomic_number) {
    case 1:
      return 7.176;

    case 6: 
      return is_s ? 14.051 : 5.572;

    case 7:
      return is_s ? 19.316 : 7.275;

    case 8:
      return is_s ? 25.390 : 9.111;

    case 9:
      return is_s ? 32.272 : 11.080;

    default:
      throw std::runtime_error("Unsupported atomic number");
  }
}

double resonance_parameter(int atomic_number) {

  switch (atomic_number) {
    case 1: return -9.0;
    case 6: return -21.0;
    case 7: return -25.0;
    case 8: return -31.0;
    case 9: return -39.0;

    default:
      throw std::runtime_error("Unsupported atomic number");
  }
}

SCFResult run_scf(
    const std::vector<Atom>& atoms,
    int num_alpha_electrons,
    int num_beta_electrons) {

    // Reuse the CNDO/2 SCF implementation from HW4

    // Step 1: Load and normalize the STO-3G basis functions 
    // Hydrogen 
    auto h_primitives = read_basis_file("basis/H_s_STO3G.json");

    // Carbon 
    auto c_s_primitives = read_basis_file("basis/C_s_STO3G.json");
    auto c_p_primitives = read_basis_file("basis/C_p_STO3G.json");

    // Nitrogen 
    auto n_s_primitives = read_basis_file("basis/N_s_STO3G.json");
    auto n_p_primitives = read_basis_file("basis/N_p_STO3G.json");

    // Oxygen
    auto o_s_primitives = read_basis_file("basis/O_s_STO3G.json");
    auto o_p_primitives = read_basis_file("basis/O_p_STO3G.json");

    // Fluorine 
    auto f_s_primitives = read_basis_file("basis/F_s_STO3G.json");
    auto f_p_primitives = read_basis_file("basis/F_p_STO3G.json");


    // Verification of three basis files
    std::cout << "H primitives: " << h_primitives.size() << std::endl;

    std::cout << "C s primitives: " << c_s_primitives.size() << '\n';
    std::cout << "C p primitives: " << c_p_primitives.size() << '\n';

    std::cout << "N s primitives: " << n_s_primitives.size() << '\n';
    std::cout << "N p primitives: " << n_p_primitives.size() << '\n';

    std::cout << "O s primitives: " << o_s_primitives.size() << '\n';
    std::cout << "O p primitives: " << o_p_primitives.size() << '\n';

    std::cout << "F s primitives: " << f_s_primitives.size() << '\n';
    std::cout << "F p primitives: " << f_p_primitives.size() << '\n';

    // Normalize hydrogen primitives
    for (PrimitiveGaussian& primitive : h_primitives) {
        primitive.normalization = normalize_s(primitive.exponent); 
    }

    // Normalize carbon primitives 
    for (PrimitiveGaussian& primitive : c_s_primitives) {
        primitive.normalization = normalize_s(primitive.exponent);
    }

    for (PrimitiveGaussian& primitive : c_p_primitives) {
        primitive.normalization = normalize_p(primitive.exponent);
    }

    // Normalize nitrogen primitives 
    for (PrimitiveGaussian& primitive : n_s_primitives) {
        primitive.normalization = normalize_s(primitive.exponent);
    }

    for (PrimitiveGaussian& primitive : n_p_primitives) {
        primitive.normalization = normalize_p(primitive.exponent);
    }

    // Normalize oxygen primitives 
    for (PrimitiveGaussian& primitive : o_s_primitives) {
        primitive.normalization = normalize_s(primitive.exponent);
    }

    for (PrimitiveGaussian& primitive : o_p_primitives) {
        primitive.normalization = normalize_p(primitive.exponent);
    }

    // Normalize fluorine primitives 
    for (PrimitiveGaussian& primitive : f_s_primitives) {
        primitive.normalization = normalize_s(primitive.exponent);
    }

    for (PrimitiveGaussian& primitive : f_p_primitives) {
        primitive.normalization = normalize_p(primitive.exponent);
    }

    std::cout << "C 2p primitive 0 normalization: "
                << c_p_primitives[0].normalization << '\n';

    auto get_s_primitives = [&](int atomic_number)
           -> const std::vector<PrimitiveGaussian>& {

        switch (atomic_number) {
        case 1: return h_primitives;
        case 6: return c_s_primitives;
        case 7: return n_s_primitives;
        case 8: return o_s_primitives;
        case 9: return f_s_primitives;
        default:
            throw std::runtime_error("Unsupported atomic number");
        }
    };

    // Store all basis functions for the molecules
    std::vector<BasisFunction> basis_functions;

    // Map each basis function to its parent atom 
    std::vector<size_t> basis_to_atom;

    for (size_t atom_index = 0; atom_index < atoms.size(); atom_index++) {
        const Atom& atom = atoms[atom_index];

        if(atom.element == 1) {
            
            BasisFunction basis;

            // Atomic number
            basis.atomic_number = atom.element;

            // Center of the orbital
            basis.x = atom.x;
            basis.y = atom.y;
            basis.z = atom.z;

            // Hydrogen 1s orbital 
            basis.l = 0;
            basis.m = 0;
            basis.n = 0;

            // Three normalized STO-3G primitives 
            basis.primitives = h_primitives;

            // Add orbital to basis.function list 
            basis_functions.push_back(basis);
            basis_to_atom.push_back(atom_index);
        }

        else if (atom.element == 6 || atom.element == 7 ||
            atom.element == 8 || atom.element == 9) {

            // select the correct STO-3G primitives 
            const std::vector<PrimitiveGaussian>* s_primitives = nullptr;
            const std::vector<PrimitiveGaussian>* p_primitives = nullptr;

            if (atom.element == 6) {
                s_primitives = &c_s_primitives;
                p_primitives = &c_p_primitives;
            }
            else if (atom.element == 7) {
                s_primitives = &n_s_primitives;
                p_primitives = &n_p_primitives;
            }
            else if (atom.element == 8) {
                s_primitives = &o_s_primitives;
                p_primitives = &o_p_primitives;
            }
            else if (atom.element == 9) {
                s_primitives = &f_s_primitives;
                p_primitives = &f_p_primitives;
            }

            // construct 2s orbital 
            BasisFunction basis;

            basis.atomic_number = atom.element;

            basis.x = atom.x;
            basis.y = atom.y;
            basis.z = atom.z;

            basis.l = 0;
            basis.m = 0;
            basis.n = 0;

            basis.primitives = *s_primitives;

            basis_functions.push_back(basis);
            basis_to_atom.push_back(atom_index);    

            // construct the three 2p orbitals
            for (int direction = 0; direction < 3; direction++) {

                BasisFunction p_basis;

                p_basis.atomic_number = atom.element;
                p_basis.x = atom.x;
                p_basis.y = atom.y;
                p_basis.z = atom.z;

                p_basis.l = (direction == 0) ? 1 : 0;
                p_basis.m = (direction == 1) ? 1 : 0;
                p_basis.n = (direction == 2) ? 1 : 0;

                p_basis.primitives = *p_primitives;

                basis_functions.push_back(p_basis);
                basis_to_atom.push_back(atom_index);
            }
        }
    }
  
    std::cout << "Constructed basis functions: " 
                << basis_functions.size() << std::endl;

    std::cout << "Basis-to-atom mapping:\n";

    for (size_t mu = 0; mu < basis_to_atom.size(); mu++) {
        std::cout << "Orbital " << mu
                  << " belongs to atom "
                  << basis_to_atom[mu] << '\n';

    }

    // Number of atomic orbitals 
    size_t n_basis = basis_functions.size();

    // Construct the overlap matrix
    arma::mat S(n_basis, n_basis, arma::fill::zeros);

    for (size_t mu = 0; mu < n_basis; mu++) {
        for (size_t nu = 0; nu < n_basis; nu++) {

            S(mu, nu) = contracted_overlap(
                basis_functions[mu],
                basis_functions[nu]
            );
        }
    }

    std::cout << "Overlap matrix S:\n ";
    S.print();

    // Number of atoms 
    size_t n_atoms = atoms.size();

    // Initialize the electron-electron repulsion matrix 
    arma::mat gamma(n_atoms, n_atoms, arma::fill::zeros);

    // Construct the gamma matrix for all atom pairs
    const double hartree_to_ev = 27.211324570273;

    for (size_t A = 0; A < atoms.size(); A++) {
        for (size_t B = A; B < atoms.size(); B++) {

            // Squared distance between atoms A and B 
            double dx = atoms[A].x - atoms[B].x;
            double dy = atoms[A].y - atoms[B].y;
            double dz = atoms[A].z - atoms[B].z;

            double R_AB_squared = dx * dx + dy * dy + dz * dz;

            // Select the normalized s_primitives 
            const auto& primitives_A = get_s_primitives(atoms[A].element);
            const auto& primitives_B = get_s_primitives(atoms[B].element);

            // Calculate the contracted electron-electron repulsion integral 
            double gamma_AB = contracted_repulsion(
                primitives_A,
                primitives_B,
                R_AB_squared
            );

            // Convert Hartree to eV
            gamma_AB *= hartree_to_ev;

            // Gamma matrix is symmetric 
            gamma(A, B) = gamma_AB;
            gamma(B, A) = gamma_AB;
        }
    }

    gamma.print("Gamma matrix (eV):");

    // Initialize the core Hamiltonian 
    arma::mat H_core(n_basis, n_basis, arma::fill::zeros);

    // Calculate diagonal core Hamiltonian elements 
    for (size_t mu = 0; mu < n_basis; mu++) {

        const BasisFunction& orbital = basis_functions[mu];

        // Identify the atom associated with orbital mu 
        size_t A = basis_to_atom[mu];

        // One-electron orbital energy contribution 
        double energy_term = -orbital_parameter(
            orbital.atomic_number,
            orbital.l,
            orbital.m,
            orbital.n
        );

        // Valence nuclear charge of atom A
        double Z_A = valence_electrons(atoms[A].element);

        // Same-atom nuclear contributions
        double same_atom = -(Z_A - 0.5) * gamma(A,A);

        // Contributions from other atoms 
        double other_atoms = 0.0;

        for (size_t C = 0; C < atoms.size(); C++) {
             if (C == A) {
                continue;
            }

            double Z_C = valence_electrons(atoms[C].element);

            other_atoms -= Z_C * gamma(A, C);
        }

        // store diagonal core Hamiltonian element 
        H_core(mu, mu) = energy_term + same_atom + other_atoms;
    }

    H_core.print("Core Hamiltonian diagonal (eV): ");

    // Calculate off-diagonal core Hamiltonian elements 
    for (size_t mu = 0; mu < n_basis; mu++) {
        for (size_t nu = mu + 1; nu < n_basis; nu++) {

            // Identify the atoms associated with each orbital
            size_t A = basis_to_atom[mu];
            size_t B = basis_to_atom[nu];

            // Resonance parameters
            double beta_A = resonance_parameter(atoms[A].element);
            double beta_B = resonance_parameter(atoms[B].element);

            // Off-diagonal core Hamiltonian contribution
            double resonance = 
            0.5 * (beta_A + beta_B) * S(mu, nu);

            // Fill both halves of the symmetric matrix
            H_core(mu, nu) = resonance;
            H_core(nu, mu) = resonance;
        }
    }

    H_core.print("Core Hamiltonian (eV): ");

    // Initialize alpha and beta density matrices 
    arma::mat P_alpha(n_basis, n_basis, arma::fill::zeros);
    arma::mat P_beta(n_basis, n_basis, arma::fill::zeros);

    // Calculate total electron density
    arma::mat P_total = P_alpha + P_beta;

    // Atomic electron populations
    arma::vec atomic_population(atoms.size(), arma::fill::zeros);

    for (size_t mu = 0; mu < n_basis; mu++) {

        // Identify the atom that owns orbital mu
        size_t A = basis_to_atom[mu];

        // Add the orbitals electron population
        atomic_population(A) += P_total(mu, mu);
    }

    atomic_population.print("Atomic electron populations:");
    
    // Initialize alpha and beta Fock matrices 
    arma::mat F_alpha(n_basis, n_basis, arma::fill::zeros);
    arma::mat F_beta(n_basis, n_basis, arma::fill::zeros);

    auto build_fock_matrices = [&]() {

        // Reset Fock matrices 
        F_alpha.zeros();
        F_beta.zeros();

        // Calculate total electron density
        arma::mat P_total = P_alpha + P_beta;

        // Calculate atomic electron populations
        arma::vec atomic_population(atoms.size(), arma::fill::zeros);

        for (size_t mu= 0; mu < n_basis; mu++) {
            size_t A = basis_to_atom[mu];
            atomic_population(A) += P_total(mu, mu);
        }

        // Calculate diagonal Fock elements
        for (size_t mu = 0; mu < n_basis; mu++) {

            const BasisFunction& orbital = basis_functions[mu];

            double energy_term = -orbital_parameter(
                orbital.atomic_number,
                orbital.l,
                orbital.m,
                orbital.n
            );

            F_alpha(mu, mu) = energy_term;
            F_beta(mu, mu) = energy_term;
    
            // Identify the parent atom of orbital mu
            size_t A = basis_to_atom[mu];

            // Valence nuclear charge for atom A
            double Z_A = valence_electrons(atoms[A].element);

            // Total electron population on atom A 
            double P_AA = atomic_population(A);

            // Same-atom contribution for alpha electrons 
            double same_atom_alpha = 
                ((P_AA - Z_A) - (P_alpha(mu, mu) - 0.5))
                 * gamma(A, A);

            // Same-atom contribution for beta electrons
            double same_atom_beta =
                ((P_AA - Z_A) - (P_beta(mu, mu) - 0.5))
                * gamma(A, A);

            // add these contributions to the diagonal
            F_alpha(mu, mu) += same_atom_alpha;
            F_beta(mu, mu) += same_atom_beta;

            // contributions from all other atoms
            double other_atom = 0.0;
        

            for (size_t B = 0; B < atoms.size(); B++) {
         
                // Exclude the atom that owns orbital mu 
                if ( B == A) {
                continue;
                }

                double Z_B = valence_electrons(atoms[B].element);
                double P_BB = atomic_population(B);

                other_atom += (P_BB - Z_B) * gamma(A, B);
            }

            F_alpha(mu, mu) += other_atom;
            F_beta(mu, mu) += other_atom;
        }

        // Calculate off-diagonal Fock matrix elements 
        for (size_t mu = 0; mu < n_basis; mu++) {
            for (size_t nu = mu + 1; nu < n_basis; nu++) {
            
                // Identify the atoms associated with each orbital
                size_t A = basis_to_atom[mu];
                size_t B = basis_to_atom[nu];

                // Resonance parameters for atoms A and B 
                double beta_A = resonance_parameter(atoms[A].element);
                double beta_B = resonance_parameter(atoms[B].element);

                // Resonance contribution 
                double resonance = 
                0.5 * (beta_A + beta_B) * S(mu,nu);

                // alpha and beta Fock contribution
                double f_alpha = 
                resonance - P_alpha(mu, nu) * gamma(A, B);

                double f_beta = 
                resonance - P_beta(mu, nu) * gamma(A, B);

                // Fill both halves of the symmetric matrices 
                F_alpha(mu, nu) = f_alpha;
                F_alpha(nu, mu) = f_alpha;

                F_beta(mu, nu) = f_beta;
                F_beta(nu, mu) = f_beta; 
            }
        }
    };
  
    // Save initial Fock matrices before SCF iterations 
    build_fock_matrices();

    arma::mat Fa_initial = F_alpha;
    arma::mat Fb_initial = F_beta;

    const int max_iterations = 100;
    const double tolerance = 1e-6;
    bool converged = false;
        
    for (int iteration = 0; iteration < max_iterations; iteration++) {
        // Construct Fock matrices from current densities
        build_fock_matrices(); 

        // Diagonalize the alpha and beta Fock matrix
        arma::vec epsilon_alpha, epsilon_beta;
        arma::mat C_alpha, C_beta;

        arma::eig_sym(epsilon_alpha, C_alpha, F_alpha);
        arma::eig_sym(epsilon_beta, C_beta, F_beta);

        // Display molecular orbital energies 
        epsilon_alpha.print("Alpha orbital energies (eV): ");
        epsilon_beta.print("Beta orbital energies (eV): ");

        // Construct new density matrices 
        arma::mat P_alpha_new(n_basis, n_basis, arma::fill::zeros);
        arma::mat P_beta_new(n_basis, n_basis, arma::fill::zeros);
    

        // Add contributions from occupied alpha molecular orbitals
        for (int i = 0; i < num_alpha_electrons; i++) {
            arma::vec occupied_orbital = C_alpha.col(i);
            P_alpha_new += occupied_orbital * occupied_orbital.t();
        }

        // Add contributions from occupied beta molecular orbitals
        for (int i =0; i < num_beta_electrons; i++) {
            arma::vec occupied_orbital = C_beta.col(i);
            P_beta_new += occupied_orbital * occupied_orbital.t();
        }

        // calculate maximum density change 
        double delta_alpha = arma::abs(P_alpha_new - P_alpha).max();
        double delta_beta = arma::abs(P_beta_new- P_beta).max();
        double delta_P = std::max(delta_alpha, delta_beta);

        std::cout << "SCF iteration " << iteration + 1
                  << ": density change = " << delta_P << std::endl;

        // Update densities
        P_alpha = P_alpha_new;
        P_beta = P_beta_new;

        // Check convergence
        if (delta_P < tolerance) {
            converged = true;
            std::cout << "SCF converged in: " 
                      << iteration + 1 << " iterations." << std::endl;
            break;
        }
    }

    if (!converged) {
        throw std::runtime_error("SCF failed to converge. ");
    }

    // Rebuild Fock matrices using updated densities 
    build_fock_matrices();

    // Solve the final converged eigenvalue problems
    arma::vec final_epsilon_alpha;
    arma::vec final_epsilon_beta;

    arma::mat final_C_alpha;
    arma::mat final_C_beta;

    arma::eig_sym(final_epsilon_alpha, final_C_alpha, F_alpha);
    arma::eig_sym(final_epsilon_beta, final_C_beta, F_beta);

    // Print final molecular orbital energies 
    final_epsilon_alpha.print("Final alpha orbital energies (eV): ");
    final_epsilon_beta.print("Final beta orbital energies (eV): ");

    // Print final molecular orbital coefficients 
    final_C_alpha.print("Final alpha eigenvectors: ");
    final_C_beta.print("Final beta eigenvectors: ");

    // Check final atomic electron populations 
    arma::mat P_total_final = P_alpha + P_beta;

    arma::vec final_atomic_population(
        atoms.size(), arma::fill::zeros
    );

    for (size_t mu = 0; mu < n_basis; mu++) {
        size_t A = basis_to_atom[mu];
        final_atomic_population(A) += P_total_final(mu, mu);
    }

    final_atomic_population.print("Final atomic electron populations: ");

    // Calculate CNDO/2 electronic energy
    double E_electronic = 0.0;

    for (size_t mu = 0; mu < n_basis; mu++) {
        for (size_t nu = 0; nu < n_basis; nu++) {

        double alpha_contribution = 
        P_alpha(mu, nu) * (H_core(mu, nu) + F_alpha(mu, nu));

        double beta_contribution = 
        P_beta(mu, nu) * (H_core(mu, nu) + F_beta(mu, nu));
      
        E_electronic += 
        0.5 * (alpha_contribution + beta_contribution);
        }
    }

    // Calculate nuclear repulsion energy 
    double E_nuclear = 0.0;

    for (size_t A = 0; A < atoms.size(); A++) {
        for (size_t B = 0; B < A; B++) {

        double dx = atoms[A].x - atoms[B].x;
        double dy = atoms[A].y - atoms[B].y;
        double dz = atoms[A].z - atoms[B].z;

        double R_AB = std::sqrt(dx*dx + dy*dy + dz*dz);

        if (R_AB < 1e-12) {
        throw std::runtime_error("Two nuclei occupy the same position.");
        }

        double Z_A = valence_electrons(atoms[A].element);
        double Z_B = valence_electrons(atoms[B].element);

        E_nuclear += (Z_A * Z_B / R_AB) * hartree_to_ev;
        }
    }

    double E_total = E_electronic + E_nuclear;

    std::cout << "Electronic energy (eV): "
              << E_electronic << std::endl;

    std::cout << "Nuclear repulsion energy (eV): "
              << E_nuclear << std::endl;

    std::cout << "Total CNDO/2 energy (eV): "
              << E_total << std::endl;

    // Store converged SCF results 
    SCFResult result;

    result.P_alpha = P_alpha;
    result.P_beta = P_beta;
    result.S = S;
    result.gamma = gamma;
    result.H_core = H_core;
    result.orbital_energies_alpha = final_epsilon_alpha;
    result.orbital_energies_beta = final_epsilon_beta;

    result.basis_functions = basis_functions;
    result.basis_to_atom = basis_to_atom;

    result.E_electronic = E_electronic;
    result.E_nuclear = E_nuclear;
    result.E_total = E_total;

    return result;
}
    