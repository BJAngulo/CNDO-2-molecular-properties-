#include <cstdlib>
#include <filesystem>
#include <format>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <vector>

#include <armadillo>
#include <nlohmann/json.hpp> 
#include "cndo2_helpers.hpp"
#include "gaussian_helpers.hpp"

namespace fs = std::filesystem;
using json = nlohmann::json;

int main(int argc, char **argv) {
  // check that a config file is supplied
  if (argc != 2) {
    std::cerr << "Usage: " << argv[0] << " path/to/config.json" << std::endl;
    return EXIT_FAILURE;
  }

  // parse the config file
  fs::path config_file_path(argv[1]);
  if (!fs::exists(config_file_path)) {
    std::cerr << "Path: " << config_file_path << " does not exist" << std::endl;
    return EXIT_FAILURE;
  }
  std::ifstream config_file(config_file_path);
  json config = json::parse(config_file);

  // extract the important info from the config file
  fs::path atoms_file_path = config["atoms_file_path"];
  fs::path output_file_path = config["output_file_path"];
  int num_alpha_electrons = config["num_alpha_electrons"];
  int num_beta_electrons = config["num_beta_electrons"];

  // Read XYZ file
  std::ifstream atoms_file(atoms_file_path);

  if(!atoms_file) {
    std::cerr << "Could not open atoms file: "
              << atoms_file_path << std::endl;
    return EXIT_FAILURE;
  }
  
  // Read the number of atoms
  int num_atoms;
  
  if (!(atoms_file >> num_atoms) || num_atoms <= 0) {
    throw std::runtime_error("Invalid number of atoms in XYZ file");
  }

  std::string line;
  std::getline(atoms_file, line);

  // Skip comment line 
  std::getline(atoms_file, line);

  // Store atomic numbers and coordinates 
  std::vector<Atom> atoms;

  for (int i = 0; i < num_atoms; i++) {
      Atom atom;
      
      if (!(atoms_file >> atom.element
                 >> atom.x
                 >> atom.y
                 >> atom.z)) {
          throw std::runtime_error("Invalid XYZ atom entry");
        }

      atoms.push_back(atom);
  }

  // Verify the data 
  std::cout << "Number of atoms: " << atoms.size() << '\n';
  std::cout << "Alpha electrons: " << num_alpha_electrons << '\n';
  std::cout << "Beta electrons: " << num_beta_electrons << '\n';

  // Run the CNDO/2 SCF calculation 
  SCFResult scf = run_scf(
    atoms,
    num_alpha_electrons,
    num_beta_electrons
  );

  // Determine the number of basis functions 
  int num_basis_functions = 
      static_cast<int>(scf.basis_functions.size());

  int num_3D_dims = 3;

  // Calculate the total electron density matrix
  arma::mat P_total = scf.P_alpha + scf.P_beta;

  // Initialize the overlap coefficient matrix 
  arma::mat x_coeff(
    num_basis_functions,
    num_basis_functions,
    arma::fill::zeros
  );

  // Loop over all pairs of basis functions
  for (int u = 0; u < num_basis_functions; u++) {
    for (int v = 0; v < num_basis_functions; v++) {

      // Find the atom associated with each basis function 
      size_t A = scf.basis_to_atom[u];
      size_t B = scf.basis_to_atom[v];

      // Retrieve the resonance parameters for atoms A and B
      double beta_A = resonance_parameter(atoms[A].element);
      double beta_B = resonance_parameter(atoms[B].element);

      // Calculate the overlap coefficient 
      x_coeff(u, v) = (beta_A + beta_B) * P_total(u, v);
    }
  }

  std::cout << "P_total:\n" << P_total << std::endl;
  std::cout << "x_coeff:\n" << x_coeff << std::endl;

  // Total electron population on each atom 
  arma::vec atomic_population(num_atoms, arma::fill::zeros);

  // Sum diagonal density matrix elements for each atom 
  for (int u = 0; u < num_basis_functions; u++) {

    // Identify the atom containing basis function u 
    size_t A = scf.basis_to_atom[u];

    // Add the electron population from this basis function 
    atomic_population(A) += P_total(u, u);
  }

  std::cout << "Atomic populations:\n" << atomic_population << std::endl;
  
  // Initialize gamma coefficient matrix 
  arma::mat y_coeff(
    num_atoms,
    num_atoms,
    arma::fill::zeros
  );

  for (size_t A = 0; A < num_atoms; A++) {
    for (size_t B = A + 1; B < num_atoms; B++) {

      double P_AA = atomic_population(A);
      double P_BB = atomic_population(B);
      
      double population_product = P_AA * P_BB;

      // retrieve the valence core changes 
      double Z_A = valence_electrons(atoms[A].element);
      double Z_B = valence_electrons(atoms[B].element);

      // Calculate the electron-core interaction terms 
      double electron_core = 
        -Z_B * P_AA - Z_A * P_BB;

      // Calculate the exchange contribution 
      double exchange_sum = 0.0;

      for (int u = 0; u < num_basis_functions; u++) {
        if (scf.basis_to_atom[u] != A) {
          continue;
        }

        for ( int v = 0; v < num_basis_functions; v++) {
          if (scf.basis_to_atom[v] != B) {
            continue;
          }

          double P_alpha_uv = scf.P_alpha(u, v);
          double P_beta_uv = scf.P_beta(u, v);

          exchange_sum += 
            P_alpha_uv * P_alpha_uv +
            P_beta_uv * P_beta_uv;
        }
      }

      // Combine all contributions to y_AB 
      double y_AB = population_product + electron_core - exchange_sum;

      // Store the symmetric coefficients
      y_coeff(A, B) = y_AB;
      y_coeff(B, A) = y_AB;
      
    }
  }

  std::cout << "y_coeff:\n" << y_coeff << std::endl;

  // Your answers go in these objects
  // Information about the convention of the requirements 
  std::cout << "Order of columns for Suv_RA is as follows: (u,v)" << std::endl;
  for (int u = 0; u < num_basis_functions; u++) {
    for (int v = 0; v < num_basis_functions; v++) {
      std::cout << std::format("({},{}) ", u, v);
    }
  }
  std::cout << std::endl;

  std::cout << "Order of columns for gammaAB_RA is as follows: (A,B)"
            << std::endl;
  for (int A = 0; A < num_atoms; A++) {
    for (int B = 0; B < num_atoms; B++) {
      std::cout << std::format("({},{}) ", A, B);
    }
  }
  std::cout << std::endl;

  std::cout << "Order of rows is as follows" << std::endl;
  std::cout << "x" << std::endl;
  std::cout << "y" << std::endl;
  std::cout << "z" << std::endl;

  arma::mat Suv_RA(num_3D_dims, num_basis_functions * num_basis_functions, arma::fill::zeros);
  // Ideally, this would be (3, n_funcs, n_funcs) rank-3 tensor
  // but we're flattening (n-funcs, n-atoms) into a single dimension (n-funcs ^
  // 2) this is because tensors are not supported in Eigen and I want students
  // to be able to submit their work in a consistent format

  size_t n_basis = scf.basis_functions.size();

  // Calculate overlap derivatives for unique orbital pairs 
  for (size_t u = 0; u < n_basis; ++u) {
    for (size_t v = u + 1; v < n_basis; ++v) {

      // Skip orbitals on the same atom 
      if (scf.basis_to_atom[u] == scf.basis_to_atom[v]) {
        continue;
      }

      arma::vec dS = contracted_overlap_derivative(
        scf.basis_functions[u],
        scf.basis_functions[v]
      );

      // Derivative for (u,v)
      size_t column_uv = u * n_basis + v;
      Suv_RA.col(column_uv) = dS;

      // Store reverse derivative for (v,u)
      size_t column_vu = v * n_basis + u;
      Suv_RA.col(column_vu) = -dS;

    }
  }

  arma::mat gammaAB_RA(num_3D_dims, num_atoms * num_atoms, arma::fill::zeros);
  // This is the same story, ideally, this would be (3, num_atoms, num_atoms)
  // instead of (3, num_atoms ^ 2)

  std::vector<size_t> atom_to_basis(num_atoms, scf.basis_functions.size());

  for (size_t u = 0; u < scf.basis_functions.size(); u++) {
    size_t A = scf.basis_to_atom[u];

    if (atom_to_basis[A] == scf.basis_functions.size()) {
      atom_to_basis[A] = u;
    }
  }

  for (size_t A = 0; A < num_atoms; A++) {
    for (size_t B = A + 1; B < num_atoms; B++) {

      // Calculate the gamma derivative for each unique atom pair 
      // Position of atom A 
      arma::vec R_A = {
        atoms[A].x,
        atoms[A].y,
        atoms[A].z
      };

      // Position of atom B
      arma::vec R_B = {
        atoms[B].x,
        atoms[B].y,
        atoms[B].z
      };

      // Find the s-type basis function for each atom 
      size_t basis_A = atom_to_basis[A];
      size_t basis_B = atom_to_basis[B];

      // Retrieve the primitive Gaussians 
      const auto& primitives_A = scf.basis_functions[basis_A].primitives;
      const auto& primitives_B = scf.basis_functions[basis_B].primitives;

      // Calculate the contracted gamma derivative
      arma::vec dGamma = contracted_repulsion_derivative(
        primitives_A,
        primitives_B,
        R_A,
        R_B
      );

      // Convert Hartree/bohr to eV/bohr
      dGamma *=  27.211324570273;

      // Column indices for the ordered atom pairs
      size_t column_AB = A * num_atoms + B;
      size_t column_BA = B * num_atoms + A;

      // Store derivatives using translation invariance
      gammaAB_RA.col(column_AB) = dGamma;
      gammaAB_RA.col(column_BA) = -dGamma;

    }
  }


  arma::mat gradient_nuclear(num_3D_dims, num_atoms, arma::fill::zeros);

  for (size_t A = 0; A < atoms.size(); A++) {
    for (size_t B = 0; B < atoms.size(); B++) {

      if (A == B) {
        continue; // skip self-interaction
      }

      double dx = atoms[A].x - atoms[B].x;
      double dy = atoms[A].y - atoms[B].y;
      double dz = atoms[A].z - atoms[B].z;

      double R_AB = std::sqrt(dx*dx + dy*dy + dz*dz);

      if (R_AB < 1e-12) {
        throw std::runtime_error("Two nuclei occupy the same position.");
      }

      double Z_A = valence_electrons(atoms[A].element);
      double Z_B = valence_electrons(atoms[B].element);

      double factor = -Z_A * Z_B * 27.211324570273 / (R_AB * R_AB * R_AB);

      gradient_nuclear(0, A) += factor * dx;
      gradient_nuclear(1, A) += factor * dy;
      gradient_nuclear(2, A) += factor * dz;
    }
  }

  arma::mat gradient_electronic(num_3D_dims, num_atoms, arma::fill::zeros);

  // Calculate the overlap contribution to the electronic gradient 
  for (int u = 0; u < num_basis_functions; u++) {
    for(int v = u + 1; v < num_basis_functions; v++) {
      
      // Identify the atoms containing these basis functions
      size_t A = scf.basis_to_atom[u];
      size_t B = scf.basis_to_atom[v];

      // Same center overlap derivatives are zero
      if (A == B) {
        continue;
      }

      // Retrieve the overlap derivative with respect to atom A 
      size_t column_uv = u * num_basis_functions + v;
      arma::vec dS = Suv_RA.col(column_uv);

      // Calculate the overlap contribution using the unique-pair coefficient 
      arma::vec contribution = x_coeff(u,v) * dS;

      // Accumulate contributions on the two atoms
      gradient_electronic.col(A) += contribution;
      gradient_electronic.col(B) -= contribution;
    }
  }

  std::cout << "Electronic gradient (overlap only):\n" << gradient_electronic << std::endl;
  
  // Calculate the gamma contribution to the electronic gradient 
  for (size_t A = 0; A < num_atoms; A++) {
    for (size_t B = A + 1; B < num_atoms; B++) {

      // Retrieve the gamma derivative for atom pair (A, B)
      size_t column_AB = A * num_atoms + B;
      arma::vec dGamma = gammaAB_RA.col(column_AB);

      // Calculate the gamma contribution using the unique-pair coefficient
      arma::vec contribution = y_coeff(A, B) * dGamma;

      // Accumulate contribution on both atoms
      gradient_electronic.col(A) += contribution;
      gradient_electronic.col(B) -= contribution;
    }
  }

  arma::mat gradient(num_3D_dims, num_atoms);

  // Combine electronic and nuclear contribution 
  gradient = gradient_electronic + gradient_nuclear;

  // Check translational variance
  arma::vec gradient_sum = arma::sum(gradient, 1);

  std::cout << "Sum of atomic gradients (x, y, z):\n"
            << gradient_sum << std::endl;


  // You do not need to modify the code below this point

  // Set print configs
  std::cout << std::fixed << std::setprecision(4) << std::setw(8) << std::right;

  // inspect your answer via printing
  Suv_RA.print("Suv_RA");
  gammaAB_RA.print("gammaAB_RA");
  gradient_nuclear.print("gradient_nuclear");
  gradient_electronic.print("gradient_electronic");
  gradient.print("gradient");

  // check that output dir exists
  if (!fs::exists(output_file_path.parent_path())) {
    fs::create_directories(output_file_path.parent_path());
  }

  // delete the file if it does exist (so that no old answers stay there by
  // accident)
  if (fs::exists(output_file_path)) {
    fs::remove(output_file_path);
  }

  // write results to file
  Suv_RA.save(
      arma::hdf5_name(output_file_path, "Suv_RA",
                      arma::hdf5_opts::append + arma::hdf5_opts::trans));
  gammaAB_RA.save(
      arma::hdf5_name(output_file_path, "gammaAB_RA",
                      arma::hdf5_opts::append + arma::hdf5_opts::trans));
  gradient_nuclear.save(
      arma::hdf5_name(output_file_path, "gradient_nuclear",
                      arma::hdf5_opts::append + arma::hdf5_opts::trans));
  gradient_electronic.save(
      arma::hdf5_name(output_file_path, "gradient_electronic",
                      arma::hdf5_opts::append + arma::hdf5_opts::trans));
  gradient.save(
      arma::hdf5_name(output_file_path, "gradient",
                      arma::hdf5_opts::append + arma::hdf5_opts::trans));
}