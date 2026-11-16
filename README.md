# CNDO/2 Molecular Properties Package 

A reusable C++ computational chemistry package for calculating molecular properties using the **Complete Neglect of Differential Overlap (CNDO/2)** semiempirical quantum chemistry method and benchmarking results against **Psi4 Density Functional Theory (DFT)** calculations.

## Project Overview

This project extends a CNDO/2 Self-Consistent field (SCF) implementation to calculate molecular properties from converged electronic structure calculations. 

The package uses C++ and the Armadillo linear algebra library to perform matrix operations, solve the SCF equations, and extract molecular properties.

The primary objectives are to:

- Develop resuable functions for calculating molecular properties from CNDO/2 results.
- Validate the calculations using unit and integration tests.
- Compare CNDO/2 predictions with reference DFT calculations performed using Psi4.
- Evaluate the accuracy and limitations of the CNDO/2 approximations.

The package currently includes calculations for atomic partial charges, charge-based molecular dipole moments, and spin-resolved HOMO-LUMO orbital-energy gaps.

## Features and Molecular Properties 

The package calculates molecular properties using the converged CNDO/2 Self-Consistent Field (SCF) results.

### Implementation Properties 

| Property | Description | Units |
| --- | --- | --- |
| Atomic Partial Charges | Calculates atomic charges from CDNO/2 electron populations. | Elementary charge (e) |
| Molecular Dipole Moment | Calculates the molecular dipole vector from atomic partial charges and atomic coordinates. | Debye (D) |
| HOMO-LUMO Energy Gap | Calculates the energy difference between the lowest unoccupied and highest occupied molecular orbitals for each spin channel. | Electrovolts (eV) |

### Property Equations 

**1. Atomic Partial Charges**

The partial charge on atom A is calculated as:

$$
q_A = Z_A^{\mathrm{valence}} - \sum_{\mu \in A} P_{\mu\mu}
$$

where: 
- $Z_A^{\mathrm{valence}}$ is the number of valence electrons associated with atom A.
- $P_{\mu\mu}$ is the diagonal element of the total electron density matrix. 

**2. Molecular Dipole Moment**

The charge-based molecular dipole moment is calculated as:

$$ 
\boldsymbol{\mu} = \sum_A q_A \mathbf{R}_A
$$

where:
- $q_A$ is the atomic partial charge.
- $\mathbf{R}_A$ is the position vector of atom A.

The result is converted from electron-bohr to Debye using:

$$
1\ e\,a_0 = 2.541746\ \mathrm{D}
$$

**3. HOMO-LUMO Gap**

The orbital-energy gap is calculated as:

$$
\Delta E = \varepsilon_{\mathrm{LUMO}} - 
\varepsilon_{\mathrm{HOMO}}
$$

where:
- $\varepsilon_{\mathrm{HOMO}}$ is the highest occupied molecular orbital energy.
- $\varepsilon_{\mathrm{LUMO}}$ is the lowest unoccupied molecular orbital energy.

Alpha and beta spin channels can be evaluated separately. 

### Building the Project 

The project is implemented in C++ and uses CMAKE for build configuration and Armadillo for linear algebra operations.

### Development Environment 

Development and testing are performed in a Docker container to provide a consistent environment with the required dependencies.

### Build instructions 

From the project root directory, configure the project:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug
```

Compile the source code:

```bash
cmake --build build -j4
```

### Running Tests 

The project uses CTest for C++ unit and integration tests.

To execute all registered C++ tests:

```bash
ctest --test-dir build --output-on-failure
```

The current test suite includes:

| Test | Purpose |
| --- | --- |
| AtomicCharges | Validates atomic charge calculations |
| SCFAtomicCharges | Tests atomic charges from SCF results |
| DipoleMoment | Validates charge-based dipole calculations |
| SCFDipole | Tests dipole moments from SCF results |
| HOMOLUMOGap | Validates orbital-energy gap calculations |
| SCFHOMOLUMOGap | Tests HOMO-LUMO gaps from SCF results |

Additional regression tests verify the CNDO/2 overlap, gamma, and gradient calculations.

Run the python regression tests using:

```bash
sh test.sh
```

At the current development stage, all 6 C++ tests and 20 python regression tests pass. 


## Project Structure 

The project separates the CNDO/2 electronic structure calculations, molecular property calculations, and automated tests.

### Core Source Files 

| File | Purpose |
| --- | --- |
| `src/cndo2_helpers.hpp` | Declares CNDO/2 data structures and SCF functions. |
| `scr/cndo2_helpers.cpp` | Implements the CNDO/2 SCF solver and supporting calculations. |
| `scr/molecular_properties.hpp` | Declares reusable molecular property functions. |
| `scr/molecular_proprties.cpp` | Implements atomic charges, dipole moments, and HOMO-LUMO energy gaps. |
| `src/hw_5.2.cpp` | Contains the original CNDO/2 calculation executable. |

### Supporting Drectories 

| Directory | Purpose |
| --- | --- |
| `basis/` | STO-3G basis-set parameter files. |
| `tests/` | C++ unit and integration tests, along with regression tests. |
| `atoms/` | Molecular geometry input files. |
| `sample_input/` | Example calculation configurations. |

### Code Organization 

The CNDO/2 solver returns an `SCFResult` structure containing density matrices, molecular orbital energies, and other electronic structure information. 

The molecular properties module uses these results to calculate properties without repeating the SCF procedure.

This separation allows additional molecular properties to be implemented independently while reusing the same electronic structure calculations. 


