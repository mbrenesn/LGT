#include "SparseOp.h"

SparseOp::SparseOp(const Environment &env, const Basis &basis)
: magnetization(NULL), hmag(NULL), hmag_1(NULL)
{
  l_ = env.l;
  n_ = env.n;
  mpirank_ = env.mpirank;
  mpisize_ = env.mpisize;
  nlocal_ = basis.nlocal;
  start_ = basis.start;
  end_ = basis.end;
  basis_size_ = basis.basis_size;
}

/*******************************************************************************/
// Copy constructor
/*******************************************************************************/
SparseOp::SparseOp(const SparseOp &rhs)
{
  l_ = rhs.l_;
  n_ = rhs.n_;
  mpirank_ = rhs.mpirank_;
  mpisize_ = rhs.mpisize_;
  nlocal_ = rhs.nlocal_;
  start_ = rhs.start_;
  end_ = rhs.end_;
  basis_size_ = rhs.basis_size_;
  magnetization = NULL;
  hmag = NULL;
  hmag_1 = NULL;
}

/*******************************************************************************/
// Assignment operator
/*******************************************************************************/
SparseOp &SparseOp::operator=(const SparseOp &rhs)
{
  VecDestroy(&magnetization);
  VecDestroy(&hmag);
  VecDestroy(&hmag_1);
  l_ = rhs.l_;
  n_ = rhs.n_;
  mpirank_ = rhs.mpirank_;
  mpisize_ = rhs.mpisize_;
  nlocal_ = rhs.nlocal_;
  start_ = rhs.start_;
  end_ = rhs.end_;
  basis_size_ = rhs.basis_size_;
  magnetization = NULL;
  hmag = NULL;
  hmag_1 = NULL;

  return *this;
}

SparseOp::~SparseOp()
{
  VecDestroy(&magnetization); 
  VecDestroy(&hmag);
  VecDestroy(&hmag_1);
}

LLInt SparseOp::mod_(LLInt a, LLInt b)
{
  return (a % b + b) % b;
}

inline
LLInt SparseOp::binary_to_int(boost::dynamic_bitset<> bs)
{
  LLInt integer = 0;

  for(unsigned int i = 0; i < l_; ++i){
    if(bs[i] == 1){
      integer += 1ULL << i;
    }
  }

  return integer;
}

/*******************************************************************************/
// Binary search: Divide and conquer. For the construction of the Hamiltonian
// matrix instead of looking through all the elements of the int basis a
// binary search will perform better for large systems
/*******************************************************************************/
inline
LLInt SparseOp::binsearch(const LLInt *array, LLInt len, LLInt value)
{
  if(len == 0) return -1;
  LLInt mid = len / 2;

  if(array[mid] == value) 
    return mid;
  else if(array[mid] < value){
    LLInt result = binsearch(array + mid + 1, len - (mid + 1), value);
    if(result == -1) 
      return -1;
    else
      return result + mid + 1;
  }
  else
    return binsearch(array, mid, value);
}

/*******************************************************************************/
// Initial Neel state index 
/*******************************************************************************/
LLInt SparseOp::get_neel_index(const Basis &bas)
{
  LLInt index;
  if(l_ / 2 != n_){
    std::cerr << "Not implemented!" << std::endl;
    std::cerr << "Neel state has only been implemented for half-filled systems" << std::endl;
    MPI_Abort(PETSC_COMM_WORLD, 1);
  }

  boost::dynamic_bitset<> neel(l_, 1);
  for(unsigned int site = 0; site < l_; site += 2){
    neel.set(site);
  }

  LLInt neel_int = binary_to_int(neel);
  index = binsearch(bas.int_basis, bas.basis_local, neel_int);
  if(index != -1) index += start_;

  return index;
}

/*******************************************************************************/
// Pick initial index out of the basis randomly 
/*******************************************************************************/
LLInt SparseOp::get_random_index(const Basis &bas, bool wtime, bool verbose)
{
  LLInt pick_ind;
  boost::random::mt19937 gen;

  if(wtime) gen.seed(static_cast<LLInt>(std::time(0)));

  if(mpirank_ == 0){
    boost::random::uniform_int_distribution<LLInt> dist(0, bas.basis_size - 1);
    pick_ind = dist(gen);
  }
  MPI_Bcast(&pick_ind, 1, MPI_LONG_LONG_INT, 0, PETSC_COMM_WORLD);

  bool check = false;
  if(pick_ind >= start_ && pick_ind < end_) check = true;
    
  if(verbose && check){
    std::cout << "Initial state randomly chosen: " << bas.int_basis[pick_ind] << std::endl;
    std::cout << "With binary representation: " << std::endl;
    boost::dynamic_bitset<> bs(l_, bas.int_basis[pick_ind - start_]);
    std::cout << bs << std::endl;
  }

  return pick_ind;
}

/*******************************************************************************/
// Collects the nonlocal start index (global indices) of every processor, to be used
// during the construction of the matrix
/*******************************************************************************/
void SparseOp::gather_nonlocal_values_(LLInt *start_inds)
{
  MPI_Allgather(&start_, 1, MPI_LONG_LONG, start_inds, 1, MPI_LONG_LONG, 
    PETSC_COMM_WORLD);
}

/*******************************************************************************/
// Determines the sparsity pattern to allocate memory only for the non-zero 
// entries of the matrix
/*******************************************************************************/
void SparseOp::determine_allocation_details_(LLInt *int_basis, 
  std::vector<LLInt> &cont, std::vector<LLInt> &st, PetscInt *diag, PetscInt *off)
{
  for(PetscInt i = 0; i < nlocal_; ++i) diag[i] = 1;

  for(PetscInt state = start_; state < end_; ++state){

    boost::dynamic_bitset<> bs(l_, int_basis[state - start_]);

    // Loop over all sites of the bit representation
    for(unsigned int site = 0; site < (l_ - 1); ++site){
      // A copy to avoid modifying the original basis
      boost::dynamic_bitset<> bitset = bs;
      
      // Case 1: There's a particle in this site
      if(bitset[site] == 1){
        int next_site1 = (site + 1);

        // If there's a particle in next site, do nothing
        if(bitset[next_site1] == 1){
          continue;
        }
        // Otherwise do a swap
        else{
          bitset[next_site1] = 1;
          bitset[site]       = 0;

          LLInt new_int1 = binary_to_int(bitset);
          // Loop over all states and look for a match
          LLInt match_ind1 = binsearch(int_basis, nlocal_, new_int1); 
          if(match_ind1 == -1){
            cont.push_back(new_int1);
            st.push_back(state);
            continue;
          }
          else{
            match_ind1 += start_;
          }

          if(match_ind1 < end_ && match_ind1 >= start_) diag[state - start_]++;
          else off[state - start_]++; 
        }
      }
      // Case 2: There's no particle in this site
      else{
        int next_site0 = (site + 1);

        // If there's a particle in the next site, a swap can occur
        if(bitset[next_site0] == 1){
          bitset[next_site0] = 0;
          bitset[site]       = 1;

          LLInt new_int0 = binary_to_int(bitset);
          // Loop over all states and look for a match
          LLInt match_ind0 = binsearch(int_basis, nlocal_, new_int0); 
          if(match_ind0 == -1){
            cont.push_back(new_int0);
            st.push_back(state);
            continue;
          }
          else{
            match_ind0 += start_;
          }
          
          if(match_ind0 < end_ && match_ind0 >= start_) diag[state - start_]++;
          else off[state - start_]++;
        }
        // Otherwise do nothing
        else{
          continue;
        }
      }    
    }
  }

  // Collective communication of global indices
  LLInt *start_inds = new LLInt[mpisize_];

  gather_nonlocal_values_(start_inds);

  // Proc 0 is always gonna have the larger section of the distribution (when rest is present)
  // so let's use this value as the size of the basis_help buffer
  LLInt basis_help_size;
  if(mpirank_ == 0) basis_help_size = nlocal_;

  MPI_Bcast(&basis_help_size, 1, MPI_LONG_LONG, 0, PETSC_COMM_WORLD);

  // Create basis_help buffers and initialize them to zero
  LLInt *basis_help = new LLInt[basis_help_size];
  for(LLInt i = 0; i < basis_help_size; ++i) basis_help[i] = 0;

  // At the beginning basis_help is just int_basis, with the remaining values set to zero
  // It's important that the array remains sorted for the binary lookup
  for(LLInt i = 0; i < nlocal_; ++i)
    basis_help[i + (basis_help_size - nlocal_)] = int_basis[i];

  // Main communication procedure. A ring exchange of the int_basis using basis_help memory
  // buffer, after a ring exchange occurs each processor looks for the missing indices of
  // the Hamiltonian and replaces the values of cont buffer
  PetscMPIInt next = (mpirank_ + 1) % mpisize_;
  PetscMPIInt prec = (mpirank_ + mpisize_ - 1) % mpisize_;

  LLInt cont_size = cont.size();

  for(PetscMPIInt exc = 0; exc < mpisize_ - 1; ++exc){
 
    MPI_Sendrecv_replace(&basis_help[0], basis_help_size, MPI_LONG_LONG, next, 0,
      prec, 0, PETSC_COMM_WORLD, MPI_STATUS_IGNORE);

    PetscMPIInt source = mod_((prec - exc), mpisize_);
    for(LLInt i = 0; i < cont_size; ++i){
      if(cont[i] > 0){
        LLInt m_ind = binsearch(basis_help, basis_help_size, cont[i]);

        if(m_ind != -1){
          if(basis_help[0] == 0) m_ind = m_ind - 1;
          cont[i] = -1ULL * (m_ind + start_inds[source]);
        }
      }
    }
  }
  
  // Flip the signs
  for(PetscInt i = 0; i < cont_size; ++i) cont[i] = -1ULL * cont[i];
  
  //std::cout << "Cont from Proc " << mpirank_ << std::endl;
  //for(LLInt i = 0; i < cont_size; ++i) std::cout << cont[i] << std::endl;

  // Now cont contains the missing indices
  for(LLInt in = 0; in < cont_size; ++in){
    LLInt st_c = st[in];
    if(cont[in] < end_ && cont[in] >= start_) diag[st_c - start_]++;
    else off[st_c - start_]++;
  }

  delete [] basis_help;
  delete [] start_inds;  

  // In case visualization of the allocation buffers is required
  /*    
  std::cout << "Diag" << "\t" << "Proc" << mpirank_ << std::endl;
  for(int j = 0; j < m; ++j){
    std::cout << diag[j] << std::endl;
  }
  std::cout << "Off" << "\t" << "Proc" << mpirank_ << std::endl;
  for(int j = 0; j < m; ++j){
    std::cout << off[j] << std::endl;
  }
  */ 

}

/*******************************************************************************/
// Computes the Hamiltonian matrix given by means of the integer basis
/*******************************************************************************/
void SparseOp::construct_schwinger_hamiltonian(Mat &ham_mat, LLInt *int_basis, 
    double V, double t, double h, bool magn, bool sites)
{
  // Preallocation. For this we need a hint on how many non-zero entries the matrix will
  // have in the diagonal submatrix and the offdiagonal submatrices for each process

  // Allocating memory only for the non-zero entries of the matrix
  PetscInt *d_nnz, *o_nnz;
  PetscCalloc1(nlocal_, &d_nnz);
  PetscCalloc1(nlocal_, &o_nnz);

  std::vector<LLInt> cont;
  cont.reserve(basis_size_ / l_);
  std::vector<LLInt> st;
  st.reserve(basis_size_ / l_);
 
  determine_allocation_details_(int_basis, cont, st, d_nnz, o_nnz);

  // Preallocation step
  MatMPIAIJSetPreallocation(ham_mat, 0, d_nnz, 0, o_nnz);

  PetscFree(d_nnz);
  PetscFree(o_nnz);

  // Magnetization measurements
  if(magn){
    VecCreate(PETSC_COMM_WORLD, &magnetization);
    VecSetSizes(magnetization, nlocal_, basis_size_);
    VecSetType(magnetization, VECMPI);
  }
  if(sites){
    VecCreate(PETSC_COMM_WORLD, &hmag);
    VecSetSizes(hmag, nlocal_, basis_size_);
    VecSetType(hmag, VECMPI);
    VecDuplicate(hmag, &hmag_1);
  }

  // Hamiltonian matrix construction
  PetscScalar ti = t;

  // Cnl matrix
  std::vector<unsigned int> Cnl(l_ * l_);
  for(unsigned int i = 0; i < l_; ++i){
    Cnl[i * l_ + i] = 0;
    for(unsigned int j = 0; j < l_; ++j){
      if(j > i) Cnl[i * l_ + j] = l_ - j - 1;
      else if(j < i) Cnl[i * l_ + j] = l_ - i - 1;
    }
  }

  // On-site term prefactor
  std::vector<unsigned int> pre_fact(l_ - 1);
  pre_fact[0] = l_ / 2;
  for(unsigned int i = 1; i < (l_ - 1); i += 2){
    pre_fact[i] = pre_fact[i - 1] - 1;
    pre_fact[i + 1] = pre_fact[i - 1] - 1;
  }

  //std::cout << "Pre_fact" << std::endl;
  //for(unsigned int i = 0; i < (l_ - 1); ++i){
  //  std::cout << pre_fact[i] << std::endl;
  //}
 
  std::vector<int> rand_seq(l_ - 1);
  if(mpirank_ == 0){
    boost::random::mt19937 gen;
    gen.seed(static_cast<LLInt>(std::time(0)));
    boost::random::uniform_int_distribution<LLInt> dist(-1, 1);
    for(unsigned int i = 0; i < (l_ - 1); ++i)
      rand_seq[i] = dist(gen);
  }
  MPI_Bcast(&rand_seq[0], (l_ - 1), MPI_INT, 0, PETSC_COMM_WORLD);

  //std::cout << "rank " << mpirank_ << std::endl; 
  //for(unsigned int i = 0; i < (l_ - 1); ++i)
  //  std::cout << rand_seq[i] << " ";
  //std::cout << std::endl;

  // Grab 1 of the states and turn it into bit representation
  for(PetscInt state = start_; state < end_; ++state){
    
    boost::dynamic_bitset<> bs(l_, int_basis[state - start_]);

    std::vector<double> spins(l_);
    for(unsigned int i = 0; i < l_; ++i)
      bs[i] == 1 ? spins[i] = 1.0 : spins[i] = -1.0; 

    // On-site term and magnetization matrix (vector) if desired
    double os_term = 0.0; double os_term2 = 0.0; double mag_term = 0.0;
    for(unsigned int site = 0; site < l_; ++site){
      os_term += h * std::pow(-1, site + 1) * spins[site];
      if(magn) mag_term += std::pow(-1, site + 1) * spins[site] + 1;
    }

    if(magn){
      mag_term = mag_term / (2 * l_);
      VecSetValue(magnetization, state, mag_term, INSERT_VALUES);
    }

    if(sites){
      double site_half_mag = 0.5 * (spins[(l_ / 2) - 1] + 1);
      double site_half_mag_1 = 0.5 * (spins[(l_ / 2)] + 1);
      VecSetValue(hmag, state, site_half_mag, INSERT_VALUES);
      VecSetValue(hmag_1, state, site_half_mag_1, INSERT_VALUES);
    }

    // Loop over all sites of the bit representation
    double int_term = 0.0;
    for(unsigned int site = 0; site < (l_ - 1); ++site){
      // A copy to avoid modifying the original basis
      boost::dynamic_bitset<> bitset = bs;
      
      // Interacting term and Hz2
      os_term2 += V * pre_fact[site] * spins[site]; // Schwinger
      //double sigma = 0.0; double q_i = 0.0;
      //for(unsigned int i = 0; i < (site + 1); ++i){
      //  sigma += spins[i];
      //  q_i += rand_seq[i]; 
      //}
      //os_term2 += V * sigma * (q_i - ( (site + 1) % 2 ));

      for(unsigned int j = 0; j < (l_ - 1); ++j){
        int_term += V * spins[site] * spins[j] * Cnl[site * l_ + j];
      }

      // Case 1: There's a particle in this site
      if(bitset[site] == 1){
        
        int next_site1 = (site + 1);

        // If there's a particle in next site, do nothing
        if(bitset[next_site1] == 1){
          continue;
        }
        // Otherwise do a swap
        else{
          bitset[next_site1] = 1;
          bitset[site]       = 0;

          LLInt new_int1 = binary_to_int(bitset);
          // Loop over all states and look for a match
          LLInt match_ind1 = binsearch(int_basis, nlocal_, new_int1); 
          if(match_ind1 == -1){
            continue;
          }
          else{
            match_ind1 += start_;
          }
          
          if(match_ind1 == -1){
            std::cerr << "Error in the bin search within the Ham mat construction" << std::endl;
            MPI_Abort(PETSC_COMM_WORLD, 1);
          } 

          MatSetValues(ham_mat, 1, &match_ind1, 1, &state, &ti, ADD_VALUES);
        }
      }      
      // Case 2: There's no particle in this site
      else{
        int next_site0 = (site + 1) % l_;

        // If there's a particle in the next site, a swap can occur
        if(bitset[next_site0] == 1){
          bitset[next_site0] = 0;
          bitset[site]       = 1;

          LLInt new_int0 = binary_to_int(bitset);
          // Loop over all states and look for a match
          LLInt match_ind0 = binsearch(int_basis, nlocal_, new_int0); 
          if(match_ind0 == -1){
            continue;
          }
          else{
            match_ind0 += start_;
          }
          
          if(match_ind0 == -1){
            std::cerr << "Error in the bin search within the Ham mat construction" << std::endl;
            MPI_Abort(PETSC_COMM_WORLD, 1);
          } 
        
          MatSetValues(ham_mat, 1, &match_ind0, 1, &state, &ti, ADD_VALUES);
        }
        // Otherwise do nothing
        else{
          continue;
        }
      }    
    }

    // Important! Cnl is counting twice!
    PetscScalar diag_term = os_term - os_term2 + (0.5 * int_term); // Schwinger
    //PetscScalar diag_term = os_term + os_term2 + (0.5 * int_term);
    MatSetValues(ham_mat, 1, &state, 1, &state, &diag_term, ADD_VALUES);

    spins.erase(spins.begin(), spins.begin() + l_);
  }

  // Now cont contains the missing indices
  for(ULLInt in = 0; in < cont.size(); ++in){
    LLInt st_c = st[in];
    LLInt cont_c = cont[in];
    MatSetValues(ham_mat, 1, &cont_c, 1, &st_c, &ti, ADD_VALUES);
  }

  if(magn){
    VecAssemblyBegin(magnetization);
    VecAssemblyEnd(magnetization);
  }
  if(sites){
    VecAssemblyBegin(hmag);
    VecAssemblyBegin(hmag_1);
    VecAssemblyEnd(hmag);
    VecAssemblyEnd(hmag_1);
  }

  MatAssemblyBegin(ham_mat, MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(ham_mat, MAT_FINAL_ASSEMBLY);

  MatSetOption(ham_mat, MAT_SYMMETRIC, PETSC_TRUE);
}
