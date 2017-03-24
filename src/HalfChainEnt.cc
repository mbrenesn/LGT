#include "HalfChainEnt.h"

HalfChainEnt::HalfChainEnt(const Environment &env, const Basis &bas)
{
  mpirank_ = env.mpirank;
  mpisize_ = env.mpisize;
  basis_size_ = bas.basis_size;
  l_ = env.l;
  n_ = env.n;
  nlocal_ = bas.nlocal;
  start_ = bas.start;
  end_ = bas.end;

  red_size_ = 1 << (l_ / 2);
  env.distribution(red_size_, red_local_, red_start_, red_end_);
}

HalfChainEnt::~HalfChainEnt()
{}

void HalfChainEnt::free_vec_(std::vector<LLInt> &vec)
{
  std::vector<LLInt> tmp;
  vec.swap( tmp );
}

inline
LLInt HalfChainEnt::binary_to_int_(boost::dynamic_bitset<> &bs, unsigned int &size)
{
  LLInt integer = 0;
  for(unsigned int i = 0; i < size; ++i){
    if(bs[i] == 1){
      integer += 1ULL << i;
    }
  }

  return integer;
}

void HalfChainEnt::construct_connected_basis_(LLInt *int_basis, LLInt *a_basis, LLInt *b_basis)
{
  unsigned int l_2 = l_ / 2;
  LLInt *local_a_basis = new LLInt[nlocal_];
  LLInt *local_b_basis = new LLInt[nlocal_];
  for(PetscInt state = start_; state < end_; ++state){

    boost::dynamic_bitset<> bs(l_, int_basis[state - start_]);
    boost::dynamic_bitset<> bitset_a = bs >> (l_2);
    boost::dynamic_bitset<> bitset_b = bs;
    bitset_b.resize(l_2);

    local_a_basis[state - start_] = binary_to_int_(bitset_a, l_2);
    local_b_basis[state - start_] = binary_to_int_(bitset_b, l_2);
  }

  int *recvcounts = new int[mpisize_];
  int *displs = new int[mpisize_];
  MPI_Allgather(&nlocal_, 1, MPI_INT, recvcounts, 1, MPI_INT, PETSC_COMM_WORLD);
  MPI_Allgather(&start_, 1, MPI_INT, displs, 1, MPI_INT, PETSC_COMM_WORLD);

  MPI_Allgatherv(local_a_basis, nlocal_, MPI_LONG_LONG, a_basis, recvcounts, displs, 
    MPI_LONG_LONG, PETSC_COMM_WORLD);
  MPI_Allgatherv(local_b_basis, nlocal_, MPI_LONG_LONG, b_basis, recvcounts, displs,
    MPI_LONG_LONG, PETSC_COMM_WORLD);

  delete [] local_a_basis;
  delete [] local_b_basis;
  delete [] recvcounts;
  delete [] displs;
}

void HalfChainEnt::construct_redmat_indices_(LLInt *int_basis, std::vector<LLInt> &i_ind,
  std::vector<LLInt> &j_ind, std::vector<LLInt> &first_coef, std::vector<LLInt> &second_coef)
{
  LLInt *a_basis = new LLInt[basis_size_];
  LLInt *b_basis = new LLInt[basis_size_];
  construct_connected_basis_(int_basis, a_basis, b_basis);

  LLInt row_ind, col_ind;
  for(PetscInt state = start_; state < end_; ++state){
  
    LLInt connected = b_basis[state];
    for(LLInt i = 0; i < basis_size_; ++i){
      if(connected != b_basis[i]) continue;
      else{
        row_ind = a_basis[state];
        col_ind = a_basis[i];
        i_ind.push_back(row_ind);
        j_ind.push_back(col_ind);
        first_coef.push_back(state);
        second_coef.push_back(i);
      }
    }
  }
 
  //std::cout << "Proc " << mpirank_ << " red local " << red_local_ << std::endl;
  //for(unsigned int i = 0; i < i_ind.size(); ++i){
  //  std::cout << "I ind " << i_ind[i] << " J ind " << j_ind[i] << std::endl;
  //}

  delete [] a_basis;
  delete [] b_basis;
}

// TODO This routine is not working properly. Overallocating memory at the moment,
// not that big of a deal to be honest... (actual allocation is (1 << l_/2) * 2l_
// at the moment)
void HalfChainEnt::determine_redmat_allocation_details_(std::vector<LLInt> &i_ind, 
  std::vector<LLInt> &j_ind, PetscInt *diag, PetscInt *off)
{
  for(LLInt i = 0; i < red_local_; ++i) diag[i] = 1;

  ULLInt size = i_ind.size();
  LLInt row_ind, col_ind;
  for(ULLInt i = 0; i < size; ++i){
    row_ind = i_ind[i]; col_ind = j_ind[i];
    if(row_ind != col_ind){
      if(row_ind < red_end_ && row_ind >= red_start_){
        if(col_ind < red_end_ && col_ind >= red_start_) diag[row_ind - red_start_]++;
        else off[row_ind - red_start_]++;
      }
    }
  }

  // In case visualization of the allocation buffers is required
  /* 
  std::cout << "Diag" << "\t" << "Proc" << mpirank_ << std::endl;
  for(int j = 0; j < red_local_; ++j){
    std::cout << diag[j] << std::endl;
  }
  std::cout << "Off" << "\t" << "Proc" << mpirank_ << std::endl;
  for(int j = 0; j < red_local_; ++j){
    std::cout << off[j] << std::endl;
  }
  */
}

void HalfChainEnt::construct_redmat_(Mat &red_mat, Vec &v_dist, Vec &v_local, 
  std::vector<LLInt> &i_ind, std::vector<LLInt> &j_ind, std::vector<LLInt> &first_coef, 
    std::vector<LLInt> &second_coef, VecScatter &ctx)
{
  MatZeroEntries(red_mat);
  PetscScalar *cij;

  VecScatterCreateToAll(v_dist, &ctx, &v_local);
  VecScatterBegin(ctx, v_dist, v_local, INSERT_VALUES, SCATTER_FORWARD);
  VecScatterEnd(ctx, v_dist, v_local, INSERT_VALUES, SCATTER_FORWARD);

  VecGetArray(v_local, &cij);

  ULLInt index_size = i_ind.size();
  for(ULLInt index = 0; index < index_size; ++index){
    LLInt row_ind = i_ind[index];
    LLInt col_ind = j_ind[index];
    LLInt vec_first = first_coef[index];
    LLInt vec_second = second_coef[index];
    PetscScalar ci = cij[vec_first];
    PetscScalar ci_dag = cij[vec_second];
    PetscScalar val = ci * PetscConjComplex(ci_dag);
    MatSetValue(red_mat, row_ind, col_ind, val, ADD_VALUES);
  }

  VecRestoreArray(v_local, &cij);
  MatSetOption(red_mat, MAT_HERMITIAN, PETSC_TRUE);
  MatAssemblyBegin(red_mat, MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(red_mat, MAT_FINAL_ASSEMBLY);
}

void HalfChainEnt::von_neumann_entropy(LLInt *int_basis, Vec &v, const Mat &ham_mat, 
  const unsigned int iterations, const double *times, const double tol, const int maxits)
{
  Mat red_den_mat;
  MatCreate(PETSC_COMM_WORLD, &red_den_mat);
  MatSetSizes(red_den_mat, red_local_, red_local_, red_size_, red_size_);
  MatSetType(red_den_mat, MATMPIAIJ);

  std::vector<LLInt> i_ind;
  i_ind.reserve(nlocal_);
  std::vector<LLInt> j_ind;
  j_ind.reserve(nlocal_);
  std::vector<LLInt> first_coef;
  first_coef.reserve(nlocal_);
  std::vector<LLInt> second_coef;
  second_coef.reserve(nlocal_);
 
  construct_redmat_indices_(int_basis, i_ind, j_ind, first_coef, second_coef);
 
  if(i_ind.size() != j_ind.size() || first_coef.size() != second_coef.size()){
    std::cerr << "Error with indices for reduced density matrix!" << std::endl;
    MPI_Abort(PETSC_COMM_WORLD, 1);
  }

  //PetscInt *d_nnz, *o_nnz;
  //PetscCalloc1(red_local_, &d_nnz);
  //PetscCalloc1(red_local_, &o_nnz);
  
  //determine_redmat_allocation_details_(i_ind, j_ind, d_nnz, o_nnz);

  MatMPIAIJSetPreallocation(red_den_mat, PETSC_DEFAULT, NULL, PETSC_DEFAULT, NULL);
  MatSetOption(red_den_mat, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE);

  //PetscFree(d_nnz);
  //PetscFree(o_nnz);
 
  MFN mfn;
  EPS eps;
  FN f;
  MFNCreate(PETSC_COMM_WORLD, &mfn);
  MFNSetOperator(mfn, ham_mat);
  MFNGetFN(mfn, &f);
  FNSetType(f, FNEXP);
  MFNSetTolerances(mfn, tol, maxits);
  
  MFNSetType(mfn, MFNEXPOKIT);
  MFNSetUp(mfn);
  
  EPSCreate(PETSC_COMM_WORLD, &eps);
  
  EPSSetProblemType(eps, EPS_HEP);
  PetscInt nev = red_size_;
  PetscInt mpd = red_size_;
  EPSSetDimensions(eps, nev, PETSC_DEFAULT, mpd);
  EPSSetTolerances(eps, tol, maxits);

  // Scatter routines for time evolved vector
  VecScatter ctx;
  Vec v_local;

  if(mpirank_ == 0){
    //std::cout << "Time" << "\t" << "S_vn" << std::endl;
    std::cout << times[0] << "\t" << "0.0" << std::endl;
  }

  PetscScalar eigr;
  PetscReal s_vn;
  for(unsigned int tt = 1; tt < (iterations + 1); ++tt){
    
    s_vn = 0.0;
    FNSetScale(f, (times[tt] - times[tt - 1]) * PETSC_i, 1.0);
    MFNSolve(mfn, v, v);
    //VecView(v, PETSC_VIEWER_STDOUT_WORLD);
  
    construct_redmat_(red_den_mat, v, v_local, i_ind, j_ind, first_coef, second_coef, ctx);
    EPSSetOperators(eps, red_den_mat, NULL);
 
    EPSSolve(eps);
    PetscInt nconv;
    EPSGetConverged(eps, &nconv);

    if(nconv != red_size_){
      std::cerr << "Didn't converge to all eigenvalues" << std::endl;
      MPI_Abort(PETSC_COMM_WORLD, 1);
    }

    for(PetscInt i = 0; i < nconv; ++i){
      EPSGetEigenvalue(eps, i, &eigr, NULL);
      PetscReal eigr_real = PetscRealPart(eigr);
      //std::cout << "Eig: " << eigr_real << std::endl;
      if(eigr_real > 1e-16) s_vn += -1.0 * eigr_real * std::log(eigr_real);
    }

    if(mpirank_ == 0){
      std::cout << times[tt] << "\t" << s_vn << std::endl;
    }
  }

  MFNDestroy(&mfn);
  EPSDestroy(&eps);
  VecScatterDestroy(&ctx);
  VecDestroy(&v_local);
  MatDestroy(&red_den_mat);
} 
