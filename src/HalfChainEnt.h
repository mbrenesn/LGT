#ifndef __HALFCHAINENT_H
#define __HALFCHAINENT_H

#include "Environment.h"
#include "Basis.h"
#include "SparseOp.h"

class HalfChainEnt
{
  private:
    PetscMPIInt mpirank_;
    PetscMPIInt mpisize_;
    LLInt basis_size_;
    LLInt red_size_;
    unsigned int l_, n_;
    PetscInt nlocal_;
    PetscInt start_;
    PetscInt end_;
    PetscInt red_local_;
    PetscInt red_start_;
    PetscInt red_end_;
    void free_vec_(std::vector<LLInt> &vec);
    LLInt binary_to_int_(boost::dynamic_bitset<> &bs, unsigned int &size);
    void construct_connected_basis_(LLInt *int_basis, LLInt *a_basis, LLInt *b_basis);
    void construct_redmat_indices_(LLInt *int_basis, std::vector<LLInt> &i_ind, 
      std::vector<LLInt> &j_ind, std::vector<LLInt> &first_coef, 
        std::vector<LLInt> &second_coef);
    void determine_redmat_allocation_details_(std::vector<LLInt> &i_ind, 
      std::vector<LLInt> &j_ind, PetscInt *diag, PetscInt *off);
    void construct_redmat_(Mat &red_mat, Vec &v_dist, Vec &v_local, std::vector<LLInt> &i_ind, 
      std::vector<LLInt> &j_ind, std::vector<LLInt> &first_coef, std::vector<LLInt> &second_coef,
        VecScatter &ctx);

  public:
    HalfChainEnt(const Environment &env, const Basis &bas);
    ~HalfChainEnt();
    void von_neumann_entropy(LLInt *int_basis, Vec &v, const Mat &ham_mat,  
      const unsigned int iterations, const double *times, const double tol, const int maxits);
};
#endif
