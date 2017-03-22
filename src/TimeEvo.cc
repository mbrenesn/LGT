#include "TimeEvo.h"

TimeEvo::TimeEvo(const Environment &env)
{
  mpirank_ = env.mpirank;
  mpisize_ = env.mpisize;
  l_ = env.l;
  n_ = env.n;

  MFNCreate(PETSC_COMM_WORLD, &mfn_);
}

TimeEvo::~TimeEvo()
{
  MFNDestroy(&mfn_);
}

void TimeEvo::loschmidt_echo(const SparseOp &sparse, const unsigned int iterations, 
  const double *times, const double tol, const int maxits, const Vec &v, const Mat &ham_mat, 
    bool magn, bool sites)
{
  MFNSetOperator(mfn_, ham_mat);
  MFNGetFN(mfn_, &f_);
  FNSetType(f_, FNEXP);
  MFNSetTolerances(mfn_, tol, maxits);

  MFNSetType(mfn_, MFNEXPOKIT);
  MFNSetUp(mfn_);

  MFNConvergedReason reason;

  PetscScalar l_echo, z_mag, mid_mag;
  PetscReal loschmidt;

  Vec vec_help, mag_help = NULL, h_help = NULL;
  VecDuplicate(v, &vec_help);
  VecCopy(v, vec_help);
  if(magn){
    if(sparse.magnetization == NULL){
      std::cerr << "Magnetization wasn't constructed!" << std::endl;
      MPI_Abort(PETSC_COMM_WORLD, 1);
    }
    VecDuplicate(v, &mag_help);

    VecPointwiseMult(mag_help, sparse.magnetization, v);
    VecDot(mag_help, v, &z_mag);
  }
  if(sites){
    if(sparse.hmag == NULL){
      std::cerr << "Mid site magnetization wasn't constructed!" << std::endl;
      MPI_Abort(PETSC_COMM_WORLD, 1);
    }

    VecDuplicate(v, &h_help);

    VecAXPY(sparse.hmag, -1.0, sparse.hmag_1);

    VecPointwiseMult(h_help, sparse.hmag, v);
    VecDot(h_help, v, &mid_mag);
  }

  if(mpirank_ == 0){
    if(!magn && !sites){
      //std::cout << "Time" << "\t" << "LE" << std::endl;
      std::cout << times[0] << "\t" << "0.0" << std::endl;
    }
    else if(magn && !sites){
      //std::cout << "Time" << "\t" << "LE" << "\t" << "M" << std::endl;
      std::cout << times[0] << "\t" << "0.0" << "\t" << PetscRealPart(z_mag) << std::endl;
    }
    else if(!magn && sites){
      //std::cout << "Time" << "\t" << "LE" << "\t" << "HM" << std::endl;
      std::cout << times[0] << "\t" << "0.0" << "\t" << PetscRealPart(mid_mag) << std::endl;
    }
    else{
      //std::cout << "Time" << "\t" << "LE" << "\t" << "M" << "\t" << "HM" << std::endl;
      std::cout << times[0] << "\t" << "0.0" << "\t" << PetscRealPart(z_mag) << 
        "\t" << PetscRealPart(mid_mag) << std::endl;
    }
  }

  for(unsigned int tt = 1; tt < (iterations + 1); ++tt){

    FNSetScale(f_, (times[tt] - times[tt - 1]) * PETSC_i, 1.0);
    MFNSolve(mfn_, vec_help, vec_help);

    MFNGetConvergedReason(mfn_, &reason);
    if(reason < 0){
      std::cerr << "Solver did not converge, aborting" << std::endl;
      std::cerr << "Change tolerance or maximum number of iterations" << std::endl;
      MPI_Abort(PETSC_COMM_WORLD, 1);
    }

    // Echo
    VecDot(v, vec_help, &l_echo);
    double g_val = (PetscRealPart(l_echo) * PetscRealPart(l_echo))
      + (PetscImaginaryPart(l_echo) * PetscImaginaryPart(l_echo));
    loschmidt = (-1.0 / l_) * std::log(g_val);

    // Magnetization
    if(magn){
      VecPointwiseMult(mag_help, sparse.magnetization, vec_help);
      VecDot(mag_help, vec_help, &z_mag);
    }
    if(sites){
      VecPointwiseMult(h_help, sparse.hmag, vec_help);
      VecDot(h_help, vec_help, &mid_mag);
    }

    if(mpirank_ == 0){
      if(!magn && !sites){
        std::cout << times[tt] << "\t" << loschmidt << std::endl;
      }
      else if(magn && !sites){
        std::cout << times[tt] << "\t" << loschmidt << "\t" << PetscRealPart(z_mag) << std::endl;
      }
      else if(!magn && sites){
        std::cout << times[tt] << "\t" << loschmidt << "\t" << PetscRealPart(mid_mag) 
          << std::endl;
      }
      else{
        std::cout << times[tt] << "\t" << loschmidt << "\t" << PetscRealPart(z_mag) << 
          "\t" << PetscRealPart(mid_mag) << std::endl;
      }
    }
  }
 
  VecDestroy(&vec_help);
  VecDestroy(&mag_help);
  VecDestroy(&h_help);
}


