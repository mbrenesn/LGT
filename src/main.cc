#include <iostream>
#include <sstream>

#include "Environment.h"
#include "Basis.h"
#include "SparseOp.h"
#include "TimeEvo.h"
#include "HalfChainEnt.h"

#include <petsctime.h>

int main(int argc, char **argv)
{
  unsigned int l = 777;
  unsigned int n = 777;
  double J = 0.7771777;
  double w = 0.7771777;
  double m = 0.7771777;

  if(argc < 7){
    std::cerr << "Usage: mpirun -np <proc> " << argv[0] << 
      " -l [sites] -n [fill] -J [J] -w [w] -m [m] -[PETSc/SLEPc options]" << std::endl;
    exit(1);
  }

  for(int i = 0; i < argc; ++i){
    std::string str = argv[i];
    if(str == "-l") l = atoi(argv[i + 1]);
    else if(str == "-n") n = atoi(argv[i + 1]);
    else if(str == "-J") J = atof(argv[i + 1]);
    else if(str == "-w") w = atof(argv[i + 1]);
    else if(str == "-m") m = atof(argv[i + 1]);
    else continue;
  }

  if(l == 777 || n == 777 || J == 0.7771777 || w == 0.7771777 || m == 0.777177){
    std::cerr << "Error setting parameters" << std::endl;
    std::cerr << "Usage: mpirun -np <proc> " << argv[0] << 
      " -l [sites] -n [fill] -J [J] -w [w] -m [m] -[PETSc/SLEPc options]" << std::endl;
    exit(1);
  }
  
  double V = J / 2.0;
  double h = m / 2.0; 
  double t = w;

  // Establish the environment
  Environment env(argc, argv, l, n);

  PetscLogDouble time1, time2;
  PetscTime(&time1);

  PetscMPIInt mpirank = env.mpirank;
  PetscMPIInt mpisize = env.mpisize;

  if(mpirank == 0){  
    std::cout << "LGT" << std::endl;    
    std::cout << "System has " << l << " sites and " << n << " spins ^" << std::endl;
    std::cout << "Simulation with " << mpisize << " total MPI processes" << std::endl;
    std::cout << "Parameters: J = " << J << " w = " << w << " m = " << m << std::endl;
  }

  // Establish the basis environment, by pointer, to call an early destructor and reclaim
  // basis memory
  Basis *basis = new Basis(env);

  // Construct basis
  basis->construct_int_basis();
  //basis->print_basis(env);

  // Establish Hamiltonian operator environment
  SparseOp schwinger(env, *basis);
  // Create a matrix object that will be used as Hamiltonian
  Mat sch_mat;
  MatCreate(PETSC_COMM_WORLD, &sch_mat);
  MatSetSizes(sch_mat, basis->nlocal, basis->nlocal, basis->basis_size, basis->basis_size);
  MatSetType(sch_mat, MATMPIAIJ);
  
  // Construct the Hamiltonian matrix
  schwinger.construct_schwinger_hamiltonian(sch_mat, basis->int_basis, V, t, h,
    true, true, true);

  // Neel index before destroying basis, returns the index on the processor that found it,
  // and -1 on all the rest
  LLInt neel_index = schwinger.get_neel_index(*basis);

  // Random index before destroying basis
  //LLInt random_index = schwinger.get_random_index(*basis, true, true);

  // Call basis destructor to reclaim memory
  //delete basis;

  // Initial state with the same parallel layout as the matrix
  Vec v;
  MatCreateVecs(sch_mat, NULL, &v);
  VecZeroEntries(v);
  if(neel_index != -1) VecSetValue(v, neel_index, 1.0, INSERT_VALUES);
  //VecSetValue(v, random_index, 1.0, INSERT_VALUES);
  VecAssemblyBegin(v);
  VecAssemblyEnd(v);

  //VecView(v, PETSC_VIEWER_STDOUT_WORLD);

  /*** Time evolultion ***/
  PetscLogDouble kryt1, kryt2;
  PetscTime(&kryt1);

  double tol = 1.0e-7;
  int maxits = 100000;

  /*
  const int iterations = 102;
  double times[iterations + 1] 
      = {0.0,0.1,0.5,1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0,
         11.0,12.0,13.0,14.0,15.0,16.0,17.0,18.0,19.0,20.0,
         21.0,22.0,23.0,24.0,25.0,26.0,27.0,28.0,29.0,30.0,
         31.0,32.0,33.0,34.0,35.0,36.0,37.0,38.0,39.0,40.0,
         41.0,42.0,43.0,44.0,45.0,46.0,47.0,48.0,49.0,50.0,
         51.0,52.0,53.0,54.0,55.0,56.0,57.0,58.0,59.0,60.0,
         61.0,62.0,63.0,64.0,65.0,66.0,67.0,68.0,69.0,70.0,
         71.0,72.0,73.0,74.0,75.0,76.0,77.0,78.0,79.0,80.0,
         81.0,82.0,83.0,84.0,85.0,86.0,87.0,88.0,89.0,90.0,
         91.0,92.0,93.0,94.0,95.0,96.0,97.0,98.0,99.0,100.0};

  */  
  const int iterations = 196;
  double times[iterations + 1] 
      = {0.0,0.5,0.6,0.7,0.8,0.9,1.0,
         1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2.0,
         2.1,2.2,2.3,2.4,2.5,2.6,2.7,2.8,2.9,3.0,
         3.1,3.2,3.3,3.4,3.5,3.6,3.7,3.8,3.9,4.0,
         4.1,4.2,4.3,4.4,4.5,4.6,4.7,4.8,4.9,5.0,
         5.1,5.2,5.3,5.4,5.5,5.6,5.7,5.8,5.9,6.0,
         6.1,6.2,6.3,6.4,6.5,6.6,6.7,6.8,6.9,7.0,
         7.1,7.2,7.3,7.4,7.5,7.6,7.7,7.8,7.9,8.0,
         8.1,8.2,8.3,8.4,8.5,8.6,8.7,8.8,8.9,9.0,
         9.1,9.2,9.3,9.4,9.5,9.6,9.7,9.8,9.9,10.0,
         10.1,10.2,10.3,10.4,10.5,10.6,10.7,10.8,10.9,11.0,
         11.1,11.2,11.3,11.4,11.5,11.6,11.7,11.8,11.9,12.0,
         12.1,12.2,12.3,12.4,12.5,12.6,12.7,12.8,12.9,13.0,
         13.1,13.2,13.3,13.4,13.5,13.6,13.7,13.8,13.9,14.0,
         14.1,14.2,14.3,14.4,14.5,14.6,14.7,14.8,14.9,15.0,
         15.1,15.2,15.3,15.4,15.5,15.6,15.7,15.8,15.9,16.0,
         16.1,16.2,16.3,16.4,16.5,16.6,16.7,16.8,16.9,17.0,
         17.1,17.2,17.3,17.4,17.5,17.6,17.7,17.8,17.9,18.0,
         18.1,18.2,18.3,18.4,18.5,18.6,18.7,18.8,18.9,19.0,
         19.1,19.2,19.3,19.4,19.5,19.6,19.7,19.8,19.9,20.0};
  
  /*
  const int iterations = 111;
  double times[iterations + 1] 
      = {0.0,0.1,0.5,1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0,
         20.0,30.0,40.0,50.0,60.0,70.0,80.0,90.0,100.0,
         110.0,120.0,130.0,140.0,150.0,160.0,170.0,180.0,190.0,200.0,
         210.0,220.0,230.0,240.0,250.0,260.0,270.0,280.0,290.0,300.0,
         310.0,320.0,330.0,340.0,350.0,360.0,370.0,380.0,390.0,400.0,
         410.0,420.0,430.0,440.0,450.0,460.0,470.0,480.0,490.0,500.0,
         510.0,520.0,530.0,540.0,550.0,560.0,570.0,580.0,590.0,600.0,
         610.0,620.0,630.0,640.0,650.0,660.0,670.0,680.0,690.0,700.0,
         710.0,720.0,730.0,740.0,750.0,760.0,770.0,780.0,790.0,800.0,
         810.0,820.0,830.0,840.0,850.0,860.0,870.0,880.0,890.0,900.0,
         910.0,920.0,930.0,940.0,950.0,960.0,970.0,980.0,990.0,1000.0};
  */

  //TimeEvo te(env);
  //te.loschmidt_echo(schwinger, iterations, times, tol, maxits, v, sch_mat, true, true);
 
  /*** End of time evolution ***/
  
  /*** Entanglement entropy ***/

  HalfChainEnt halfent(env, *basis);
  halfent.von_neumann_entropy(basis->int_basis, v, sch_mat, iterations, times, tol, maxits);

  /*** End of entanglement entropy ***/

  PetscTime(&kryt2);

  delete basis;

  VecDestroy(&v);
  MatDestroy(&sch_mat);

  PetscTime(&time2);

  if(mpirank == 0){  
    std::cout << "Time Krylov Evolution: " << kryt2 - kryt1 << " seconds" << std::endl;
    std::cout << "Time total: " << time2 - time1 << " seconds" << std::endl; 
  }

  return 0;
}
