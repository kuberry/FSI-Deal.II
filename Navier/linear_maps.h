#ifndef LINEAR_MAPS_H
#define LINEAR_MAPS_H
#include "FSI_Project.h"

namespace LinearMap {

  template <int dim>
    class Linearized_Operator
    {
    public:
    Linearized_Operator(FSIProblem<dim> *sim): problem_space(sim), matrix_assembled(false), matrix_initialized(false) {};
      // Return the L2 norm of the vector as defined by
      // sqrt( \int_{fluid_interface} function^2 dx )

    void vmult (Vector<double> &dst,
		const Vector<double> &src) {

      if (!matrix_assembled) total_solves = 0; // restart the count since we are dealing with a new sequence of runs

      problem_space->rhs_for_linear *= 0;
      problem_space->vector_vector_transfer_interface_dofs(src, problem_space->rhs_for_linear.block(0),0,0);
      problem_space->vector_vector_transfer_interface_dofs(src, problem_space->rhs_for_linear.block(1),0,1,problem_space->Displacement);
      problem_space->rhs_for_linear.block(1) *= -1;

      // only assembly the matrix operator if it isn't currently assembled (once each time step)
      Threads::Task<void> s_assembly = Threads::new_task(&FSIProblem<dim>::assemble_structure, *problem_space, problem_space->linear, !matrix_assembled);
      Threads::Task<void> f_assembly = Threads::new_task(&FSIProblem<dim>::assemble_fluid, *problem_space, problem_space->linear, !matrix_assembled);	      
      f_assembly.join();
      problem_space->dirichlet_boundaries(static_cast<enum FSIProblem<dim>::System >(0),problem_space->linear);
      s_assembly.join();
      problem_space->dirichlet_boundaries(static_cast<enum FSIProblem<dim>::System >(1),problem_space->linear);

      if (matrix_initialized) {
        Threads::Task<void> f_factor = Threads::new_task(&SparseDirectUMFPACK::factorize<SparseMatrix<double> >,problem_space->linear_solver[0], problem_space->linear_matrix.block(0,0));
        Threads::Task<void> s_factor = Threads::new_task(&SparseDirectUMFPACK::factorize<SparseMatrix<double> >,problem_space->linear_solver[1], problem_space->linear_matrix.block(1,1));
        f_factor.join();
        Threads::Task<void> f_solve = Threads::new_task(&FSIProblem<dim>::solve,*problem_space,problem_space->linear_solver[0],0,problem_space->linear);
        s_factor.join();
        Threads::Task<void> s_solve = Threads::new_task(&FSIProblem<dim>::solve,*problem_space,problem_space->linear_solver[1],1,problem_space->linear);
        f_solve.join();
        s_solve.join();
      } else {
	ExcNotInitialized();
      }

      total_solves += 2;

      dst *= 0;
      if (problem_space->fem_properties.adjoint_type==1)
        {
          // -Ax = -w^n + phi^n/dt	  
	  Vector<double> tmp(src.size());
          problem_space->vector_vector_transfer_interface_dofs(problem_space->linear_solution.block(1),dst,1,0,problem_space->Displacement);
          dst*=1./problem_space->time_step;
          problem_space->vector_vector_transfer_interface_dofs(problem_space->linear_solution.block(0),tmp,0,0);
          dst-=tmp;
        }
      else
        {
          // -Ax = -w^n + phi_dot^n
	  Vector<double> tmp(src.size());
          problem_space->vector_vector_transfer_interface_dofs(problem_space->linear_solution.block(1),dst,1,0,problem_space->Velocity);
          problem_space->vector_vector_transfer_interface_dofs(problem_space->linear_solution.block(0),tmp,0,0);
          dst-=tmp;
        }

      //matrix_assembled = true;
      //matrix_initialized = true;
    };
    // Application of transpose to a vector.
    // Only used by some iterative methods.
    void Tvmult (Vector<double> &dst,
		 const Vector<double> &src) const {
      AssertThrow(false, ExcNotImplemented());
    };

    void set_matrix_assembled(bool value) {
      matrix_assembled = value;
    };

    void initialize_matrix() {
      problem_space->linear_solver[0].initialize(problem_space->linear_matrix.block(0,0));
      problem_space->linear_solver[1].initialize(problem_space->linear_matrix.block(1,1));
      Threads::Task<void> f_solve = Threads::new_task(&FSIProblem<dim>::solve,*problem_space,problem_space->linear_solver[0],0,problem_space->linear);
      Threads::Task<void> s_solve = Threads::new_task(&FSIProblem<dim>::solve,*problem_space,problem_space->linear_solver[1],1,problem_space->linear);
      f_solve.join();
      s_solve.join();
      matrix_initialized = true;
    };
  
    private:
      FSIProblem<dim> *problem_space;
      unsigned int total_solves;
      bool matrix_assembled;
      bool matrix_initialized;
    };


  class Wilkinson
  {
    /*
      This matrix is for testing purposes.
    */
  public:
  Wilkinson(const unsigned int size=100):n(size),A(size,size){
      for (unsigned int i=0; i<n; i++) {
	for (unsigned int j=0; j<n; j++) {
	  if (i==j) {
	    A.set(i,i,std::fabs(.5*(n-1)-j));
	  } else if (j==i+1 || j==i-1) {
	    A.set(i,j,1.0);
	  }
	}
      }
    };   

    // Application of matrix to vector src.
    // Write result into dst
    void vmult (Vector<double> &dst,
		const Vector<double> &src) const {
      A.vmult(dst, src);
    };
    // Application of transpose to a vector.
    // Only used by some iterative methods.
    void Tvmult (Vector<double> &dst,
		 const Vector<double> &src) const {
      AssertThrow(false, ExcNotImplemented());
    };

  private:
    const unsigned int n;
    FullMatrix<double> A;
    unsigned int total_solves;
  };

  template <int dim>
    class NeumannVector: public Vector<double>
    {
    public:
    NeumannVector(Vector<double> &src, FSIProblem<dim> *sim):Vector<double>(src), problem_space(sim) {};
      // Return the L2 norm of the vector as defined by
      // sqrt( \int_{fluid_interface} function^2 dx )
      double l2_norm () const {
	return problem_space->interface_norm(*this);
      };
  
    private:
      FSIProblem<dim> *problem_space;
    };


  /* class Vector//: public Vector<double> */
  /* { */
  /*  public: */
  /*   // Resize the current object to have */
  /*   // the same size and layout as the model_vector */
  /*   // argument provided. The second argument */
  /*   // indicates whether to clear the current */
  /*   // object after resizing. */
  /*   // The second argument must have */
  /*   // a default value equal to false */
  /*   void reinit (const Vector &model_vector, */
  /* 	       const bool leave_elements_uninitialized = false); */
  /*   // Inner product between the current object */
  /*   // and the argument */
  /*   double operator * (const Vector &v) const; */
  /*   // Addition of vectors */
  /*   void add (const Vector &x); */
  /*   // Scaled addition of vectors */
  /*   void add (const double a, */
  /* 	    const Vector &x); */
  /*   // Scaled addition of vectors */
  /*   void sadd (const double a, */
  /* 	     const double b, */
  /* 	     const Vector &x); */
  /*   // Scaled assignment of a vector */
  /*   void equ (const double a, */
  /* 	    const Vector &x); */
  /*   // Combined scaled addition of vector x into */
  /*   // the current object and subsequent inner */
  /*   // product of the current object with v */
  /*   double add_and_dot (const double a, */
  /* 		      const Vector &x, */
  /* 		      const Vector &v); */
  /*   // Multiply the elements of the current */
  /*   // object by a fixed value */
  /*   Vector & operator *= (const double a); */
  /*   // Return the l2 norm of the vector */
  /*   double l2_norm () const; */
  /* }; */

    }

#endif
