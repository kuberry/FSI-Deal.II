#ifndef LINEAR_MAPS_H
#define LINEAR_MAPS_H
#include "FSI_Project.h"
#include "data1.h"

// temporarily: never let matrix_assembled be true
// solve ALE in loop
// 

namespace LinearMap {

  template <int dim>
    class Linearized_Operator
    {
    public:
    Linearized_Operator(FSIProblem<dim> *sim): problem_space(sim), matrix_assembled(false), matrix_initialized(false) {};
      // Return the L2 norm of the vector as defined by
      // sqrt( \int_{fluid_interface} function^2 dx )

    void vmult (Vector<double> &dst,
		const Vector<double> &src) const {
      double error_norm = 1;
      while (error_norm > 1e-12) {
	Vector<double> reference = dst;
	AleBoundaryValues<dim> ale_boundary_values(problem_space->physical_properties);
	problem_space->stress.block(0) = problem_space->stress_star.block(0);
	problem_space->stress.block(0).add(1.0, dst);
	Threads::Task<> s_solver = Threads::new_task(&FSIProblem<dim>::structure_state_solve,*problem_space, initialized_timestep_number);
	s_solver.join();
	if (problem_space->physical_properties.moving_domain)
	  {
	    problem_space->assemble_ale(problem_space->state,true);
	    problem_space->dirichlet_boundaries((enum FSIProblem<dim>::System)2,problem_space->state);
	    problem_space->state_solver[2].factorize(problem_space->system_matrix.block(2,2));
	    problem_space->solve(problem_space->state_solver[2],2,problem_space->state);
	    problem_space->transfer_all_dofs(problem_space->solution,problem_space->mesh_displacement_star,2,0);

	    if (problem_space->physical_properties.simulation_type==2)
	      {
		// Overwrites the Laplace solve since the velocities compared against will not be correct
		ale_boundary_values.set_time(problem_space->time);
		VectorTools::project(problem_space->ale_dof_handler, problem_space->ale_constraints, QGauss<dim>(problem_space->fem_properties.fluid_degree+2),
				     ale_boundary_values,
				     problem_space->mesh_displacement_star.block(2)); // move directly to fluid block 
		problem_space->transfer_all_dofs(problem_space->mesh_displacement_star,problem_space->mesh_displacement_star,2,0);
	      }
	    problem_space->mesh_displacement_star_old.block(0) = problem_space->mesh_displacement_star.block(0); // Not currently implemented, but will allow for half steps

	    if (problem_space->fem_properties.time_dependent) {
	      problem_space->mesh_velocity.block(0)=problem_space->mesh_displacement_star.block(0);
	      problem_space->mesh_velocity.block(0)-=problem_space->old_mesh_displacement.block(0);
	      problem_space->mesh_velocity.block(0)*=1./problem_space->time_step;
	    }
	  }

	//if (!matrix_assembled) total_solves = 0; // restart the count since we are dealing with a new sequence of runs
	assemble_matrix(dst, src);
	/* problem_space->rhs_for_linear *= 0; */
	/* problem_space->vector_vector_transfer_interface_dofs(src, problem_space->rhs_for_linear.block(0),0,0); */
	/* problem_space->vector_vector_transfer_interface_dofs(src, problem_space->rhs_for_linear.block(1),0,1,problem_space->Displacement); */
	/* problem_space->rhs_for_linear.block(1) *= -1; */

	/* // only assembly the matrix operator if it isn't currently assembled (once each time step) */
	/* Threads::Task<void> s_assembly = Threads::new_task(&FSIProblem<dim>::assemble_structure, *problem_space, problem_space->linear, !matrix_assembled); */
	/* Threads::Task<void> f_assembly = Threads::new_task(&FSIProblem<dim>::assemble_fluid, *problem_space, problem_space->linear, !matrix_assembled);	       */
	/* f_assembly.join(); */
	/* problem_space->dirichlet_boundaries(static_cast<enum FSIProblem<dim>::System >(0),problem_space->linear); */
	/* s_assembly.join(); */
	/* problem_space->dirichlet_boundaries(static_cast<enum FSIProblem<dim>::System >(1),problem_space->linear); */

	if (matrix_initialized) {
	  if (mode==problem_space->linear) {
	    Threads::Task<void> f_solve = Threads::new_task(&FSIProblem<dim>::solve,*problem_space,problem_space->linear_solver[0],0,problem_space->linear);
	    Threads::Task<void> s_solve = Threads::new_task(&FSIProblem<dim>::solve,*problem_space,problem_space->linear_solver[1],1,problem_space->linear);
	    f_solve.join();
	    s_solve.join();
	  } else { // adjoint
	    Threads::Task<void> f_solve = Threads::new_task(&FSIProblem<dim>::solve,*problem_space,problem_space->adjoint_solver[0],0,problem_space->adjoint);
	    Threads::Task<void> s_solve = Threads::new_task(&FSIProblem<dim>::solve,*problem_space,problem_space->adjoint_solver[1],1,problem_space->adjoint);
	    f_solve.join();
	    s_solve.join();
	  }
	} else {
	  ExcNotInitialized();
	}

	//total_solves += 2;

	//*************************************************************
	//      FORM OPERATOR OUTPUT FROM SUBSYSTEM SOLUTIONS
	//*************************************************************
	dst *= 0;
	if (mode==problem_space->linear) {
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
	} else {//adjoint 
	  Vector<double> tmp(src.size());
	  if (problem_space->fem_properties.adjoint_type==1)
	    {
	      problem_space->vector_vector_transfer_interface_dofs(problem_space->adjoint_solution.block(1),dst,1,0,problem_space->Displacement);
	    }
	  else
	    {
	      problem_space->vector_vector_transfer_interface_dofs(problem_space->adjoint_solution.block(1),dst,1,0,problem_space->Velocity);
	    }
	  dst *= problem_space->fem_properties.structure_theta;
	  dst.add(-problem_space->fem_properties.fluid_theta,problem_space->adjoint_solution.block(0));

	  /* // THIS NEEDS TO BE LOOKED AT!!! */
	  /* if (problem_space->fem_properties.adjoint_type==1) */
	  /*   { */
	  /*     // -Ax = -w^n + phi^n/dt	   */
	  /*     Vector<double> tmp(src.size()); */
	  /*     problem_space->vector_vector_transfer_interface_dofs(problem_space->adjoint_solution.block(1),dst,1,0,problem_space->Displacement); */
	  /*     dst*=-1./problem_space->time_step; */
	  /*     problem_space->vector_vector_transfer_interface_dofs(problem_space->adjoint_solution.block(0),tmp,0,0); */
	  /*     dst+=tmp; */
	  /*   } */
	  /* else */
	  /*   { */
	  /*     // -Ax = -w^n + phi_dot^n */
	  /*     Vector<double> tmp(src.size()); */
	  /*     problem_space->vector_vector_transfer_interface_dofs(problem_space->adjoint_solution.block(1),dst,1,0,problem_space->Velocity); */
	  /*     dst*=-1; */
	  /*     problem_space->vector_vector_transfer_interface_dofs(problem_space->adjoint_solution.block(0),tmp,0,0); */
	  /*     dst+=tmp; */
	  /*   } */
	}
	problem_space->old_old_solution.block(0)*=0;
	problem_space->vector_vector_transfer_interface_dofs(problem_space->adjoint_solution.block(1),problem_space->old_old_solution.block(0),1,0,problem_space->Displacement);
	reference.add(-1.0,dst);
	error_norm = reference.l2_norm();
	std::cout << "Error in loop: " << error_norm << std::endl;
      };
    };


    // Application of transpose to a vector.
    // Only used by some iterative methods.
    void Tvmult (Vector<double> &dst,
		 const Vector<double> &src) const {
      AssertThrow(false, ExcNotImplemented());
    };



    void initialize_matrix(Vector<double> &dst,
			   const Vector<double> &src, enum FSIProblem<dim>::Mode mode_, unsigned int initialized_timestep_number_) {
      mode = mode_;
      reassemble_operator(dst, src);
      if (mode==problem_space->linear) {
	problem_space->linear_solver[0].initialize(problem_space->linear_matrix.block(0,0));
	problem_space->linear_solver[1].initialize(problem_space->linear_matrix.block(1,1));
      } else { //adjoint
	problem_space->adjoint_solver[0].initialize(problem_space->adjoint_matrix.block(0,0));
	problem_space->adjoint_solver[1].initialize(problem_space->adjoint_matrix.block(1,1));
      }
      matrix_initialized = true;
      initialized_timestep_number = initialized_timestep_number_;
    };      


    void assemble_matrix(Vector<double> &dst,
		const Vector<double> &src) const {
      if (mode==problem_space->linear) {
	problem_space->rhs_for_linear *= 0;
	problem_space->vector_vector_transfer_interface_dofs(src, problem_space->rhs_for_linear.block(0),0,0);
	problem_space->vector_vector_transfer_interface_dofs(src, problem_space->rhs_for_linear.block(1),0,1,problem_space->Displacement);
	problem_space->rhs_for_linear.block(1) *= -1;
      } else { //adjoint
	problem_space->rhs_for_adjoint *= 0;
	problem_space->vector_vector_transfer_interface_dofs(src, problem_space->rhs_for_adjoint.block(0),0,0);
	if (problem_space->fem_properties.adjoint_type==1) {
	  problem_space->vector_vector_transfer_interface_dofs(src, problem_space->rhs_for_adjoint.block(1),0,1,problem_space->Displacement);
	} else {
	  problem_space->vector_vector_transfer_interface_dofs(src, problem_space->rhs_for_adjoint.block(1),0,1,problem_space->Velocity);
	}
	problem_space->rhs_for_adjoint.block(1) *= -1;
      }

      // only assembly the matrix operator if it isn't currently assembled (once each time step)
      Threads::Task<void> s_assembly = Threads::new_task(&FSIProblem<dim>::assemble_structure, *problem_space, mode, !matrix_assembled);
      Threads::Task<void> f_assembly = Threads::new_task(&FSIProblem<dim>::assemble_fluid, *problem_space, mode, !matrix_assembled);	      
      f_assembly.join();
      problem_space->dirichlet_boundaries(static_cast<enum FSIProblem<dim>::System >(0), mode);
      s_assembly.join();
      problem_space->dirichlet_boundaries(static_cast<enum FSIProblem<dim>::System >(1), mode);
    };

    void reassemble_operator(Vector<double> &dst,
		const Vector<double> &src) {
      assemble_matrix(dst, src);
      if (mode==problem_space->linear) {
	Threads::Task<void> f_factor = Threads::new_task(&SparseDirectUMFPACK::factorize<SparseMatrix<double> >,problem_space->linear_solver[0], problem_space->linear_matrix.block(0,0));
	Threads::Task<void> s_factor = Threads::new_task(&SparseDirectUMFPACK::factorize<SparseMatrix<double> >,problem_space->linear_solver[1], problem_space->linear_matrix.block(1,1));
	f_factor.join();
	s_factor.join();
      } else { // adjoint
	Threads::Task<void> f_factor = Threads::new_task(&SparseDirectUMFPACK::factorize<SparseMatrix<double> >,problem_space->adjoint_solver[0], problem_space->adjoint_matrix.block(0,0));
	Threads::Task<void> s_factor = Threads::new_task(&SparseDirectUMFPACK::factorize<SparseMatrix<double> >,problem_space->adjoint_solver[1], problem_space->adjoint_matrix.block(1,1));
	f_factor.join();
	s_factor.join();
      }
      //matrix_assembled = true;
    };

    void set_matrix_assembled_false() {
      matrix_assembled = false;
    };

    private:
      FSIProblem<dim> *problem_space;
      unsigned int total_solves;
      bool matrix_assembled;
      bool matrix_initialized;
      enum FSIProblem<dim>::Mode mode;
      unsigned int initialized_timestep_number;
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
