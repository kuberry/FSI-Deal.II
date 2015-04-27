#include "FSI_Project.h"
#include <deal.II/base/timer.h>
#include <deal.II/lac/solver_control.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/solver_bicgstab.h>
#include <deal.II/lac/solver_cg.h>
#include "linear_maps.h"
#include "solver_controls.h"

template <int dim>
void FSIProblem<dim>::run ()
{
  unsigned int total_timesteps = (double)(fem_properties.T-fem_properties.t0)/time_step;
  master_thread = Threads::this_thread_id();
  ConditionalOStream pcout(std::cout,Threads::this_thread_id()==master_thread); 

  if (!fem_properties.time_dependent) {
    pcout << "STATIONARY Problem Selected. structure_theta, fluid_theta set to 1.0." << std::endl; 
    fem_properties.structure_theta = 1.0;
    fem_properties.fluid_theta = 1.0;
  }

  // Make a quick check that we don't try to initialize to a non-one timestep with Richardson extrapolation
  // because it requires saving more time steps of data than we are storing.
  if (timestep_number>1 && fem_properties.richardson) AssertThrow(false, ExcNotImplemented());

  std::ofstream structure_file_out;
  std::ofstream fluid_file_out;
  if (physical_properties.simulation_type==1 || physical_properties.simulation_type==3 || physical_properties.simulation_type==4)
    {
      structure_file_out.open("structure_output.txt");
      fluid_file_out.open("fluid_output.txt");
    }
 
  TimerOutput timer (pcout, TimerOutput::summary,
  		     TimerOutput::wall_times);
  timer.enter_subsection ("Everything");
  timer.enter_subsection ("Setup dof system");

  setup_system();

  // Threads::Task<void>
  //  task = Threads::new_task (&FSIProblem<dim>::build_dof_mapping,*this);
  build_dof_mapping();

  timer.leave_subsection();

  StructureBoundaryValues<dim> structure_boundary_values(physical_properties);
  FluidBoundaryValues<dim> fluid_boundary_values(physical_properties, fem_properties);
  AleBoundaryValues<dim> ale_boundary_values(physical_properties);

  StructureStressValues<dim> structure_boundary_stress(physical_properties);
  FluidStressValues<dim> fluid_boundary_stress(physical_properties);

  structure_boundary_values.set_time(fem_properties.t0-time_step);
  fluid_boundary_values.set_time(fem_properties.t0-time_step);

  if (fem_properties.richardson || fem_properties.optimization_method.compare("DN")==0) {
    VectorTools::project (fluid_dof_handler, fluid_constraints, QGauss<dim>(fem_properties.fluid_degree+2),
			  fluid_boundary_values,
			  old_old_solution.block(0));
    VectorTools::project (structure_dof_handler, structure_constraints, QGauss<dim>(fem_properties.structure_degree+2),
			  structure_boundary_values,
			  old_old_solution.block(1));
  }

  structure_boundary_values.set_time(fem_properties.t0);
  fluid_boundary_values.set_time(fem_properties.t0);

  structure_boundary_stress.set_time(fem_properties.t0);
  fluid_boundary_stress.set_time(fem_properties.t0);

  if (timestep_number == 1) {
    VectorTools::project (fluid_dof_handler, fluid_constraints, QGauss<dim>(fem_properties.fluid_degree+2),
			  fluid_boundary_values,
			  old_solution.block(0));
    VectorTools::project (structure_dof_handler, structure_constraints, QGauss<dim>(fem_properties.structure_degree+2),
			  structure_boundary_values,
			  old_solution.block(1));
    if (physical_properties.simulation_type!=2)
      {
	VectorTools::project(fluid_dof_handler, fluid_constraints, QGauss<dim>(fem_properties.fluid_degree+2),
			     fluid_boundary_stress,
			     old_stress.block(0));
      }
  } else {
    const std::string solution_filename = "solution.data";
    std::ifstream solution_input (solution_filename.c_str());
    //solution_input.precision(20);
    old_solution.block_read(solution_input);
    solution = old_solution;
    solution_input.close();
    const std::string stress_filename = "stress.data";
    std::ifstream stress_input (stress_filename.c_str());
    //stress_input.precision(20);
    old_stress.block_read(stress_input);
    stress_input.close();
  }
  // task.join();
  transfer_interface_dofs(old_stress,old_stress,0,1);
  stress=old_stress;
  stress_star=stress;
  // Note to self: On a moving domain, predotting stress tensor with unit normal requires extra work (pull backs)
  // If only retrieving the stress tensor, it can be dotted with the moving unit normal

  if (physical_properties.moving_domain)
    {
      if (timestep_number==1) {
	if (physical_properties.simulation_type==2)
	  {
	    // Directly solved instead of Laplace solve since the velocities compared against would otherwise not be correct
	    ale_boundary_values.set_time(fem_properties.t0);
	    VectorTools::project(ale_dof_handler, ale_constraints, QGauss<dim>(fem_properties.fluid_degree+2),
				 ale_boundary_values,
				 mesh_displacement_star.block(2)); // move directly to fluid block 
	    transfer_all_dofs(mesh_displacement_star,mesh_displacement_star,2,0);
	  }
	else
	  {
	    solution.block(1)=old_solution.block(1); // solutions sets boundary values for Laplace solve
	    assemble_ale(state,true);
	    dirichlet_boundaries((System)2,state);
	    state_solver[2].factorize(system_matrix.block(2,2));
	    solve(state_solver[2],2,state);
	    transfer_all_dofs(solution,mesh_displacement_star,2,0);
	  }
      } else {
	const std::string mesh_filename = "mesh.data";
	std::ifstream mesh_input (mesh_filename.c_str());
	mesh_displacement_star.block_read(mesh_input);
	mesh_input.close();
      }
    }

  Vector<double> lift(total_timesteps);
  Vector<double> drag(total_timesteps);
  Vector<double> x_displacement(total_timesteps);
  Vector<double> y_displacement(total_timesteps);
  Vector<double> outflow_rate(total_timesteps);
  // Read in lift, drag, and displacement data since it is available
  if (timestep_number != 1) {
    const std::string lift_filename = "lift.data";
    std::ifstream lift_input (lift_filename.c_str());
    lift.block_read(lift_input);
    lift_input.close();
    const std::string drag_filename = "drag.data";
    std::ifstream drag_input (drag_filename.c_str());
    drag.block_read(drag_input);
    drag_input.close();	  
    const std::string x_displacement_filename = "x_displacement.data";
    std::ifstream x_displacement_input (x_displacement_filename.c_str());
    x_displacement.block_read(x_displacement_input);
    x_displacement_input.close();	  
    const std::string y_displacement_filename = "y_displacement.data";
    std::ifstream y_displacement_input (y_displacement_filename.c_str());
    y_displacement.block_read(y_displacement_input);
    y_displacement_input.close();
    const std::string outflow_rate_filename = "outflow_rate.data";
    std::ifstream outflow_rate_input (outflow_rate_filename.c_str());
    outflow_rate.block_read(outflow_rate_input);
    outflow_rate_input.close();
  }

  // Dirichlet-Neumann coupling is wrapped in its own solver
  DN_solver<dim> DN(this); 

  // *****************************************************************************************
  //                                    TIME STEP LOOP
  // *****************************************************************************************
  double total_time = 0;
  const unsigned int initialized_timestep_number = timestep_number;
  // timestep_number = 1 by default, is something else if given as 2nd command line argument to FSI_Project
  for (; timestep_number<=total_timesteps; ++timestep_number)
    {
      // AG_line_search is a mode of operation that is entered when an update to the control is computed
      // We remain in this mode until sufficient improvement is made and then this mode is exited from.
      // At the beginning of each time step, we are not in line search mode.
      bool AG_line_search = false;
      // Backwards line search has simpler criteria for exiting the line search mode
      bool backwards_line_search = false;

      // This ensures that we only perform one 'timestep' in the case of a stationary problem
      if (!fem_properties.time_dependent) {
	timestep_number=total_timesteps;
      }

      boost::timer t;
      time = fem_properties.t0 + timestep_number*time_step;
      pcout << std::endl << "----------------------------------------" << std::endl;
      pcout << "Time step " << timestep_number
  		<< " at t=" << time
  		<< std::endl;

      double velocity_jump = 1;
      double velocity_jump_old = 2;

      unsigned int count = 0;

      rhs_for_adjoint=1;

      double alpha = fem_properties.steepest_descent_alpha;
      unsigned int imprecord = 0;
      unsigned int relrecord = 0;
      unsigned int consecutiverelrecord = 0;

      // double update_alpha = 1.0;

      //stress=old_stress;

      unsigned int total_solves = 0;


      if (physical_properties.moving_domain) {
	old_mesh_displacement.block(0) = mesh_displacement_star.block(0);
      }

      // (Armijo-Goldstein/Backwards) Line Search Parameters
      double n_max = 0;
      double tau_t = 0;
      BlockVector<double> update_direction = stress;

      double alpha_j = 1.0;      
      if (fem_properties.optimization_method.compare("Gradient")==0)
	alpha_j = fem_properties.steepest_descent_alpha;
      double t_val = 0;

      // *****************************************************************************************
      //                                OUTER OPTIMIZATION ITERATION LOOP
      // *****************************************************************************************
      while (true)
        {
	  // (Armijo-Goldstein/Backwards) Line Search Parameters
	  double n_val = n_max;
	  double m_val = 0;

	  BlockVector<double> structure_previous_iterate = solution;

	  if (fem_properties.optimization_method.compare("DN")==0) { 
	    if (DN.update(time, initialized_timestep_number)) break;
	  } else {  
	    if (AG_line_search || backwards_line_search) { // line search
	      pcout << "Line search. " << std::endl;

              // Need a reasonable first guess at the solution
              // Otherwise, previous diverging may cause issues
              solution = pre_linesearch_solution;

	      stress *= 0;
	      transfer_interface_dofs(stress_star, stress, 0, 0);

	      if (fem_properties.optimization_method.compare("Gradient")==0) {
		stress.block(0)*=(1-alpha_j);
		stress.block(0).add(float(alpha_j)/fem_properties.penalty_epsilon, update_direction.block(0));
	      } else {
		stress.block(0).add(alpha_j, update_direction.block(0));
	      }
	      transfer_interface_dofs(stress, stress, 0, 1, Displacement);

	      // Solve both fluid and structure simultaneously
	      // Threads::Task<> s_solver = Threads::new_task(&FSIProblem<dim>::structure_state_solve,*this, initialized_timestep_number);
	      // s_solver.join();

	      // if (physical_properties.moving_domain) {
	      //   //
	      //   // Laplace solve for domain update
	      //   // 
	      //   assemble_ale(state,true);
	      //   dirichlet_boundaries((System)2,state);
	      //   state_solver[2].factorize(system_matrix.block(2,2));
	      //   solve(state_solver[2],2,state);
	      //   transfer_all_dofs(solution,mesh_displacement_star,2,0);

	      //   if (physical_properties.simulation_type==2)
	      //     {
	      //       // Overwrites the Laplace solve since the velocities compared against will not be correct
	      //       ale_boundary_values.set_time(time);
	      //       VectorTools::project(ale_dof_handler, ale_constraints, QGauss<dim>(fem_properties.fluid_degree+2),
	      //   			 ale_boundary_values,
	      //   			 mesh_displacement_star.block(2)); // move directly to fluid block 
	      //       transfer_all_dofs(mesh_displacement_star,mesh_displacement_star,2,0);
	      //     }
	      //   mesh_displacement_star_old.block(0) = mesh_displacement_star.block(0); // Not currently implemented, but will allow for half steps

	      //   if (fem_properties.time_dependent) {
	      //     //
	      //     // Laplace solve for domain velocity update
	      //     // 
	      //     assemble_ale(state,true);
	      //     dirichlet_boundaries((System)2,state,true); // 3rd argument, special_case=true
	      //     state_solver[2].factorize(system_matrix.block(2,2));
	      //     solve(state_solver[2],2,state);
	      //     transfer_all_dofs(solution,mesh_velocity,2,0);

	      //     // mesh_velocity.block(0)=mesh_displacement_star.block(0);
	      //     // mesh_velocity.block(0)-=old_mesh_displacement.block(0);
	      //     // mesh_velocity.block(0)*=1./time_step;
	      //   }
	      // }

	      // Solve both fluid and structure simultaneously
	      Threads::Task<> s_solver = Threads::new_task(&FSIProblem<dim>::structure_state_solve,*this, initialized_timestep_number);
	      Threads::Task<int> f_solver = Threads::new_task(&FSIProblem<dim>::fluid_state_solve,*this, initialized_timestep_number);

              // return_value() implicitly calls join() on the Task
              // A return value of 1 indicates that the solver succeeded.
              if (f_solver.return_value() != 1) {
	        if (AG_line_search || backwards_line_search) {
                  alpha_j *= 0.5;
                  continue; // sends back to the reduction of alpha_j
                } else {
                  AssertThrow (false, ExcNotImplemented());// Bad first guess at beginning of time step.
                }
              }
                
	      s_solver.join();
	      // f_solver.join();
	    } else {
	      // The method for using alpha is significantly different for Gradient vs other optimization methods
	      if (fem_properties.optimization_method.compare("Gradient")==0) {
                // Do not adjust previous alpha value
		// alpha_j = fem_properties.steepest_descent_alpha;
	      } else {
		alpha_j = 1.0;
	      }

	      ++count;
	      if (count == 1 && fem_properties.true_control)
		{
		  fluid_boundary_stress.set_time(time);
		  VectorTools::project(fluid_dof_handler, fluid_constraints, QGauss<dim>(fem_properties.fluid_degree+2),
				       fluid_boundary_stress,
				       stress.block(0));
		  transfer_interface_dofs(stress,stress,0,1);
		  stress_star = stress;
		}

	      // RHS and Neumann conditions are inside these functions
	      // Solve for the state variables
	      timer.enter_subsection ("Assemble"); 
	      if (physical_properties.moving_domain)
		{
		  assemble_ale(state,true);
		  dirichlet_boundaries((System)2,state);
		  state_solver[2].factorize(system_matrix.block(2,2));
		  solve(state_solver[2],2,state);
		  transfer_all_dofs(solution,mesh_displacement_star,2,0);

		  if (physical_properties.simulation_type==2)
		    {
		      // Overwrites the Laplace solve since the velocities compared against will not be correct
		      ale_boundary_values.set_time(time);
		      VectorTools::project(ale_dof_handler, ale_constraints, QGauss<dim>(fem_properties.fluid_degree+2),
					   ale_boundary_values,
					   mesh_displacement_star.block(2)); // move directly to fluid block 
		      transfer_all_dofs(mesh_displacement_star,mesh_displacement_star,2,0);
		    }
		  mesh_displacement_star_old.block(0) = mesh_displacement_star.block(0); // Not currently implemented, but will allow for half steps

		  if (fem_properties.time_dependent) {
		    //
		    // Laplace solve for domain velocity update
		    // 
		    assemble_ale(state,true);
		    dirichlet_boundaries((System)2,state,true); // 3rd argument, special_case=true
		    state_solver[2].factorize(system_matrix.block(2,2));
		    solve(state_solver[2],2,state);
		    transfer_all_dofs(solution,mesh_velocity,2,0);

		    // mesh_velocity.block(0)=mesh_displacement_star.block(0);
		    // mesh_velocity.block(0)-=old_mesh_displacement.block(0);
		    // mesh_velocity.block(0)*=1./time_step;
		  }
		}

	      // Threads::Task<> s_assembly = Threads::new_task(&FSIProblem<dim>::assemble_structure,*this,state,true);
	  
	      // s_assembly.join();
	      // dirichlet_boundaries((System)1,state);
	      // Threads::Task<void> s_factor = Threads::new_task(&SparseDirectUMFPACK::factorize<SparseMatrix<double> >, state_solver[1], system_matrix.block(1,1));
	      // s_factor.join();
	      // Threads::Task<void> s_solve = Threads::new_task(&FSIProblem<dim>::solve, *this, state_solver[1], 1, state);				
	      // s_solve.join();

	      // pcout << "Norm of structure: " << system_matrix.block(0,0).frobenius_norm() << std::endl;  
	      // As timestep decreases, this makes it increasing difficult to get within some tolerance on the interface error
	      // This really only becomes noticeable using the first order finite difference in the objective

	      timer.leave_subsection();

	      // Solve both fluid and structure simultaneously
	      Threads::Task<> s_solver = Threads::new_task(&FSIProblem<dim>::structure_state_solve,*this, initialized_timestep_number);
	      Threads::Task<int> f_solver = Threads::new_task(&FSIProblem<dim>::fluid_state_solve,*this, initialized_timestep_number);

              // return_value() implicitly calls join() on the Task
              // A return value of 1 indicates that the solver succeeded.
              if (f_solver.return_value() != 1) {
	        if (AG_line_search || backwards_line_search) {
                  alpha_j *= 0.5;
                  continue; // sends back to the reduction of alpha_j
                } else {
                  AssertThrow (false, ExcNotImplemented());// Bad first guess at beginning of time step.
                }
              }
                
	      s_solver.join();
	      // f_solver.join();

              pre_linesearch_solution = solution;
	    } // Moving Domain, AG-Line Search or Not


	  } // end of not DN case

	  build_adjoint_rhs();

	  if (fem_properties.optimization_method.compare("DN")==0) {
	    // This case breaks the loop at the .update call at the beginning of the optimization loop
	  } else {
	    if (AG_line_search || backwards_line_search) {
	      // *****************************************************************************************
	      //                                  (Armijo-Goldstein) Line Search Check
	      // *****************************************************************************************
	      double velocity_with_update = interface_error();
	      pcout << "Original velocity jump: " << velocity_jump << std::endl;
	      pcout << "Updated  velocity jump: " << velocity_with_update << std::endl;
	      pcout << "Difference: " << velocity_jump - velocity_with_update << std::endl;
	      if (AG_line_search) {
		pcout << "Criteria: " << std::abs(alpha_j) * t_val << std::endl;
		if ((velocity_jump - velocity_with_update) >= std::abs(alpha_j) * t_val) AG_line_search = false;
		else alpha_j *= .5;
	      } else {
		if ((velocity_jump - velocity_with_update) >= 0) backwards_line_search = false;
		else alpha_j *= .5;
	      }
	    
	      pcout << "alpha_j: " << alpha_j << std::endl;

	    } else {

	      // *****************************************************************************************
	      //                              STOPPING CRITERIA CALCULATION
	      // *****************************************************************************************
	      velocity_jump_old = velocity_jump;

	      if (count==1) tau_t = fem_properties.cg_tolerance * std::sqrt(velocity_jump);

	      if (fem_properties.optimization_method.compare("DN")!=0) {
		velocity_jump=interface_error();
	      } else {

	      } 
	      if (count%1==0) pcout << "Jump Error: " << velocity_jump << std::endl;
	      if (count >= fem_properties.max_optimization_iterations || velocity_jump < fem_properties.jump_tolerance) break;
  	  
	      // *****************************************************************************************
	      //                                  STEEPEST DESCENT
	      // *****************************************************************************************
	      if (fem_properties.optimization_method.compare("Gradient")==0) {
		LinearMap::Linearized_Operator<dim> A(this);
		LinearMap::NeumannVector<dim> x(rhs_for_adjoint.block(0), this);
		x*=0;
		LinearMap::NeumannVector<dim> b(rhs_for_adjoint.block(0), this);
		A.initialize_matrix(x, b, adjoint);
		A.vmult(x,b);

		total_solves += 1;
	    	
		if (AG_line_search) {
		  m_val = interface_inner_product(rhs_for_adjoint.block(0),x);//solver_control.last_value();
		  pcout << "m_val : " << m_val << std::endl;
		  double c_val = 0.5;
		  if (count == 1)
		    t_val = -m_val*c_val;
		}

		// t_val = 0; // <-- This would turn off any criteria for line search except for nonnegative


		// // // if (velocity_jump>velocity_jump_old)
		// // //   {
		// // //     ++imprecord;
		// // //     //pcout << "Bad Move." << std::endl;
		// // //     consecutiverelrecord = 0;
		// // //   }
		// // // else if ((velocity_jump/velocity_jump_old)>=0.995) 
		// // //   {
		// // //     ++relrecord;
		// // //     ++consecutiverelrecord;
		// // //     //pcout << "Rel. Bad Move." << std::endl;
		// // //     //pcout << consecutiverelrecord << std::endl;
		// // //   }
		// // // else
		// // //   {
		// // //     imprecord = 0;
		// // //     relrecord = 0;
		// // //     alpha *= 1.01;
		// // //     //pcout << "Good Move." << std::endl;
		// // //     consecutiverelrecord = 0;
		// // //   }

		// // // if (relrecord > 1) 
		// // //   {
		// // //     alpha *= 1.01;
		// // //     relrecord = 0;
		// // //   }
		// // // else if (imprecord > 0)
		// // //   {
		// // //     alpha *= 0.95;
		// // //     imprecord = 0;
		// // //   }
	    
		// // // if (consecutiverelrecord>50)
		// // //   {
		// // //     pcout << "Break!" << std::endl;
		// // //     //break;
		// // //   }

		stress_star = stress;
		update_direction.block(0) = x;
		//x *= -1;

		stress.block(0) *= 1-alpha_j;
		stress.block(0).add(alpha_j, update_direction.block(0));
		transfer_interface_dofs(stress, stress, 0, 1, Displacement);
		// Update the stress using the adjoint variables
		// stress.block(0)*=(1-alpha);

		// // not negated since tmp has reverse of proper negation
		// double multiplier = float(alpha)/fem_properties.penalty_epsilon;
            
		// stress.block(0).add(multiplier,x);

		// tmp *= 0;
		// transfer_interface_dofs(stress,tmp,0,0);
		// stress *= 0;
		// transfer_interface_dofs(tmp,stress,0,0);

		// transfer_interface_dofs(stress,stress,0,1,Displacement);
	      }
	      // *****************************************************************************************
	      //                                  CG
	      // *****************************************************************************************
	      else if (fem_properties.optimization_method.compare("CG")==0) {
		LinearMap::Linearized_Operator<dim> A(this);
		LinearMap::NeumannVector<dim> output_vector(rhs_for_adjoint.block(0), this);
		for (Vector<double>::iterator it=output_vector.begin(); it!=output_vector.end(); ++it) *it = std::max(physical_properties.rho_f,physical_properties.rho_s) * rhs_for_adjoint.block(0).l2_norm()*1e10;
		LinearMap::NeumannVector<dim> input_vector(rhs_for_adjoint.block(0), this);
		//input_vector *= -1;
		//A.vmult(output_vector, input_vector);
		//tmp.block(0).add(-1.0, output_vector);
		// total_solves is passed by reference and updated
		unsigned int convergence_flag = 1;
		//ReductionControl solver_control(1000, 1e-50, fem_properties.cg_tolerance, false, false);
		SolverControl solver_control(1000, 1e-50, false, false);
		//GrowingVectorMemory<Vector<double> > mem;
		PrimitiveVectorMemory<Vector<double> > mem;
		SolverCG<Vector<double> > solver (solver_control);//, mem, SolverCG<Vector<double> >::AdditionalData(false /*exact residual */, -1.e-250 /* breakdown */));
		A.initialize_matrix(output_vector, input_vector, linear);
		try {
		  solver.solve(A, output_vector, input_vector, PreconditionIdentity());
		} catch (std::exception &e) {
		  AssertThrow (false, ExcMessage(e.what()));
		}
	      
		pcout << "last val: " << solver_control.last_value() << std::endl;
		pcout << "last step:" << solver_control.last_step() << std::endl;
		//pcout << input_vector << std::endl; 
		stress.block(0).add(1.0, output_vector);
		tmp=0;
		transfer_interface_dofs(stress,tmp,0,0);
		transfer_interface_dofs(stress,tmp,1,1,Displacement);
		stress=0;
		transfer_interface_dofs(tmp,stress,0,0);
		transfer_interface_dofs(tmp,stress,1,1,Displacement);

		transfer_interface_dofs(stress,stress,0,1,Displacement);

		// LinearMap::Linearized_Operator<dim> A(this);
		// LinearMap::NeumannVector<dim> output_vector(rhs_for_adjoint.block(0), this);
		// for (Vector<double>::iterator it=output_vector.begin(); it!=output_vector.end(); ++it) *it = std::max(physical_properties.rho_f,physical_properties.rho_s) * rhs_for_adjoint.block(0).l2_norm();
		// LinearMap::NeumannVector<dim> input_vector(rhs_for_adjoint.block(0), this);
		// //input_vector *= -1;
		// //A.vmult(output_vector, input_vector);
		// //tmp.block(0).add(-1.0, output_vector);
		// // total_solves is passed by reference and updated
		// unsigned int convergence_flag = 1;
		// ReductionControl solver_control(1000, 1e-50, fem_properties.cg_tolerance, false, false);
		// //SolverControl solver_control(1000, 1e-50, false, false);
		// PrimitiveVectorMemory<Vector<double> > mem;
		// SolverCG<Vector<double> > solver (solver_control, mem);//, SolverGMRES<Vector<double> >::AdditionalData(53,false));
		// A.initialize_matrix(output_vector, input_vector, linear);
		// try {
		//   solver.solve(A, output_vector, input_vector, PreconditionIdentity());
		// } catch (std::exception &e) {
		//   Assert (false, ExcMessage(e.what()));
		// }
	      
		// pcout << "last val: " << solver_control.last_value() << std::endl;
		// pcout << "last step:" << solver_control.last_step() << std::endl;
		// //pcout << input_vector << std::endl; 
		// stress.block(0).add(1.0, output_vector);
		// tmp=0;
		// transfer_interface_dofs(stress,tmp,0,0);
		// transfer_interface_dofs(stress,tmp,1,1,Displacement);
		// stress=0;
		// transfer_interface_dofs(tmp,stress,0,0);
		// transfer_interface_dofs(tmp,stress,1,1,Displacement);

		// transfer_interface_dofs(stress,stress,0,1,Displacement);
		// //total_solves = optimization_CG(total_solves, initialized_timestep_number);
	      }
	      // *****************************************************************************************
	      //                                  BICG
	      // *****************************************************************************************
	      else if (fem_properties.optimization_method.compare("BICG")==0) {
		// This version of deal.II's BICG can not be used since it runs  .vmult in parallel,
		// while  vmult makes changes to the RHS (resulting in unpredictable, and generally bad behavior)
 
		// if (velocity_jump > velocity_jump_old) {
		// 	stress = stress_star;
		// 	update_alpha *= 0.5;
		// 	velocity_jump = velocity_jump_last_good_value;
		// } else {
		// 	stress_star = stress;
		// 	update_alpha  = 1.0;
		// }

		// LinearMap::Linearized_Operator<dim> A(this);
		// LinearMap::NeumannVector<dim> output_vector(rhs_for_adjoint.block(0), this);
		// output_vector *= 0;
		// // for (Vector<double>::iterator it=output_vector.begin(); it!=output_vector.end(); ++it) *it = std::max(physical_properties.rho_f,physical_properties.rho_s) * rhs_for_adjoint.block(0).l2_norm();
		// LinearMap::NeumannVector<dim> input_vector(rhs_for_adjoint.block(0), this);
		// //input_vector *= -1;
		// //A.vmult(output_vector, input_vector);
		// //tmp.block(0).add(-1.0, output_vector);
		// // total_solves is passed by reference and updated
		// unsigned int convergence_flag = 1;
		// //ReductionControl solver_control(1000, 1e-50, fem_properties.cg_tolerance, false, false);
		// SolverControl solver_control(1000, 1e-50, false, false);
		// //GrowingVectorMemory<Vector<double> > mem;
		// PrimitiveVectorMemory<Vector<double> > mem;
		// SolverBicgstab<Vector<double> > solver (solver_control);//, mem, SolverBicgstab<Vector<double> >::AdditionalData(false /*exact residual */, 1.e-250 /* breakdown */));
		// A.initialize_matrix(output_vector, input_vector, linear);
		// try {
		//   solver.solve(A, output_vector, input_vector, PreconditionIdentity());
		// } catch (std::exception &e) {
		//   pcout << "Minimize failed." << std::endl;
		//   Assert (false, ExcMessage(e.what()));
		// }

		// pcout << "last val: " << solver_control.last_value() << std::endl;
		// pcout << "last step:" << solver_control.last_step() << std::endl;
		// //pcout << input_vector << std::endl; 

		// stress_star = stress;
		// update_direction.block(0) = output_vector;
		// AG_line_search = true;

		// stress.block(0).add(1.0, output_vector);
		// tmp=0;
		// transfer_interface_dofs(stress,tmp,0,0);
		// transfer_interface_dofs(stress,tmp,1,1,Displacement);
		// stress=0;
		// transfer_interface_dofs(tmp,stress,0,0);
		// transfer_interface_dofs(tmp,stress,1,1,Displacement);

		// transfer_interface_dofs(stress,stress,0,1,Displacement);



		// total_solves is passed by reference and updated
		unsigned int convergence_flag = 1;
		while (convergence_flag!=0) {
		  convergence_flag = optimization_BICGSTAB(total_solves, initialized_timestep_number, true, 1000, 1.0);
		}
	      }
	      // *****************************************************************************************
	      //                                  GMRES
	      // *****************************************************************************************
	      else if (fem_properties.optimization_method.compare("GMRES")==0) {
		// if (count == 1) update_alpha = 0.01;
		// if (count >= 2 && count <= 6) update_alpha += .195;
		// else update_alpha = 1.0;
		// if (velocity_jump > velocity_jump_old) {
		// 	//stress = stress_star;
		// 	update_alpha *= 0.6;
		// 	//velocity_jump = velocity_jump_last_good_value;
		// } else {
		// 	stress_star = stress;
		// 	update_alpha  *= 1.05;
		// 	//velocity_jump_last_good_value = velocity_jump;
		// }
		double gamma = 0.9;
		// n^A_n = gamma * norm(F(x_n))^2 / norm(F(x_{n-1}))^2
	      
		double n_n_A = gamma * velocity_jump / velocity_jump_old;
		double n_n_C;
		if (count == 1) n_n_C = n_max;
		else n_n_C = std::min(n_max, std::max(n_n_A, gamma*std::pow(n_val,2)));
		n_val = std::min(n_max, std::max(n_n_C, .5*tau_t/std::sqrt(velocity_jump)));

		LinearMap::Linearized_Operator<dim> A(this);
		LinearMap::NeumannVector<dim> output_vector(rhs_for_adjoint.block(0), this);
		for (Vector<double>::iterator it=output_vector.begin(); it!=output_vector.end(); ++it) *it = std::max(physical_properties.rho_f,physical_properties.rho_s) * rhs_for_adjoint.block(0).l2_norm();
		LinearMap::NeumannVector<dim> input_vector(rhs_for_adjoint.block(0), this);
		//input_vector *= -1;
		//A.vmult(output_vector, input_vector);
		//tmp.block(0).add(-1.0, output_vector);
		// total_solves is passed by reference and updated
		unsigned int convergence_flag = 1;
		//ReductionControl solver_control(1000, 1e-50, n_val, false, false);
		ReductionControl solver_control(1000, 1e-50, fem_properties.cg_tolerance, false, false);
		//SolverControl solver_control(1000, 1e-50, false, false);
		PrimitiveVectorMemory<Vector<double> > mem;
		SolverGMRES<Vector<double> > solver (solver_control, mem, SolverGMRES<Vector<double> >::AdditionalData(53,false));
		A.initialize_matrix(output_vector, input_vector, linear);
		try {
		  solver.solve(A, output_vector, input_vector, PreconditionIdentity());
		} catch (std::exception &e) {
		  AssertThrow (false, ExcMessage(e.what()));
		}
	      
		m_val = -solver_control.last_value();
		double c_val = 0.5;
		t_val = -m_val*c_val;
		//m_val = A.vmult(output_vector);

		pcout << "last val: " << solver_control.last_value() << std::endl;
		pcout << "last step:" << solver_control.last_step() << std::endl;
		//pcout << input_vector << std::endl; 
		stress_star = stress;
		update_direction.block(0) = output_vector;

		stress.block(0).add(alpha_j, update_direction.block(0));
		transfer_interface_dofs(stress, stress, 0, 1, Displacement);

		// stress.block(0).add(update_alpha, output_vector);
		// tmp=0;
		// transfer_interface_dofs(stress,tmp,0,0);
		// transfer_interface_dofs(stress,tmp,1,1,Displacement);
		// stress=0;
		// transfer_interface_dofs(tmp,stress,0,0);
		// transfer_interface_dofs(tmp,stress,1,1,Displacement);

		// transfer_interface_dofs(stress,stress,0,1,Displacement);
		// while (convergence_flag!=0) {
		// 	convergence_flag = optimization_GMRES(total_solves, initialized_timestep_number, true, 1);
		// }
	      }
	      AG_line_search = (fem_properties.line_search_method.compare("AG")==0);
	      backwards_line_search = (fem_properties.line_search_method.compare("BACKWARDS")==0);     
	    }
	  }
  	}
      // *****************************************************************************************
      //                                  UPDATE OLD SOLUTIONS
      // *****************************************************************************************
      pcout << "Total Solves: " << total_solves << std::endl;
      if (fem_properties.make_plots) output_results ();
      //if (fem_properties.richardson) 
      //	{
  	  old_old_solution = old_solution;
	  //	}
      old_solution = solution;
      old_stress = stress;

      // *****************************************************************************************
      //                                SAVE CALCULATED VARIABLES
      // *****************************************************************************************
      if (fem_properties.print_error) compute_error();
      pcout << "Comp. Time: " << t.elapsed() << std::endl;
      total_time += t.elapsed();
      pcout << "Est. Rem.: " << (fem_properties.T-time)/time_step*total_time/timestep_number << std::endl;
      t.restart();
      if (physical_properties.simulation_type==1) {
	if (timestep_number%(unsigned int)(std::ceil((double)total_timesteps/100))==0)
	  {
	    dealii::Functions::FEFieldFunction<dim> fe_function (structure_dof_handler, solution.block(1));
	    Point<dim> p1(1.5,1);
	    Point<dim> p2(3,1);
	    Point<dim> p3(4.5,1);
	    structure_file_out << time << " " << fe_function.value(p1,1) << " " << fe_function.value(p2,1) << " " << fe_function.value(p3,1) << std::endl; 
	  }
      } else if (physical_properties.simulation_type==3) {
	  // STRUCTURE OUTPUT
	  dealii::Functions::FEFieldFunction<dim> fe_function (structure_dof_handler, solution.block(1));
	  Point<dim> p1(0.6,0.2);
	  x_displacement[timestep_number] = fe_function.value(p1,0);
	  y_displacement[timestep_number] = fe_function.value(p1,1);
	  // double x_displ = fe_function.value(p1,0);
	  // double y_displ = fe_function.value(p1,1);
	  // structure_file_out << time << " " << x_displ << " " << y_displ << std::endl; 
	  // if (x_displ < displacement_min_max_mean[0][0]) displacement_min_max_mean[0][0]=x_displ;
	  // if (x_displ > displacement_min_max_mean[0][1]) displacement_min_max_mean[0][1]=x_displ;
	  // displacement_min_max_mean[0][2] += x_displ;
	  // if (y_displ < displacement_min_max_mean[1][0]) displacement_min_max_mean[1][0]=y_displ;
	  // if (y_displ > displacement_min_max_mean[1][1]) displacement_min_max_mean[1][1]=y_displ;
	  // displacement_min_max_mean[1][2] += y_displ;
	  // last_displacements[0] = x_displ;
	  // last_displacements[1] = y_displ;

	  // FLUID OUTPUT
	  // sigma_f * n = sigma_s * n where n is A unit normal vector to the interface
	  // To get \int sigma_f * n for lift and drag, we break the integral up into 
	  // \int sigma_f * n = \int sigma_f * n_f = \int_fluid sigma_f * n_f + \int_structure_sigma_s * n_f
	  //     = \int_fluid sigma_f * n_f - \int_structure sigma_s * n_s
	  // In the following case, we actually use the opposite (but the result can be multiplied by -1 to get the result of using the other unit normal)
	  ale_transform_fluid();
	  Tensor<1,dim> lift_drag = -lift_and_drag_fluid();
	  ref_transform_fluid();
	  lift_drag += lift_and_drag_structure();
	  drag[timestep_number] = lift_drag[0];
	  lift[timestep_number] = lift_drag[1];
	  pcout << time << " drag: " << lift_drag[0] << " lift: " << lift_drag[1] << std::endl;
	  // drag_min_max[0] = std::min(drag_min_max[0],lift_drag[0]);
	  // drag_min_max[1] = std::max(drag_min_max[1],lift_drag[0]);
	  // lift_min_max[0] = std::min(lift_min_max[0],lift_drag[1]);
	  // lift_min_max[1] = std::max(lift_min_max[1],lift_drag[1]);
	  // last_lift_drag[0] = lift_drag[0];
	  // last_lift_drag[1] = lift_drag[1];
      } else if (physical_properties.simulation_type==4) {
	dealii::Functions::FEFieldFunction<dim> fe_function (structure_dof_handler, solution.block(1));
	//Point<dim> p1(-0.11126, -0.487464, 2.5); Point used in Burman paper
	Point<dim> p1(-0.11126, -0.487464, 2.5);
	y_displacement[timestep_number] = fe_function.value(p1,1);
	outflow_rate[timestep_number] = flowrate_through_surface(2);
	structure_file_out << time << " " << y_displacement[timestep_number] << " " << outflow_rate[timestep_number] << std::endl; 
      }
      // Write these vectors to the hard drive 100 times
      if (timestep_number%(unsigned int)(std::ceil((double)total_timesteps/100))==0) {
	  // This could be made more robust in the future by copying data before saving the new data
	  // However, this means the job would have to fail in the middle of writing which seems unlikely
	  std::ofstream recent_iteration_num ("recent.data");
	  recent_iteration_num << "-1"; 
	  // Initially indicate that the time step information is corrupt.
	  // Once all data for the time step has been saved, this will be corrected to the appropriate time step
	  recent_iteration_num.close();
	  // Here is the form for if we would want to save every n time steps
	  // const std::string solution_filename = "solution." + std::to_string(i) + ".data";

	  // First, write solution, stress, and mesh variables to file
	  const std::string solution_filename = "solution.data";
	  std::ofstream solution_output (solution_filename.c_str());
	  //solution_output << std::setprecision(20);
	  solution.block_write(solution_output);
	  solution_output.close();
	  const std::string stress_filename = "stress.data";
	  std::ofstream stress_output (stress_filename.c_str());
	  //stress_output << std::setprecision(20);
	  stress.block_write(stress_output);
	  stress_output.close();
	  const std::string mesh_filename = "mesh.data";
	  std::ofstream mesh_output (mesh_filename.c_str());
	  //mesh_output << std::setprecision(20);
	  mesh_displacement_star.block_write(mesh_output);
	  mesh_output.close();

	  // Second, write displacements, lift, and drag to file
	  const std::string lift_filename = "lift.data";
	  std::ofstream lift_output (lift_filename.c_str());
	  lift.block_write(lift_output);
	  lift_output.close();
	  const std::string drag_filename = "drag.data";
	  std::ofstream drag_output (drag_filename.c_str());
	  drag.block_write(drag_output);
	  drag_output.close();	  
	  const std::string x_displacement_filename = "x_displacement.data";
	  std::ofstream x_displacement_output (x_displacement_filename.c_str());
	  x_displacement.block_write(x_displacement_output);
	  x_displacement_output.close();	  
	  const std::string y_displacement_filename = "y_displacement.data";
	  std::ofstream y_displacement_output (y_displacement_filename.c_str());
	  y_displacement.block_write(y_displacement_output);
	  y_displacement_output.close();
	  const std::string outflow_rate_filename = "outflow_rate.data";
	  std::ofstream outflow_rate_output (outflow_rate_filename.c_str());
	  outflow_rate.block_write(outflow_rate_output);
	  outflow_rate_output.close();	  

	  const std::string temp_output_filename = Utilities::int_to_string (timestep_number, 5) + ".log";
	  const std::string last_output_filename = "last.log";
	  std::ofstream temp_qoi_output (temp_output_filename.c_str());
	  std::ofstream last_qoi_output (last_output_filename.c_str());
	  for (unsigned int i=0; i<timestep_number; ++i) {
	    temp_qoi_output << fem_properties.t0 + (i+1)*time_step << " " << x_displacement[i] << " " << y_displacement[i] << " " << lift[i] << " " << drag[i] << " " << outflow_rate[i] << std::endl;
	    last_qoi_output << fem_properties.t0 + (i+1)*time_step << " " << x_displacement[i] << " " << y_displacement[i] << " " << lift[i] << " " << drag[i] << " " << outflow_rate[i] << std::endl;
	  }
	  temp_qoi_output.close();
	  last_qoi_output.close();

	  // Last, confirm that all things have been written to file
	  std::ofstream recent_iteration_num_rewrite ("recent.data");
	  recent_iteration_num_rewrite << (timestep_number+1);
	  recent_iteration_num_rewrite.close();
      }
    }
  timer.leave_subsection ();
  if (physical_properties.simulation_type==3) {
    double x_displacement_max=0, x_displacement_min=0;
    double y_displacement_max=0, y_displacement_min=0;
    double lift_max=0, lift_min=0, drag_max=0, drag_min=0;
    for (unsigned int i=0; i<total_timesteps; ++i) {
      x_displacement_max = std::max(x_displacement_max, x_displacement[i]);
      x_displacement_min = std::min(x_displacement_min, x_displacement[i]);
      y_displacement_max = std::max(y_displacement_max, y_displacement[i]);
      y_displacement_min = std::min(y_displacement_min, y_displacement[i]);
      lift_max = std::max(lift_max, lift[i]);
      lift_min = std::min(lift_min, lift[i]);
      drag_max = std::max(drag_max, drag[i]);
      drag_min = std::min(drag_min, drag[i]);
      // pcout << i << ". x: " << x_displacement[i] << " y: " << y_displacement[i] << std::endl;
    }
    // for (unsigned int i=0; i<total_timesteps; ++i) {
    //   // pcout << i << ". lift: " << lift[i] << " drag: " << drag[i] << std::endl;
    // }

    const std::string output_filename = "output.data";
    std::ofstream qoi_output (output_filename.c_str());
    for (unsigned int i=0; i<total_timesteps; ++i) {
      qoi_output << fem_properties.t0 + (i+1)*time_step << " " << x_displacement[i] << " " << y_displacement[i] << " " << lift[i] << " " << drag[i] << std::endl;
    }
    //lift.block_write(lift_output);
    qoi_output.close();
    
    pcout << "STRUCTURE: " << std::endl;
    pcout << "horizontal: " << .5*(x_displacement_max+x_displacement_min) << " +/- " << .5*(x_displacement_max-x_displacement_min) << ", vertical: " << .5*(y_displacement_max+y_displacement_min) << " +/- " << .5*(y_displacement_max-y_displacement_min) << std::endl;
    pcout << "last step: horizontal: " << x_displacement[total_timesteps-1] << ", vertical: " << y_displacement[total_timesteps-1] << std::endl;
    pcout << std::endl << "FLUID: " << std::endl;
    pcout << "drag: " << .5*(drag_max+drag_min) << " +/- " << .5*(drag_max-drag_min) << ", lift: " << .5*(lift_max+lift_min) << " +/- " << .5*(lift_max-lift_min) << std::endl;
    pcout << "last step: drag: " << drag[total_timesteps-1] << ", lift: " << lift[total_timesteps-1] << std::endl;
  } else if (physical_properties.simulation_type==4) {

    // THIS NEEDS FIXED TO KEEP TRACK OF THINGS!!!
    const std::string output_filename = "output.data";
    std::ofstream qoi_output (output_filename.c_str());
    for (unsigned int i=0; i<total_timesteps; ++i) {
      qoi_output << fem_properties.t0 + (i+1)*time_step << " " << y_displacement[i] << " " << outflow_rate[i] << std::endl;
    }
    //lift.block_write(lift_output);
    qoi_output.close();
  }
}


template void FSIProblem<2>::run ();
template void FSIProblem<3>::run ();
