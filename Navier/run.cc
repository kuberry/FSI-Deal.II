#include "FSI_Project.h"
#include <deal.II/base/timer.h>

template <int dim>
void FSIProblem<dim>::run ()
{
  unsigned int total_timesteps = (double)(fem_properties.T-fem_properties.t0)/time_step;

  std::ofstream structure_file_out;
  std::ofstream fluid_file_out;
  if (physical_properties.simulation_type==1 || physical_properties.simulation_type==3)
    {
      structure_file_out.open("structure_output.txt");
      fluid_file_out.open("fluid_output.txt");
    }
 
  TimerOutput timer (std::cout, TimerOutput::summary,
  		     TimerOutput::wall_times);
  timer.enter_subsection ("Everything");
  timer.enter_subsection ("Setup dof system");

  setup_system();
  Threads::Task<void>
   task = Threads::new_task (&FSIProblem<dim>::build_dof_mapping,*this);
  //build_dof_mapping();

  timer.leave_subsection();

  StructureBoundaryValues<dim> structure_boundary_values(physical_properties);
  FluidBoundaryValues<dim> fluid_boundary_values(physical_properties);
  AleBoundaryValues<dim> ale_boundary_values(physical_properties);

  StructureStressValues<dim> structure_boundary_stress(physical_properties);
  FluidStressValues<dim> fluid_boundary_stress(physical_properties);

  structure_boundary_values.set_time(fem_properties.t0-time_step);
  fluid_boundary_values.set_time(fem_properties.t0-time_step);

  VectorTools::project (fluid_dof_handler, fluid_constraints, QGauss<dim>(fem_properties.fluid_degree+2),
			fluid_boundary_values,
			old_old_solution.block(0));

  structure_boundary_values.set_time(fem_properties.t0);
  fluid_boundary_values.set_time(fem_properties.t0);

  structure_boundary_stress.set_time(fem_properties.t0);
  fluid_boundary_stress.set_time(fem_properties.t0);

  
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
  task.join();
  transfer_interface_dofs(old_stress,old_stress,0,1);
  stress=old_stress;
  // Note to self: On a moving domain, predotting stress tensor with unit normal requires extra work (pull backs)
  // If only retrieving the stress tensor, it can be dotted with the moving unit normal

  if (physical_properties.moving_domain)
    {
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
    }

  std::vector<std::vector<double> > displacement_min_max_mean(2); // spatial dimension is 2 
  std::vector<double> lift_min_max(dim); 
  std::vector<double> drag_min_max(dim); 
  std::vector<double> last_displacements(2);
  if (physical_properties.simulation_type==3) {
    std::vector<double> zero(3);
    displacement_min_max_mean[0] = zero;
    displacement_min_max_mean[1] = zero;
  }

  double total_time = 0;

  // direct_solver.initialize (system_matrix.block(block_num,block_num));
  for (timestep_number=1, time=fem_properties.t0+time_step;
       timestep_number<=total_timesteps;++timestep_number)
    {
      if (!fem_properties.time_dependent) {
	timestep_number=total_timesteps;
      }
      boost::timer t;
      time = fem_properties.t0 + timestep_number*time_step;
      std::cout << std::endl << "----------------------------------------" << std::endl;
      std::cout << "Time step " << timestep_number
  		<< " at t=" << time
  		<< std::endl;

      double velocity_jump = 1;
      double velocity_jump_old = 2;
      //unsigned int imprecord=0;
      //unsigned int relrecord=0;
      //unsigned int total_relrecord=0;

      unsigned int count = 0;
      update_domain = true;

      rhs_for_adjoint=1;

      double alpha = fem_properties.steepest_descent_alpha;
      unsigned int imprecord = 0;
      unsigned int relrecord = 0;
      unsigned int consecutiverelrecord = 0;

      //stress=old_stress;

      unsigned int total_solves = 0;


      if (physical_properties.moving_domain)
  	{
  	  old_mesh_displacement.block(0) = mesh_displacement_star.block(0);
  	}

      while (true)
        {
  	  ++count;
  	  if (count == 1 && fem_properties.true_control)
  	    {
  	      fluid_boundary_stress.set_time(time);
  	      VectorTools::project(fluid_dof_handler, fluid_constraints, QGauss<dim>(fem_properties.fluid_degree+2),
  				   fluid_boundary_stress,
  				   stress.block(0));
  	      transfer_interface_dofs(stress,stress,0,1);
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

  	      mesh_velocity.block(0)=mesh_displacement_star.block(0);
  	      mesh_velocity.block(0)-=old_mesh_displacement.block(0);
  	      mesh_velocity.block(0)*=1./time_step;
  	    }

  	  // Threads::Task<> s_assembly = Threads::new_task(&FSIProblem<dim>::assemble_structure,*this,state,true);
	  
  	  // s_assembly.join();
  	  // dirichlet_boundaries((System)1,state);
  	  // Threads::Task<void> s_factor = Threads::new_task(&SparseDirectUMFPACK::factorize<SparseMatrix<double> >, state_solver[1], system_matrix.block(1,1));
  	  // s_factor.join();
  	  // Threads::Task<void> s_solve = Threads::new_task(&FSIProblem<dim>::solve, *this, state_solver[1], 1, state);				
  	  // s_solve.join();

  	  // std::cout << "Norm of structure: " << system_matrix.block(0,0).frobenius_norm() << std::endl;  
  	  // As timestep decreases, this makes it increasing difficult to get within some tolerance on the interface error
  	  // This really only becomes noticeable using the first order finite difference in the objective

  	  timer.leave_subsection();



  	  // Get the first assembly started ahead of time
  	  //Threads::Task<> f_assembly = Threads::new_task(&FSIProblem<dim>::assemble_fluid,*this,state,true);

  	  // STRUCTURE SOLVER ITERATIONS
  	  std::cout <<"Before structure"<<std::endl;
  	  //solution_star.block(1)=1;
  	  solution_star.block(1) = solution.block(1); 
  	  do {
  	      solution_star.block(1)=solution.block(1);
  	      timer.enter_subsection ("Assemble");
  	      assemble_structure(state, true);
  	      timer.leave_subsection();
  	      dirichlet_boundaries((System)1,state);
  	      timer.enter_subsection ("State Solve"); 
  	      if (timestep_number==1)
  	      	{
  	      	  state_solver[1].initialize(system_matrix.block(1,1));
  	      	}
	      else 
		{
		  state_solver[1].factorize(system_matrix.block(1,1));
		}
  	      solve(state_solver[1],1,state);
  	      timer.leave_subsection ();
  	      solution_star.block(1)-=solution.block(1);
  	      ++total_solves;
  	      std::cout << solution_star.block(1).l2_norm() << std::endl;
  	  } while (solution_star.block(1).l2_norm()>1e-8);
  	  solution_star.block(1) = solution.block(1); 



  	  //f_assembly.join();
  	  // FLUID SOLVER ITERATIONS
  	  solution_star.block(0)=1;
	  bool newton = fem_properties.newton;
	  unsigned int picard_iterations = 7;
	  unsigned int loop_count = 0;
  	  while (solution_star.block(0).l2_norm()>1e-8)
  	    {
  	      solution_star.block(0)=solution.block(0);
  	      timer.enter_subsection ("Assemble");
	      if (loop_count < picard_iterations) fem_properties.newton = false; 
	      // Turn off Newton's method for a few picard iterations
	      assemble_fluid(state, true);
	      if (loop_count < picard_iterations) fem_properties.newton = newton;
	      timer.leave_subsection();

  	      dirichlet_boundaries((System)0,state);
  	      timer.enter_subsection ("State Solve"); 
  	      if (timestep_number==1) {
		state_solver[0].initialize(system_matrix.block(0,0));
	      } else {
		state_solver[0].factorize(system_matrix.block(0,0));
	      }
  	      solve(state_solver[0],0,state);
	      
	      // Pressure needs rescaled, since it was scaled/balanced against rho_f  in the operator
	      tmp = 0; tmp2 = 0;
	      transfer_all_dofs(solution, tmp, 0, 2);
	      transfer_all_dofs(tmp2, solution, 2, 0);
	      solution.block(0) *= physical_properties.rho_f;
	      transfer_all_dofs(tmp, solution, 2, 0);
	      // This is done by:
	      // copying out all except pressure
	      // copying in zeros over all but pressure
	      // scaling the pressure
	      // copying the other values back in

  	      timer.leave_subsection ();
  	      solution_star.block(0)-=solution.block(0);
  	      ++total_solves;
  	      if ((fem_properties.richardson && !fem_properties.newton) || !physical_properties.navier_stokes) {
		break;
	      } else {
		std::cout << solution_star.block(0).l2_norm() << std::endl;
	      }
	      
	      loop_count++;
  	    }
  	  solution_star.block(0) = solution.block(0); 




  	  build_adjoint_rhs();

  	  velocity_jump_old = velocity_jump;
  	  velocity_jump=interface_error();

  	  if (count%1==0) std::cout << "Jump Error: " << velocity_jump << std::endl;
  	  if (physical_properties.moving_domain && fem_properties.optimization_method.compare("Gradient")==0)
  	    {
  	      if (count >= fem_properties.max_optimization_iterations || velocity_jump < fem_properties.jump_tolerance)
  	  	{
  	  	  if (update_domain) break; // previous iteration had updated domain
  	  	  else update_domain = true;
  	  	}
  	      else
  	  	{
  	  	  if (count%3==0) update_domain = true;
  	  	  else update_domain = false;
  	  	}
  	    }
  	  else
  	    {
  	      if (count >= fem_properties.max_optimization_iterations || velocity_jump < fem_properties.jump_tolerance) break;
  	    }

  	  if (fem_properties.optimization_method.compare("Gradient")==0)
  	    {
	      
	      // assemble_fluid(adjoint, true);
	      // assemble_structure(adjoint, true);
  	      Threads::Task<void> f_assembly = Threads::new_task(&FSIProblem<dim>::assemble_fluid, *this, adjoint, true);
  	      Threads::Task<void> s_assembly = Threads::new_task(&FSIProblem<dim>::assemble_structure, *this, adjoint, true);
	      // dirichlet_boundaries((System)1, adjoint);
	      // dirichlet_boundaries((System)0, adjoint);

  	      s_assembly.join();
  	      dirichlet_boundaries((System)1, adjoint);
  	      f_assembly.join();
  	      dirichlet_boundaries((System)0, adjoint);

	      // if (timestep_number==1) {
	      // 	adjoint_solver[0].initialize(adjoint_matrix.block(0,0));
	      // 	adjoint_solver[1].initialize(adjoint_matrix.block(1,1));
	      // } else {
	      // 	adjoint_solver[0].factorize(adjoint_matrix.block(0,0));
	      // 	adjoint_solver[1].factorize(adjoint_matrix.block(1,1));
	      // }

	      if (timestep_number==1) {
	        adjoint_solver[0].initialize(adjoint_matrix.block(0,0));
	        adjoint_solver[1].initialize(adjoint_matrix.block(1,1));
		Threads::Task<void> s_solve = Threads::new_task(&FSIProblem<dim>::solve, *this, adjoint_solver[1], 1, adjoint);
		Threads::Task<void> f_solve = Threads::new_task(&FSIProblem<dim>::solve, *this, adjoint_solver[0], 0, adjoint);				
		f_solve.join();
		s_solve.join();
	      } else {
		Threads::Task<void> f_factor = Threads::new_task(&SparseDirectUMFPACK::factorize<SparseMatrix<double> >, adjoint_solver[0], adjoint_matrix.block(0,0));
		Threads::Task<void> s_factor = Threads::new_task(&SparseDirectUMFPACK::factorize<SparseMatrix<double> >, adjoint_solver[1], adjoint_matrix.block(1,1));
		s_factor.join();
		Threads::Task<void> s_solve = Threads::new_task(&FSIProblem<dim>::solve, *this, adjoint_solver[1], 1, adjoint);
		f_factor.join();
		Threads::Task<void> f_solve = Threads::new_task(&FSIProblem<dim>::solve, *this, adjoint_solver[0], 0, adjoint);				
		f_solve.join();
		s_solve.join();
	      }


	      // solve(adjoint_solver[1], 1, adjoint);
	      // solve(adjoint_solver[0], 0, adjoint);


  	      total_solves += 2;

  	      if (velocity_jump>velocity_jump_old)
  		{
  		  ++imprecord;
  		  //std::cout << "Bad Move." << std::endl;
  		  consecutiverelrecord = 0;
  		}
  	      else if ((velocity_jump/velocity_jump_old)>=0.995) 
  		{
  		  ++relrecord;
  		  ++consecutiverelrecord;
  		  //std::cout << "Rel. Bad Move." << std::endl;
  		  //std::cout << consecutiverelrecord << std::endl;
  		}
  	      else
  		{
  		  imprecord = 0;
  		  relrecord = 0;
  		  alpha *= 1.01;
  		  //std::cout << "Good Move." << std::endl;
  		  consecutiverelrecord = 0;
  		}

  	      if (relrecord > 1) 
  		{
  		  alpha *= 1.01;
  		  relrecord = 0;
  		}
  	      else if (imprecord > 0)
  	      	{
  	      	  alpha *= 0.95;
  	      	  imprecord = 0;
  	      	}
	    
  	      if (consecutiverelrecord>50)
  		{
  		  std::cout << "Break!" << std::endl;
  		  //break;
  		}

  	      // Update the stress using the adjoint variables
  	      stress.block(0)*=(1-alpha);
  	      tmp=0;

  	      if (fem_properties.adjoint_type==1)
  		{
  		  transfer_interface_dofs(adjoint_solution,tmp,1,0,Displacement);
  		  tmp.block(0)*= fem_properties.structure_theta;
  		}
  	      else
  		{
  		  transfer_interface_dofs(adjoint_solution,tmp,1,0,Velocity);
  		  tmp.block(0)*= fem_properties.structure_theta;
  		}

  	      tmp.block(0).add(-fem_properties.fluid_theta,adjoint_solution.block(0));

  	      //std::cout << "L2 " << tmp.block(0).l2_norm() << std::endl;

  	      // not negated since tmp has reverse of proper negation
  	      double multiplier = float(alpha)/fem_properties.penalty_epsilon;
            
  	      stress.block(0).add(multiplier,tmp.block(0));

  	      //std::cout << "STRESS: " << stress.block(0).l2_norm() << std::endl;

  	      tmp=0;
  	      transfer_interface_dofs(stress,tmp,0,0);
  	      stress=0;
  	      transfer_interface_dofs(tmp,stress,0,0);

  	      transfer_interface_dofs(stress,stress,0,1,Displacement);
  	      //if (count%50==0) std::cout << "alpha: " << alpha << std::endl;
  	    }
  	  else if (fem_properties.optimization_method.compare("CG")==0) total_solves = optimization_CG(total_solves);
	  else total_solves = optimization_BICGSTAB(total_solves);

  	}
      std::cout << "Total Solves: " << total_solves << std::endl;
      if (fem_properties.make_plots) output_results ();
      if (fem_properties.richardson) 
  	{
  	  old_old_solution = old_solution;
  	}
      old_solution = solution;
      old_stress = stress;
      if (fem_properties.print_error) compute_error();
      std::cout << "Comp. Time: " << t.elapsed() << std::endl;
      total_time += t.elapsed();
      std::cout << "Est. Rem.: " << (fem_properties.T-time)/time_step*total_time/timestep_number << std::endl;
      t.restart();
      if (physical_properties.simulation_type==1)
  	{
  	  if (timestep_number%(unsigned int)(std::ceil((double)total_timesteps/100))==0)
  	    {
  	      dealii::Functions::FEFieldFunction<dim> fe_function (structure_dof_handler, solution.block(1));
  	      Point<dim> p1(1.5,1);
  	      Point<dim> p2(3,1);
  	      Point<dim> p3(4.5,1);
  	      structure_file_out << time << " " << fe_function.value(p1,1) << " " << fe_function.value(p2,1) << " " << fe_function.value(p3,1) << std::endl; 
  	    }
  	}
      else if (physical_properties.simulation_type==3)
	{

	  // STRUCTURE OUTPUT
	  dealii::Functions::FEFieldFunction<dim> fe_function (structure_dof_handler, solution.block(1));
	  Point<dim> p1(0.6,0.2);
	  double x_displ = fe_function.value(p1,0);
	  double y_displ = fe_function.value(p1,1);
	  structure_file_out << time << " " << x_displ << " " << y_displ << std::endl; 
	  if (x_displ < displacement_min_max_mean[0][0]) displacement_min_max_mean[0][0]=x_displ;
	  if (x_displ > displacement_min_max_mean[0][1]) displacement_min_max_mean[0][1]=x_displ;
	  displacement_min_max_mean[0][2] += x_displ;
	  if (y_displ < displacement_min_max_mean[1][0]) displacement_min_max_mean[1][0]=y_displ;
	  if (y_displ > displacement_min_max_mean[1][1]) displacement_min_max_mean[1][1]=y_displ;
	  displacement_min_max_mean[1][2] += y_displ;
	  last_displacements[0] = x_displ;
	  last_displacements[1] = y_displ;

	  // FLUID OUTPUT
	  Tensor<1,dim> lift_drag = lift_and_drag_fluid();
	  //lift_drag += lift_and_drag_structure();
	  fluid_file_out << time << " " << lift_drag[0] << " " << lift_drag[1] << std::endl;
	  drag_min_max[0] = std::min(drag_min_max[0],lift_drag[0]);
	  drag_min_max[1] = std::max(drag_min_max[1],lift_drag[0]);
	  lift_min_max[0] = std::min(lift_min_max[0],lift_drag[1]);
	  lift_min_max[1] = std::max(lift_min_max[1],lift_drag[1]);
	}
    }
  timer.leave_subsection ();
  if (physical_properties.simulation_type==3) {
    std::cout << "STRUCTURE: " << std::endl;
    std::cout << "horizontal: " << .5*(displacement_min_max_mean[0][0]+displacement_min_max_mean[0][1]) << "+/-" << .5*(displacement_min_max_mean[0][1]-displacement_min_max_mean[0][0]) << ", vertical: " << .5*(displacement_min_max_mean[1][0]+displacement_min_max_mean[1][1]) << "+/-" << .5*(displacement_min_max_mean[1][1]-displacement_min_max_mean[1][0]) << std::endl;
    std::cout << "last step: horizontal: " << last_displacements[0] << ", vertical: " << last_displacements[1] << std::endl;
    std::cout << std::endl << "FLUID: " << std::endl;
    std::cout << "drag: " << .5*(drag_min_max[0]+drag_min_max[1]) << "+/-" << .5*(drag_min_max[1]-drag_min_max[0]) << ", lift: " << .5*(lift_min_max[0]+lift_min_max[1]) << "+/-" << .5*(lift_min_max[1]-lift_min_max[0]) << std::endl;
  }
}


template void FSIProblem<2>::run ();
