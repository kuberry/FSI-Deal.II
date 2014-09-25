#include "FSI_Project.h"
#include <deal.II/base/timer.h>

template <int dim>
void FSIProblem<dim>::run ()
{
  TimerOutput timer (std::cout, TimerOutput::summary,
		     TimerOutput::wall_times);
  timer.enter_subsection ("Everything");
  timer.enter_subsection ("Setup dof system");

  setup_system();
  Threads::Task<void>
    task = Threads::new_task (&FSIProblem<dim>::build_dof_mapping,*this);

  timer.leave_subsection();

  StructureBoundaryValues<dim> structure_boundary_values(physical_properties);
  FluidBoundaryValues<dim> fluid_boundary_values(physical_properties);

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

  VectorTools::project(fluid_dof_handler, fluid_constraints, QGauss<dim>(fem_properties.fluid_degree+2),
		       fluid_boundary_stress,
		       old_stress.block(0));

  task.join();
  transfer_interface_dofs(old_stress,old_stress,0,1);
  stress=old_stress;
  double total_time = 0;


	   
  // direct_solver.initialize (system_matrix.block(block_num,block_num));
  for (timestep_number=1, time=fem_properties.t0+time_step;
       timestep_number<=(double)(fem_properties.T-fem_properties.t0)/time_step;++timestep_number)
    {
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
      rhs_for_adjoint=1;

      double alpha = fem_properties.steepest_descent_alpha;
      unsigned int imprecord = 0;
      unsigned int relrecord = 0;
      unsigned int consecutiverelrecord = 0;

      //stress=old_stress;

      unsigned int total_solves = 0;

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
	  Threads::Task<> structure_assembly = Threads::new_task(&FSIProblem<dim>::assemble_structure,*this,state,true); 
	  Threads::Task<> fluid_assembly = Threads::new_task(&FSIProblem<dim>::assemble_fluid,*this,state,true); 
	  //assemble_ale(state, true);
	  timer.leave_subsection();
	  // This solving order will need changed later since the Dirichlet bcs for the ALE depend on the solution to the structure problem
	  
	  structure_assembly.join();
	  for (unsigned int i=1; i<2; ++i) // no ale solve currently
	    {
	      dirichlet_boundaries((System)i,state);
	      //dirichlet_boundaries((System)i,state);
	      timer.enter_subsection ("State Solve");
	      if (timestep_number==1)
	      	{
	      	  state_solver[i].initialize(system_matrix.block(i,i));
	      	}
	      // solver uses vmult which doesn't require factorization
	      solve(state_solver[i],i,state);
	      timer.leave_subsection ();
	    }

	  solution_star=1;
	  bool not_first_newton=false;
	  fluid_assembly.join();
	  while (solution_star.l2_norm()>1e-8)
	    {
	      solution_star=solution;
	      if (not_first_newton)
		{
		  timer.enter_subsection ("Assemble");
		  assemble_fluid(state, true);
		  timer.leave_subsection();
		}
	      dirichlet_boundaries((System)0,state);
	      timer.enter_subsection ("State Solve"); 
	      if (timestep_number==1)
	      	{
	      	  state_solver[0].initialize(system_matrix.block(0,0));
	      	}
	      state_solver[0].factorize(system_matrix.block(0,0));
	      solve(state_solver[0],0,state);
	      timer.leave_subsection ();
	      solution_star-=solution;
	      ++total_solves;
	      if ((fem_properties.richardson && !fem_properties.newton) || !physical_properties.navier_stokes)
	      	{
	      	  break;
	      	}
	      else
	      	{
	      	  std::cout << solution_star.l2_norm() << std::endl;
	      	}
	      not_first_newton=true;
	    }
	  solution_star = solution; 
	  build_adjoint_rhs();

	  velocity_jump_old = velocity_jump;
	  velocity_jump=interface_error();

	  if (count%1==0) std::cout << "Jump Error: " << velocity_jump << std::endl;
	  if (count >= fem_properties.max_optimization_iterations || velocity_jump < 1e-10) break; //pow(time_step,4)) break;

	  if (fem_properties.optimization_method.compare("Gradient")==0)
	    {
	      assemble_structure(adjoint, true);
	      assemble_fluid(adjoint, true);
	      assemble_ale(adjoint, true);
	      // Solve for the adjoint
	      for (unsigned int i=0; i<2; ++i)
		{
		  dirichlet_boundaries((System)i,adjoint);
		  adjoint_solver[i].factorize(adjoint_matrix.block(i,i));
		  solve(adjoint_solver[i], i, adjoint);
		}
	      ++total_solves;


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
		  tmp.block(0)*=1./time_step;
		}
	      else
		{
		  transfer_interface_dofs(adjoint_solution,tmp,1,0,Velocity);
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
	  else // fem_properties.optimization_method==CG
	    {
	      tmp=fem_properties.cg_tolerance;
	      //tmp=rhs_for_adjoint;
	      //tmp*=-1;
	      // x^0 = guess
	      // get adjoint variables 
	      // assemble_structure(adjoint);
	      // assemble_fluid(adjoint);
	      // for (unsigned int i=0; i<2; ++i)
	      // 	{
	      // 	  dirichlet_boundaries((System)i,adjoint);
	      // 	  solve(i,adjoint);
	      // 	}
	      // ++total_solves;
	      // tmp=0; tmp2=0;
	      // rhs_for_linear_p=0;
	      // transfer_interface_dofs(adjoint_solution,tmp,1,0);
	      // tmp.block(0)*=-1/time_step;
	      // transfer_interface_dofs(adjoint_solution,tmp2,0,0);
	      // tmp.block(0)+=tmp2.block(0);
	      // tmp.block(0).add(sqrt(fem_properties.penalty_epsilon),rhs_for_adjoint_s.block(0));
	      // transfer_interface_dofs(tmp,rhs_for_linear_p,0,0);
	      // transfer_interface_dofs(rhs_for_linear_p,rhs_for_linear_p,0,1);
	      // rhs_for_linear_p.block(1)*=-1;   // copy, negate
	      // rhs_for_linear_p*=-1;
	      // Generate a random vector
	      //for  (Vector<double>::iterator it=tmp.block(0).begin(); it!=tmp.block(0).end(); ++it)
	      // *it = ((double)std::rand() / (double)(RAND_MAX)) * fem_properties.cg_tolerance; //std::rand(0,10);
	      //std::cout << *it << std::endl;

	      rhs_for_linear_h=0;
	      transfer_interface_dofs(tmp,rhs_for_linear_h,0,0);
	      transfer_interface_dofs(rhs_for_linear_h,rhs_for_linear_h,0,1,Displacement);
	      rhs_for_linear_h.block(1) *= -1;   // copy, negate

	      if (fem_properties.adjoint_type==1)
		{
		  // b = -u + [n^n-n^n-1]/dt	       
		  tmp=0;
		  rhs_for_adjoint=0;
		  transfer_interface_dofs(solution,rhs_for_adjoint,1,0,Displacement);
		  rhs_for_adjoint.block(0)*=1./time_step;
		  transfer_interface_dofs(old_solution,tmp,1,0,Displacement);
		  rhs_for_adjoint.block(0).add(-1./time_step,tmp.block(0));
		  tmp=0;
		  transfer_interface_dofs(solution,tmp,0,0);
		  rhs_for_adjoint.block(0)-=tmp.block(0);
		}
	      else
		{
		  // b = -u + v^	       
		  tmp=0;
		  rhs_for_adjoint=0;
		  transfer_interface_dofs(solution,rhs_for_adjoint,1,0,Velocity);
		  tmp=0;
		  transfer_interface_dofs(solution,tmp,0,0);
		  rhs_for_adjoint.block(0)-=tmp.block(0);
		}	     

	      // get linearized variables
	      rhs_for_linear = rhs_for_linear_h;

	      timer.enter_subsection ("Assemble");
	      Threads::Task<void> s_assembly = Threads::new_task(&FSIProblem<dim>::assemble_structure, *this, linear, true);
	      Threads::Task<void> f_assembly = Threads::new_task(&FSIProblem<dim>::assemble_fluid, *this, linear, true);	      
	      f_assembly.join();
	      dirichlet_boundaries((System)0,linear);
	      s_assembly.join();
	      dirichlet_boundaries((System)1,linear);
	      timer.leave_subsection ();

	      timer.enter_subsection ("Linear Solve");
	      Threads::Task<void> f_factor = Threads::new_task(&SparseDirectUMFPACK::factorize<SparseMatrix<double> >,linear_solver[0], linear_matrix.block(0,0));
	      Threads::Task<void> s_factor = Threads::new_task(&SparseDirectUMFPACK::factorize<SparseMatrix<double> >,linear_solver[1], linear_matrix.block(1,1));
	      
	      f_factor.join();
	      Threads::Task<void> f_solve = Threads::new_task(&FSIProblem<dim>::solve,*this,linear_solver[1],1,linear);
	      s_factor.join();
	      Threads::Task<void> s_solve = Threads::new_task(&FSIProblem<dim>::solve,*this,linear_solver[0],0,linear);
	      f_solve.join();
	      s_solve.join();
	      timer.leave_subsection ();
	      total_solves += 2;

	      

	      if (fem_properties.adjoint_type==1)
		{
		  // -Ax = -w^n + phi^n/dt
		  tmp=0;tmp2=0;
		  transfer_interface_dofs(linear_solution,tmp,1,0,Displacement);
		  tmp.block(0)*=1./time_step;
		  transfer_interface_dofs(linear_solution,tmp2,0,0);
		  tmp.block(0)-=tmp2.block(0);
		}
	      else
		{
		  // -Ax = -w^n + phi_dot^n
		  tmp=0;tmp2=0;
		  transfer_interface_dofs(linear_solution,tmp,1,0,Velocity);
		  transfer_interface_dofs(linear_solution,tmp2,0,0);
		  tmp.block(0)-=tmp2.block(0);
		}
	      
	      // r^0 = b - Ax
	      rhs_for_adjoint.block(0)+=tmp.block(0);

	      if (fem_properties.adjoint_type==1)
		{
		  transfer_interface_dofs(rhs_for_adjoint,rhs_for_adjoint,0,1,Displacement);
		}
	      else
		{
		  transfer_interface_dofs(rhs_for_adjoint,rhs_for_adjoint,0,1,Velocity);
		}

	      rhs_for_adjoint.block(1)*=-1;   // copy, negate
	      // r_s^0 = - sqrt(delta)g^n - sqrt(delta)h^n
	      rhs_for_adjoint_s=0;
	      transfer_interface_dofs(rhs_for_linear_h,rhs_for_adjoint_s,0,0);
	      rhs_for_adjoint_s.block(0)+=stress.block(0);
	      rhs_for_adjoint_s.block(0)*=-sqrt(fem_properties.penalty_epsilon);


	      // get adjoint variables
	      timer.enter_subsection ("Assemble"); 
	      s_assembly = Threads::new_task(&FSIProblem<dim>::assemble_structure, *this, adjoint, true);
	      f_assembly = Threads::new_task(&FSIProblem<dim>::assemble_fluid, *this, adjoint, true);
	      f_assembly.join();
	      dirichlet_boundaries((System)0,adjoint);
	      s_assembly.join();
	      dirichlet_boundaries((System)1,adjoint);
	      timer.leave_subsection ();
	      
	      timer.enter_subsection ("Linear Solve");
	      f_factor = Threads::new_task(&SparseDirectUMFPACK::factorize<SparseMatrix<double> >, adjoint_solver[0], adjoint_matrix.block(0,0));
	      s_factor = Threads::new_task(&SparseDirectUMFPACK::factorize<SparseMatrix<double> >, adjoint_solver[1], adjoint_matrix.block(1,1));
	      
	      f_factor.join();
	      f_solve = Threads::new_task(&FSIProblem<dim>::solve,*this,adjoint_solver[1],1,adjoint);
	      s_factor.join();
	      s_solve = Threads::new_task(&FSIProblem<dim>::solve,*this,adjoint_solver[0],0,adjoint);				
	      f_solve.join();
	      s_solve.join();
	      timer.leave_subsection ();		
	      total_solves += 2;
	      
	      //fluid_constraints.distribute(
	      // apply preconditioner
	      //std::cout << solution.block(0).size() << " " << system_matrix.block(0,0).m() << std::endl; 
	      // for (unsigned int i=0; i<solution.block(0).size(); ++i)
	      //   adjoint_solution.block(0)[i] *= system_matrix.block(0,0).diag_element(i);
	      // for (unsigned int i=0; i<solution.block(1).size(); ++i)
	      //   adjoint_solution.block(1)[i] *= time_step*system_matrix.block(1,1).diag_element(i);
	      // tmp=adjoint_solution;
	      // PreconditionJacobi<SparseMatrix<double> > preconditioner;
	      // preconditioner.initialize(system_matrix.block(0,0), 0.6);
	      // preconditioner.step(adjoint_solution.block(0),tmp.block(0));
	      // preconditioner.initialize(system_matrix.block(1,1), 0.6);
	      // preconditioner.step(adjoint_solution.block(1),tmp.block(1));
	      
	      //adjoint_solution*=float(time_step)/(time_step-1);

	      // p^0 = beta^n - psi^n/dt + sqrt(delta)(-sqrt(delta) g^n -sqrt(delta) h^n)
	      tmp=0; tmp2=0;
	      rhs_for_linear_p=0;

	      if (fem_properties.adjoint_type==1)
		{
		  transfer_interface_dofs(adjoint_solution,tmp,1,0,Displacement);
		  tmp.block(0)*=-1/time_step;
		}
	      else
		{
		  transfer_interface_dofs(adjoint_solution,tmp,1,0,Velocity);
		  tmp.block(0)*=-1;
		}

	      transfer_interface_dofs(adjoint_solution,tmp2,0,0);
	      tmp.block(0)+=tmp2.block(0);
	      tmp.block(0).add(sqrt(fem_properties.penalty_epsilon),rhs_for_adjoint_s.block(0));
	      transfer_interface_dofs(tmp,rhs_for_linear_p,0,0);
	      transfer_interface_dofs(rhs_for_linear_p,rhs_for_linear_p,0,1,Displacement);
	      rhs_for_linear_p.block(1)*=-1;   // copy, negate

	      //rhs_for_linear_p = rhs_for_adjoint; // erase!! not symmetric
	      premultiplier.block(0)=rhs_for_adjoint.block(0); // premult
	      double p_n_norm_square = interface_norm(rhs_for_linear_p.block(0));
	      //double p_n_norm_square = rhs_for_linear_p.block(0).l2_norm();
	      //std::cout <<  p_n_norm_square << std::endl;
	      rhs_for_linear_Ap_s=0;


	      while (std::abs(p_n_norm_square) > fem_properties.cg_tolerance)
		{
		  //std::cout << "more text" << std::endl;
		  // get linearized variables
		  rhs_for_linear = rhs_for_linear_p;
		  timer.enter_subsection ("Assemble"); 
		  assemble_structure(linear, false);
		  assemble_fluid(linear, false);
		  timer.leave_subsection ();
		  for (unsigned int i=0; i<2; ++i)
		    {
		      dirichlet_boundaries((System)i,linear);
		      timer.enter_subsection ("Linear Solve");
		      solve(linear_solver[i], i, linear);
		      timer.leave_subsection ();
		    }
		  ++total_solves;

		  // ||Ap||^2 = (w-phi/dt)^2+delta*h^2
		  tmp=0;tmp2=0;
		  if (fem_properties.adjoint_type==1)
		    {
		      transfer_interface_dofs(linear_solution,tmp,1,0,Displacement);
		      tmp.block(0)*=-1./time_step;
		    }
		  else
		    {
		      transfer_interface_dofs(linear_solution,tmp,1,0,Velocity);
		      tmp.block(0)*=-1;
		    }
		  transfer_interface_dofs(linear_solution,tmp2,0,0);
		  tmp.block(0)+=tmp2.block(0);
		  rhs_for_linear_Ap_s.block(0) = rhs_for_linear_p.block(0);
		  rhs_for_linear_Ap_s *= sqrt(fem_properties.penalty_epsilon);
		  premultiplier.block(0)=rhs_for_linear_p.block(0);
		  double ap_norm_square = interface_norm(tmp.block(0));
		  //double ap_norm_square = tmp.block(0).l2_norm();
		  ap_norm_square += interface_norm(rhs_for_linear_p.block(0));
		  //ap_norm_square += rhs_for_linear_p.block(0).l2_norm();
		  double sigma = p_n_norm_square/ap_norm_square;

		  // h^{n+1} = h^n + sigma * p^n
		  rhs_for_linear_h.block(0).add(sigma,rhs_for_linear_p.block(0));
		  transfer_interface_dofs(rhs_for_linear_h,rhs_for_linear_h,0,1,Displacement);
		  rhs_for_linear_h.block(1)*=-1;   // copy, negate

		  // r^{n+1} = r^n - sigma * Ap
		  // Ap still stored in tmp, could make new vector rhs_for_linear_Ap
		  rhs_for_adjoint.block(0).add(-sigma, tmp.block(0));
		  if (fem_properties.adjoint_type==1)
		    {
		      transfer_interface_dofs(rhs_for_adjoint,rhs_for_adjoint,0,1,Displacement);
		    }
		  else
		    {
		      transfer_interface_dofs(rhs_for_adjoint,rhs_for_adjoint,0,1,Velocity);
		    }
		  rhs_for_adjoint.block(1)*=-1;   // copy, negate
		  rhs_for_adjoint_s.block(0).add(-sigma, rhs_for_linear_Ap_s.block(0));
		  
		  // get adjoint variables (b^{n+1},....)
		  timer.enter_subsection ("Assemble"); 
		  assemble_structure(adjoint, false);
		  assemble_fluid(adjoint, false);
		  timer.leave_subsection ();
		  for (unsigned int i=0; i<2; ++i)
		    {
		      dirichlet_boundaries((System)i,adjoint);
		      timer.enter_subsection ("Linear Solve");
		      solve(adjoint_solver[i], i, adjoint);
		      timer.leave_subsection ();
		    }
		  ++total_solves;

		  // apply preconditioner
		  // adjoint_solution*=float(time_step)/(time_step-1);
		  // for (unsigned int i=0; i<solution.block(0).size(); ++i)
		  // 	adjoint_solution.block(0)[i] *= system_matrix.block(0,0).diag_element(i);
		  // for (unsigned int i=0; i<solution.block(1).size(); ++i)
		  // 	adjoint_solution.block(1)[i] *= time_step*system_matrix.block(1,1).diag_element(i);
		 

		  // tmp=adjoint_solution;
		  // PreconditionJacobi<SparseMatrix<double> > preconditioner;
		  // preconditioner.initialize(system_matrix.block(0,0), 0.6);
		  // preconditioner.step(adjoint_solution.block(0),tmp.block(0));
		  // preconditioner.initialize(system_matrix.block(1,1), 0.6);
		  // preconditioner.step(adjoint_solution.block(1),tmp.block(1));

		  // A*r^{n+1} = beta^{n+1} - psi^{n+1}/dt + sqrt(delta)(second part of r)
		  tmp=0; tmp2=0;
		  if (fem_properties.adjoint_type==1)
		    {
		      transfer_interface_dofs(adjoint_solution,tmp,1,0,Displacement);
		      tmp.block(0)*=-1/time_step;
		    }
		  else
		    {
		      transfer_interface_dofs(adjoint_solution,tmp,1,0,Displacement);
		      tmp.block(0)*=-1;
		    }
		  transfer_interface_dofs(adjoint_solution,tmp2,0,0);
		  tmp.block(0)+=tmp2.block(0);
		  tmp.block(0).add(sqrt(fem_properties.penalty_epsilon),rhs_for_adjoint_s.block(0)); // not sure about this one

		  //rhs_for_linear_p = rhs_for_adjoint; // erase!! not symmetric
		  premultiplier.block(0)=rhs_for_adjoint.block(0);
		  double Astar_r_np1_norm_square = interface_norm(tmp.block(0));
		  //double Astar_r_np1_norm_square = tmp.block(0).l2_norm();
		  double tau = Astar_r_np1_norm_square / p_n_norm_square;

		  // p^{n+1} = A*r^{n+1} + tau * p^{n}
		  rhs_for_linear_p.block(0) *= tau;
		  rhs_for_linear_p.block(0)+=tmp.block(0);
		  transfer_interface_dofs(rhs_for_linear_p,rhs_for_linear_p,0,1,Displacement);
		  rhs_for_linear_p.block(1)*=-1;   // copy, negate
		  p_n_norm_square = interface_norm(rhs_for_linear_p.block(0));
		  //p_n_norm_square = rhs_for_linear_p.block(0).l2_norm();
		  //std::cout << p_n_norm_square << std::endl;
		}		  
	      // update stress
	      stress.block(0) += rhs_for_linear_h.block(0);
	      tmp=0;
	      transfer_interface_dofs(stress,tmp,0,0);
	      transfer_interface_dofs(stress,tmp,1,1,Displacement);
	      stress=0;
	      transfer_interface_dofs(tmp,stress,0,0);
	      transfer_interface_dofs(tmp,stress,1,1,Displacement);
	      transfer_interface_dofs(stress,stress,0,1,Displacement);
	    }

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
    }
  timer.leave_subsection ();
}


template void FSIProblem<2>::run ();
