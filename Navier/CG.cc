#include "FSI_Project.h"

template <int dim>
unsigned int FSIProblem<dim>::optimization_CG (unsigned int total_solves, const unsigned int initial_timestep_number)
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

  // timer.enter_subsection ("Assemble");
  Threads::Task<void> s_assembly = Threads::new_task(&FSIProblem<dim>::assemble_structure, *this, linear, true);
  Threads::Task<void> f_assembly = Threads::new_task(&FSIProblem<dim>::assemble_fluid, *this, linear, true);	      
  f_assembly.join();
  dirichlet_boundaries((System)0,linear);
  s_assembly.join();
  dirichlet_boundaries((System)1,linear);
  // timer.leave_subsection ();

  // timer.enter_subsection ("Linear Solve");
  if (timestep_number==1) {
    linear_solver[0].initialize(linear_matrix.block(0,0));
    linear_solver[1].initialize(linear_matrix.block(1,1));
    Threads::Task<void> f_solve = Threads::new_task(&FSIProblem<dim>::solve,*this,linear_solver[0],0,linear);
    Threads::Task<void> s_solve = Threads::new_task(&FSIProblem<dim>::solve,*this,linear_solver[1],1,linear);
    f_solve.join();
    s_solve.join();
  } else {
    Threads::Task<void> f_factor = Threads::new_task(&SparseDirectUMFPACK::factorize<SparseMatrix<double> >,linear_solver[0], linear_matrix.block(0,0));
    Threads::Task<void> s_factor = Threads::new_task(&SparseDirectUMFPACK::factorize<SparseMatrix<double> >,linear_solver[1], linear_matrix.block(1,1));
    f_factor.join();
    Threads::Task<void> f_solve = Threads::new_task(&FSIProblem<dim>::solve,*this,linear_solver[0],0,linear);
    s_factor.join();
    Threads::Task<void> s_solve = Threads::new_task(&FSIProblem<dim>::solve,*this,linear_solver[1],1,linear);
    f_solve.join();
    s_solve.join();
  }
  // timer.leave_subsection ();
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
  // timer.enter_subsection ("Assemble"); 
  s_assembly = Threads::new_task(&FSIProblem<dim>::assemble_structure, *this, adjoint, true);
  f_assembly = Threads::new_task(&FSIProblem<dim>::assemble_fluid, *this, adjoint, true);
  f_assembly.join();
  dirichlet_boundaries((System)0,adjoint);
  s_assembly.join();
  dirichlet_boundaries((System)1,adjoint);
  // timer.leave_subsection ();
	      
  // timer.enter_subsection ("Linear Solve");
  if (timestep_number==1) {
    adjoint_solver[0].initialize(adjoint_matrix.block(0,0));
    adjoint_solver[1].initialize(adjoint_matrix.block(1,1));
    Threads::Task<void> f_solve = Threads::new_task(&FSIProblem<dim>::solve,*this,adjoint_solver[0],0,adjoint);
    Threads::Task<void> s_solve = Threads::new_task(&FSIProblem<dim>::solve,*this,adjoint_solver[1],1,adjoint);				
    f_solve.join();
    s_solve.join();
  } else {
    Threads::Task<void> f_factor = Threads::new_task(&SparseDirectUMFPACK::factorize<SparseMatrix<double> >, adjoint_solver[0], adjoint_matrix.block(0,0));
    Threads::Task<void> s_factor = Threads::new_task(&SparseDirectUMFPACK::factorize<SparseMatrix<double> >, adjoint_solver[1], adjoint_matrix.block(1,1));
    f_factor.join();
    Threads::Task<void> f_solve = Threads::new_task(&FSIProblem<dim>::solve,*this,adjoint_solver[0],0,adjoint);
    s_factor.join();
    Threads::Task<void> s_solve = Threads::new_task(&FSIProblem<dim>::solve,*this,adjoint_solver[1],1,adjoint);				
    f_solve.join();
    s_solve.join();
  }
  // timer.leave_subsection ();		
  total_solves += 2;
	      
  //fluid_constraints.distribute(
  // apply preconditioner
  // std::cout << solution.block(0).size() << " " << system_matrix.block(0,0).m() << std::endl; 
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
      // timer.enter_subsection ("Assemble"); 
      Threads::Task<void> s_assembly = Threads::new_task(&FSIProblem<dim>::assemble_structure, *this, linear, false);
      Threads::Task<void> f_assembly = Threads::new_task(&FSIProblem<dim>::assemble_fluid, *this, linear, false);	      
      f_assembly.join();
      dirichlet_boundaries((System)0,linear);
      s_assembly.join();
      dirichlet_boundaries((System)1,linear);
      // timer.leave_subsection ();

      // timer.enter_subsection ("Linear Solve");
      Threads::Task<void> f_solve = Threads::new_task(&FSIProblem<dim>::solve,*this,linear_solver[0],0,linear);
      Threads::Task<void> s_solve = Threads::new_task(&FSIProblem<dim>::solve,*this,linear_solver[1],1,linear);
      f_solve.join();
      s_solve.join();
      // timer.leave_subsection ();
      total_solves += 2;

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
      // timer.enter_subsection ("Assemble"); 
      s_assembly = Threads::new_task(&FSIProblem<dim>::assemble_structure, *this, adjoint, false);
      f_assembly = Threads::new_task(&FSIProblem<dim>::assemble_fluid, *this, adjoint, false);
      f_assembly.join();
      dirichlet_boundaries((System)0,adjoint);
      s_assembly.join();
      dirichlet_boundaries((System)1,adjoint);
      // timer.leave_subsection ();
	      
      // timer.enter_subsection ("Linear Solve");	      
      f_solve = Threads::new_task(&FSIProblem<dim>::solve,*this,adjoint_solver[0],0,adjoint);
      s_solve = Threads::new_task(&FSIProblem<dim>::solve,*this,adjoint_solver[1],1,adjoint);				
      f_solve.join();
      s_solve.join();
      // timer.leave_subsection ();		
      total_solves += 2;

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

  return total_solves;
}

template unsigned int FSIProblem<2>::optimization_CG (unsigned int total_solves, const unsigned int initial_timestep_number);
