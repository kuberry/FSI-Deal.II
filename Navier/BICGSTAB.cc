#include "FSI_Project.h"

/* Based off of the following Matlab code for BICGSTAB (Preconditioned)

Variables in this program - Equivalent in PDE:
A[some vector] - linear operator results -> solution_adjoint(fluid) - solution_adjoint(structure_displacement)./dt [ or solution_adjoint(structure_velocity)]
b = -u + [n^n-n^n-1]/dt [ or -u + v^n] = 
x = rhs_for_linear_h


%%%%%%%%%%%%%%  MATLAB PROGRAM FOR BICGSTAB %%%%%%%%%%%%%%
dim = 200;
A= magic(dim);
for i=1:dim
    A(i,i) = A(i,i) * 1.1^i;
end
b= rand(dim,1);

x = ones(dim,1);

r = b - A*x;
r_tilde = r;

rho_n = 1;
alpha = 1;
w_n = 1;
v = 0*x;
p_n = 0*x;

count = 0;
P = eye(size(A));
%P = eye(size(A));%diag(1:size(A))*eye(size(A));
for i=1:dim
   P(i,i) = P(i,i) * 1/1.1^i; 
end
while (norm(r,2)>1e-8)
   rho_np1 = r_tilde' * r;
   beta = (rho_np1/rho_n)*(alpha/w_n);
   rho_n = rho_np1;
   
   p_n = r + beta *( p_n -w_n*v);
   y = P*p_n;
   v = A*y;  % optional preconditioner here
   alpha = rho_n / (r_tilde'* v);
   s = r - alpha * v;
   z = P*s;
   t = A*z;
   % w_n = t'*s / (t'*t);
   w_n = (P*t)'*(P*s) / ((P*t)'*(P*t));
   % x = x + alpha*p_n + w_n*s;
   x = x + alpha*y + w_n*z;
   r = s - w_n * t;
   norm(b-A*x,2)
   count = count + 1;
end

A*x-b
P*A*x-P*b
fprintf('%i iterations.\n', count);
%%%%%%%%%%%%%% END OF MATLAB PROGRAM %%%%%%%%%%%%%%
 */


template <int dim>
unsigned int FSIProblem<dim>::optimization_BICGSTAB (unsigned int &total_solves, const unsigned int initial_timestep_number, const bool random_initial_guess, const unsigned int max_iterations)
{
  // This gives the initial guess x_0
  if (random_initial_guess) {
    // Generate a random vector
    for (Vector<double>::iterator it=tmp.block(0).begin(); it!=tmp.block(0).end(); ++it) *it = ((double)std::rand() / (double)(RAND_MAX)) * std::max(physical_properties.rho_f,physical_properties.rho_s) * fem_properties.cg_tolerance;
  } else {
    for (Vector<double>::iterator it=tmp.block(0).begin(); it!=tmp.block(0).end(); ++it) *it = std::max(physical_properties.rho_f,physical_properties.rho_s)*fem_properties.cg_tolerance;
  }
  rhs_for_linear_h=0;
  transfer_interface_dofs(tmp,rhs_for_linear_h,0,0);
  transfer_interface_dofs(rhs_for_linear_h,rhs_for_linear_h,0,1,Displacement);
  rhs_for_linear_h.block(1) *= -1;   // copy, negate


  premultiplier.block(0)=rhs_for_adjoint.block(0); // used by interface_norm
  double original_error_norm = interface_norm(rhs_for_adjoint.block(0));
  
  BlockVector<double> b = rhs_for_adjoint; // just to initialize it
  BlockVector<double> r = rhs_for_adjoint; // just to initialize it
  BlockVector<double> v = rhs_for_adjoint; // just to initialize it
  v *= 0;
  BlockVector<double> p_n = rhs_for_adjoint; // just to initialize it
  p_n *= 0;
  double rho_n = 1;
  double alpha = 1;
  double w_n = 1;
  if (fem_properties.adjoint_type==1)
    {
      // b = -u + [n^n-n^n-1]/dt	       
      tmp=0;
      b=0;
      transfer_interface_dofs(solution,b,1,0,Displacement);
      b.block(0)*=1./time_step;
      transfer_interface_dofs(old_solution,tmp,1,0,Displacement);
      b.block(0).add(-1./time_step,tmp.block(0));
      tmp=0;
      transfer_interface_dofs(solution,tmp,0,0);
      b.block(0)-=tmp.block(0);
    }
  else
    {
      // b = -u + v^	       
      tmp=0;
      b=0;
      transfer_interface_dofs(solution,b,1,0,Velocity);
      tmp=0;
      transfer_interface_dofs(solution,tmp,0,0);
      b.block(0)-=tmp.block(0);
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
  if (timestep_number==initial_timestep_number) {
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
  r.block(0)  = b.block(0);
  r.block(0) += tmp.block(0);
  //r = b + tmp;
  BlockVector<double> r_tilde = r; // only block(0) has values

  
  double error_norm = 1;

  unsigned int loop_count = 1;
  while (error_norm/original_error_norm > fem_properties.cg_tolerance && loop_count <= max_iterations) {
    premultiplier.block(0)=r_tilde.block(0); // used by interface_norm
    double rho_np1 = interface_norm(r.block(0));
    //     rho_np1 = r_tilde * r;
    
    double beta = (rho_np1/rho_n)*(alpha/w_n);
    double rho_n = rho_np1;
   
    // p_n = r + beta *( p_n -w_n*v);
    p_n.block(0) *= beta;
    p_n.block(0) += r.block(0);
    p_n.block(0).add(-beta*w_n,v.block(0));
    // p_n.block(0) = r.block(0) + beta *( p_n.block(0) - w_n*v.block(0));

    // Optional preconditioning step
    // y = P*p_n;
    // v = A*y;

    rhs_for_linear = p_n;
    transfer_interface_dofs(rhs_for_linear,rhs_for_linear,0,1,Displacement);
    rhs_for_linear.block(1) *= -1;
    // IMPORTANT - BUT NOT SURE WHAT TO DO WITH IT RIGHT NOW
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
    v.block(0)=tmp.block(0);	  

    premultiplier.block(0)=r_tilde.block(0); // used by interface_norm
    alpha = rho_n / interface_norm(v.block(0));

    BlockVector<double> s = r; // only block(0) has values
    s.block(0)  = r.block(0);
    s.block(0).add(-alpha, v.block(0));
    //s.block(0) = r.block(0) - alpha* v.block(0);
    //s = r - alpha * v;
    
    // Optional preconditioning step
    //z = P*s;
    // t = A*z;
    rhs_for_linear = s;
    transfer_interface_dofs(rhs_for_linear,rhs_for_linear,0,1,Displacement);
    rhs_for_linear.block(1) *= -1;
    // IMPORTANT - BUT NOT SURE WHAT TO DO WITH IT RIGHT NOW
    // timer.enter_subsection ("Assemble"); 
    s_assembly = Threads::new_task(&FSIProblem<dim>::assemble_structure, *this, linear, false);
    f_assembly = Threads::new_task(&FSIProblem<dim>::assemble_fluid, *this, linear, false);	      
    f_assembly.join();
    dirichlet_boundaries((System)0,linear);
    s_assembly.join();
    dirichlet_boundaries((System)1,linear);
    // timer.leave_subsection ();
    // timer.enter_subsection ("Linear Solve");
    f_solve = Threads::new_task(&FSIProblem<dim>::solve,*this,linear_solver[0],0,linear);
    s_solve = Threads::new_task(&FSIProblem<dim>::solve,*this,linear_solver[1],1,linear);
    f_solve.join();
    s_solve.join();
    // timer.leave_subsection ();
    total_solves += 2;
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
    BlockVector<double> t = tmp;
    
    premultiplier.block(0)=t.block(0); // used by interface_norm
    double ts = interface_norm(s.block(0));
    double tt = interface_norm(t.block(0));
    
    w_n = ts / tt;
    // w_n = t*s / (t*t);
    //w_n = (P*t)*(P*s) / ((P*t)*(P*t));

    rhs_for_linear_h.block(0).add(alpha,p_n.block(0));
    rhs_for_linear_h.block(0).add(w_n,  s.block(0));
    // x = x + alpha*p_n + w_n*s;
    // x = x + alpha*y + w_n*z;


    // r = s - w_n * t;
    r.block(0) = s.block(0);
    r.block(0).add(-w_n, t.block(0));

    premultiplier.block(0)=r.block(0); // used by interface_norm
    error_norm = interface_norm(r.block(0));
    if (loop_count % 100 == 0) std::cout << "BICG Err: " << error_norm << std::endl;

    std::vector<std::vector<std::string> > solution_names(3);
    switch (dim)
      {
      case 2:
	solution_names[0].push_back ("r_x");
	solution_names[0].push_back ("r_y");
	solution_names[0].push_back ("r_p");
	break;
      default:
	AssertThrow (false, ExcNotImplemented());
      }
    DataOut<dim> residual_data_out;
    residual_data_out.add_data_vector (fluid_dof_handler,r.block(0), solution_names[0]);
    residual_data_out.build_patches (fem_properties.fluid_degree-1);
    const std::string residual_filename = "residual-" +
      Utilities::int_to_string (loop_count, 4) +
      ".vtk";
    std::ofstream residual_output (residual_filename.c_str());
    residual_data_out.write_vtk (residual_output);

    loop_count++;
  }

  if (loop_count > max_iterations) {
    std::cout << "BICGSTAB Stalled: Restarting..." << std::endl;
    return 1;
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
  return 0; // success
}

template unsigned int FSIProblem<2>::optimization_BICGSTAB (unsigned int &total_solves, const unsigned int initial_timestep_number, const bool random_initial_guess, const unsigned int max_iterations);
