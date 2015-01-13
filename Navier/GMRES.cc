#include "FSI_Project.h"
/*
%%%%%%%%%%%%%%  MATLAB PROGRAM FOR BICGSTAB %%%%%%%%%%%%%%
function [ c, s ] = rotmat( a, b )

%
% Compute the Givens rotation matrix parameters for a and b.
%
   if ( b == 0.0 ),
      c = 1.0;
      s = 0.0;
   elseif ( abs(b) > abs(a) ),
      temp = a / b;
      s = 1.0 / sqrt( 1.0 + temp^2 );
      c = temp * s;
   else
      temp = b / a;
      c = 1.0 / sqrt( 1.0 + temp^2 );
      s = temp * c;
   end

function [x, error, iter, flag] = GMRES( A, x, b, M, restrt, max_it, tol )

%  -- Iterative template routine --
%     Univ. of Tennessee and Oak Ridge National Laboratory
%     October 1, 1993
%     Details of this algorithm are described in "Templates for the
%     Solution of Linear Systems: Building Blocks for Iterative
%     Methods", Barrett, Berry, Chan, Demmel, Donato, Dongarra,
%     Eijkhout, Pozo, Romine, and van der Vorst, SIAM Publications,
%     1993. (ftp netlib2.cs.utk.edu; cd linalg; get templates.ps).
%
% [x, error, iter, flag] = gmres( A, x, b, M, restrt, max_it, tol )
%
% gmres.m solves the linear system Ax=b
% using the Generalized Minimal residual ( GMRESm ) method with restarts .
%
% input   A        REAL nonsymmetric positive definite matrix
%         x        REAL initial guess vector
%         b        REAL right hand side vector
%         M        REAL preconditioner matrix
%         restrt   INTEGER number of iterations between restarts
%         max_it   INTEGER maximum number of iterations
%         tol      REAL error tolerance
%
% output  x        REAL solution vector
%         error    REAL error norm
%         iter     INTEGER number of iterations performed
%         flag     INTEGER: 0 = solution found to tolerance
%                           1 = no convergence given max_it

   iter = 0;                                         % initialization
   flag = 0;

   bnrm2 = norm( b );
   if  ( bnrm2 == 0.0 ), bnrm2 = 1.0; end

   r = M \ ( b-A*x );
   error = norm( r ) / bnrm2;
   if ( error < tol ) return, end

   [n,n] = size(A);                                  % initialize workspace
   m = restrt;
   V(1:n,1:m+1) = zeros(n,m+1);
   H(1:m+1,1:m) = zeros(m+1,m);
   cs(1:m) = zeros(m,1);
   sn(1:m) = zeros(m,1);
   e1    = zeros(n,1);
   e1(1) = 1.0;

   for iter = 1:max_it,                              % begin iteration

      r = M \ ( b-A*x );
      V(:,1) = r / norm( r );
      s = norm( r )*e1;
      for i = 1:m,                                   % construct orthonormal
	 w = M \ (A*V(:,i));                         % basis using Gram-Schmidt
	 for k = 1:i,
	   H(k,i)= w'*V(:,k);
	   w = w - H(k,i)*V(:,k);
	 end
	 H(i+1,i) = norm( w );
	 V(:,i+1) = w / H(i+1,i);
	 for k = 1:i-1,                              % apply Givens rotation
            temp     =  cs(k)*H(k,i) + sn(k)*H(k+1,i);
            H(k+1,i) = -sn(k)*H(k,i) + cs(k)*H(k+1,i);
            H(k,i)   = temp;
	 end
	 [cs(i),sn(i)] = rotmat( H(i,i), H(i+1,i) ); % form i-th rotation matrix
         temp   = cs(i)*s(i);                        % approximate residual norm
         s(i+1) = -sn(i)*s(i);
	 s(i)   = temp;
         H(i,i) = cs(i)*H(i,i) + sn(i)*H(i+1,i);
         H(i+1,i) = 0.0;
	 error  = abs(s(i+1)) / bnrm2;
	 if ( error <= tol ),                        % update approximation
	    y = H(1:i,1:i) \ s(1:i);                 % and exit
            x = x + V(:,1:i)*y;
	    break;
	 end
      end

      if ( error <= tol ), break, end
      y = H(1:m,1:m) \ s(1:m);
      x = x + V(:,1:m)*y;                            % update approximation
      r = M \ ( b-A*x )                              % compute residual
      s(i+1) = norm(r);
      error = s(i+1) / bnrm2;                        % check convergence
      if ( error <= tol ), break, end;
   end

   if ( error > tol ) flag = 1; end;                 % converged

% END of gmres.m
*/

void rotmat(const double a, const double b, double &c, double &s ) {
  //
  // Compute the Givens rotation matrix parameters for a and b.
  //
  if ( b == 0.0 ) {
    c = 1.0;
    s = 0.0;
  } else if (abs(b) > abs(a)) {
    double temp = a / b;
    s = 1.0 / std::sqrt( 1.0 + pow(temp,2) );
    c = temp * s;
  } else {
    double temp = b / a;
    c = 1.0 / std::sqrt( 1.0 + pow(temp,2) );
    s = temp * c;
  }
}

template <int dim>
unsigned int FSIProblem<dim>::optimization_GMRES (unsigned int &total_solves, const unsigned int initial_timestep_number, const bool random_initial_guess, const unsigned int max_iterations)
{
//   unsigned int restrt = 50;
//   unsigned int iter = 0;  //                                       % initialization
//   unsigned int flag = 0;

//   // This gives the initial guess x_0
//   if (random_initial_guess) {
//     // Generate a random vector
//     for (Vector<double>::iterator it=tmp.block(0).begin(); it!=tmp.block(0).end(); ++it) *it = ((double)std::rand() / (double)(RAND_MAX)) * std::max(physical_properties.rho_f,physical_properties.rho_s) * fem_properties.cg_tolerance;
//   } else {
//     for (Vector<double>::iterator it=tmp.block(0).begin(); it!=tmp.block(0).end(); ++it) *it = std::max(physical_properties.rho_f,physical_properties.rho_s)*fem_properties.cg_tolerance;
//   }
//   rhs_for_linear_h=0;
//   transfer_interface_dofs(tmp,rhs_for_linear_h,0,0);
//   transfer_interface_dofs(rhs_for_linear_h,rhs_for_linear_h,0,1,Displacement);
//   rhs_for_linear_h.block(1) *= -1;   // copy, negate


//   premultiplier.block(0)=rhs_for_adjoint.block(0); // used by interface_norm
//   double original_error_norm = interface_norm(rhs_for_adjoint.block(0));
  
//   BlockVector<double> b = rhs_for_adjoint; // just to initialize it
//   BlockVector<double> r = rhs_for_adjoint; // just to initialize it
//   BlockVector<double> v = rhs_for_adjoint; // just to initialize it
//   v *= 0;

//   // bnrm2 = norm( b );
//   if (fem_properties.adjoint_type==1)
//     {
//       // b = -u + [n^n-n^n-1]/dt	       
//       tmp=0;
//       b=0;
//       transfer_interface_dofs(solution,b,1,0,Displacement);
//       b.block(0)*=1./time_step;
//       transfer_interface_dofs(old_solution,tmp,1,0,Displacement);
//       b.block(0).add(-1./time_step,tmp.block(0));
//       tmp=0;
//       transfer_interface_dofs(solution,tmp,0,0);
//       b.block(0)-=tmp.block(0);
//     }
//   else
//     {
//       // b = -u + v^	       
//       tmp=0;
//       b=0;
//       transfer_interface_dofs(solution,b,1,0,Velocity);
//       tmp=0;
//       transfer_interface_dofs(solution,tmp,0,0);
//       b.block(0)-=tmp.block(0);
//     }
//   premultiplier.block(0) = b.block(0);
//   double bnrm2 = interface_norm(b.block(0));

//   if (bnrm2 == 0.0) bnrm2 = 1.0;

//   //  r = M \ ( b-A*x );
//   // get linearized variables
//   rhs_for_linear = rhs_for_linear_h;
//   // timer.enter_subsection ("Assemble");
//   Threads::Task<void> s_assembly = Threads::new_task(&FSIProblem<dim>::assemble_structure, *this, linear, true);
//   Threads::Task<void> f_assembly = Threads::new_task(&FSIProblem<dim>::assemble_fluid, *this, linear, true);	      
//   f_assembly.join();
//   dirichlet_boundaries((System)0,linear);
//   s_assembly.join();
//   dirichlet_boundaries((System)1,linear);
//   // timer.leave_subsection ();

//   // timer.enter_subsection ("Linear Solve");
//   if (timestep_number==initial_timestep_number) {
//     linear_solver[0].initialize(linear_matrix.block(0,0));
//     linear_solver[1].initialize(linear_matrix.block(1,1));
//     Threads::Task<void> f_solve = Threads::new_task(&FSIProblem<dim>::solve,*this,linear_solver[0],0,linear);
//     Threads::Task<void> s_solve = Threads::new_task(&FSIProblem<dim>::solve,*this,linear_solver[1],1,linear);
//     f_solve.join();
//     s_solve.join();
//   } else {
//     Threads::Task<void> f_factor = Threads::new_task(&SparseDirectUMFPACK::factorize<SparseMatrix<double> >,linear_solver[0], linear_matrix.block(0,0));
//     Threads::Task<void> s_factor = Threads::new_task(&SparseDirectUMFPACK::factorize<SparseMatrix<double> >,linear_solver[1], linear_matrix.block(1,1));
//     f_factor.join();
//     Threads::Task<void> f_solve = Threads::new_task(&FSIProblem<dim>::solve,*this,linear_solver[0],0,linear);
//     s_factor.join();
//     Threads::Task<void> s_solve = Threads::new_task(&FSIProblem<dim>::solve,*this,linear_solver[1],1,linear);
//     f_solve.join();
//     s_solve.join();
//   }
//   // timer.leave_subsection ();
//   total_solves += 2;
//   if (fem_properties.adjoint_type==1)
//     {
//       // -Ax = -w^n + phi^n/dt
//       tmp=0;tmp2=0;
//       transfer_interface_dofs(linear_solution,tmp,1,0,Displacement);
//       tmp.block(0)*=1./time_step;
//       transfer_interface_dofs(linear_solution,tmp2,0,0);
//       tmp.block(0)-=tmp2.block(0);
//     }
//   else
//     {
//       // -Ax = -w^n + phi_dot^n
//       tmp=0;tmp2=0;
//       transfer_interface_dofs(linear_solution,tmp,1,0,Velocity);
//       transfer_interface_dofs(linear_solution,tmp2,0,0);
//       tmp.block(0)-=tmp2.block(0);
//     }
//   // r^0 = b - Ax
//   r.block(0)  = b.block(0);
//   r.block(0) += tmp.block(0);
//   //r = b + tmp;
  
//   premultiplier.block(0)=r.block(0);
//   double error = interface_norm(r.block(0))/bnrm2;
  
//   if (error < fem_properties.cg_tolerance) {
//       return 0;
//     }
//     unsigned int n = stress.block(0).size();
//     // [n,n] = size(A);                                  % initialize workspace
//     unsigned int m = restrt;
//     std::vector<std::vector< double > > V(n,std::vector<double>(m+1));
//     // V(1:n,1:m+1) = zeros(n,m+1);
//     LAPACKFullMatrix<double> H(m+1,m);	
//     //H(1:m+1,1:m) = zeros(m+1,m);
//     std::vector<double> cs(m);
//     // cs(1:m) = zeros(m,1);
//     std::vector<double> sn(m);
//     // sn(1:m) = zeros(m,1);
//     std::vector<double> e1(n);
//     //e1    = zeros(n,1);
//     e1[0] = 1.0;
//     // e1(1) = 1.0;
    
//     for (unsigned int iter = 0; iter < max_iterations; iter++) {
//       //  r = M \ ( b-A*x );
//       // get linearized variables
//       rhs_for_linear = rhs_for_linear_h;
//       // timer.enter_subsection ("Assemble");
//       Threads::Task<void> s_assembly = Threads::new_task(&FSIProblem<dim>::assemble_structure, *this, linear, true);
//       Threads::Task<void> f_assembly = Threads::new_task(&FSIProblem<dim>::assemble_fluid, *this, linear, true);	      
//       f_assembly.join();
//       dirichlet_boundaries((System)0,linear);
//       s_assembly.join();
//       dirichlet_boundaries((System)1,linear);
//       // timer.leave_subsection ();

//       // timer.enter_subsection ("Linear Solve");
//       if (timestep_number==initial_timestep_number) {
// 	linear_solver[0].initialize(linear_matrix.block(0,0));
// 	linear_solver[1].initialize(linear_matrix.block(1,1));
// 	Threads::Task<void> f_solve = Threads::new_task(&FSIProblem<dim>::solve,*this,linear_solver[0],0,linear);
// 	Threads::Task<void> s_solve = Threads::new_task(&FSIProblem<dim>::solve,*this,linear_solver[1],1,linear);
// 	f_solve.join();
// 	s_solve.join();
//       } else {
// 	Threads::Task<void> f_factor = Threads::new_task(&SparseDirectUMFPACK::factorize<SparseMatrix<double> >,linear_solver[0], linear_matrix.block(0,0));
// 	Threads::Task<void> s_factor = Threads::new_task(&SparseDirectUMFPACK::factorize<SparseMatrix<double> >,linear_solver[1], linear_matrix.block(1,1));
// 	f_factor.join();
// 	Threads::Task<void> f_solve = Threads::new_task(&FSIProblem<dim>::solve,*this,linear_solver[0],0,linear);
// 	s_factor.join();
// 	Threads::Task<void> s_solve = Threads::new_task(&FSIProblem<dim>::solve,*this,linear_solver[1],1,linear);
// 	f_solve.join();
// 	s_solve.join();
//       }
//       // timer.leave_subsection ();
//       total_solves += 2;
//       if (fem_properties.adjoint_type==1)
// 	{
// 	  // -Ax = -w^n + phi^n/dt
// 	  tmp=0;tmp2=0;
// 	  transfer_interface_dofs(linear_solution,tmp,1,0,Displacement);
// 	  tmp.block(0)*=1./time_step;
// 	  transfer_interface_dofs(linear_solution,tmp2,0,0);
// 	  tmp.block(0)-=tmp2.block(0);
// 	}
//       else
// 	{
// 	  // -Ax = -w^n + phi_dot^n
// 	  tmp=0;tmp2=0;
// 	  transfer_interface_dofs(linear_solution,tmp,1,0,Velocity);
// 	  transfer_interface_dofs(linear_solution,tmp2,0,0);
// 	  tmp.block(0)-=tmp2.block(0);
// 	}
//       // r^0 = b - Ax
//       r.block(0)  = b.block(0);
//       r.block(0) += tmp.block(0);

//       // r = b.block(0) - M \ ( b-A*x );
//       premultiplier = r.block(0);
//       double r_norm = interface_norm(r.block(0));
//       for (unsigned int l=0; l<n; l++)
// 	V[l][0]=r.block(0)[l]/r_norm;
//       // V(:,1) = r / norm( r );
//       std::vector<double> s = e1;
//       s *= r_norm;
//       // s = norm( r )*e1;
//       for (unsigned int i=0; i<m; i++) {
// 	// construct orthonormal basis using Gram-Schmidt
	
// 	rhs_for_linear = rhs_for_linear_h;
// 	// timer.enter_subsection ("Assemble");
// 	Threads::Task<void> s_assembly = Threads::new_task(&FSIProblem<dim>::assemble_structure, *this, linear, true);
// 	Threads::Task<void> f_assembly = Threads::new_task(&FSIProblem<dim>::assemble_fluid, *this, linear, true);	      
// 	f_assembly.join();
// 	dirichlet_boundaries((System)0,linear);
// 	s_assembly.join();
// 	dirichlet_boundaries((System)1,linear);
// 	// timer.leave_subsection ();

// 	// timer.enter_subsection ("Linear Solve");
// 	if (timestep_number==initial_timestep_number) {
// 	  linear_solver[0].initialize(linear_matrix.block(0,0));
// 	  linear_solver[1].initialize(linear_matrix.block(1,1));
// 	  Threads::Task<void> f_solve = Threads::new_task(&FSIProblem<dim>::solve,*this,linear_solver[0],0,linear);
// 	  Threads::Task<void> s_solve = Threads::new_task(&FSIProblem<dim>::solve,*this,linear_solver[1],1,linear);
// 	  f_solve.join();
// 	  s_solve.join();
// 	} else {
// 	  Threads::Task<void> f_factor = Threads::new_task(&SparseDirectUMFPACK::factorize<SparseMatrix<double> >,linear_solver[0], linear_matrix.block(0,0));
// 	  Threads::Task<void> s_factor = Threads::new_task(&SparseDirectUMFPACK::factorize<SparseMatrix<double> >,linear_solver[1], linear_matrix.block(1,1));
// 	  f_factor.join();
// 	  Threads::Task<void> f_solve = Threads::new_task(&FSIProblem<dim>::solve,*this,linear_solver[0],0,linear);
// 	  s_factor.join();
// 	  Threads::Task<void> s_solve = Threads::new_task(&FSIProblem<dim>::solve,*this,linear_solver[1],1,linear);
// 	  f_solve.join();
// 	  s_solve.join();
// 	}
// 	// timer.leave_subsection ();
// 	total_solves += 2;
// 	if (fem_properties.adjoint_type==1)
// 	  {
// 	    // -Ax = -w^n + phi^n/dt
// 	    tmp=0;tmp2=0;
// 	    transfer_interface_dofs(linear_solution,tmp,1,0,Displacement);
// 	    tmp.block(0)*=1./time_step;
// 	    transfer_interface_dofs(linear_solution,tmp2,0,0);
// 	    tmp.block(0)-=tmp2.block(0);
// 	  }
// 	else
// 	  {
// 	    // -Ax = -w^n + phi_dot^n
// 	    tmp=0;tmp2=0;
// 	    transfer_interface_dofs(linear_solution,tmp,1,0,Velocity);
// 	    transfer_interface_dofs(linear_solution,tmp2,0,0);
// 	    tmp.block(0)-=tmp2.block(0);
// 	  }
// 	BlockVector<double> w = tmp; 
// 	// NOTE HERE: if tmp is usually -Ax, then w = -tmp since w=Av
// 	// w = M \ (A*V(:,i)); 
// 	for (unsigned int k=0; k<i; k++) {
// 	  H(k,i) = 0.0;
// 	  for (unsigned int l=0; l<n; l++) {
// 	    H[k,i] += w.block(0)[l]*V[l][k];
// 	    // H[k,i] = transpose(w)*V(:,k);
// 	  }
// 	  for (unsigned int l=0; l<n; l++) {
// 	    w.block(0)[l] -= H[k,i]*V[l][k];
// 	      // w - H(k,i)*V(:,k);
// 	  }
// 	}
// 	premultiplier.block(0) = w.block(0);
// 	double w_norm = interface_norm(w.block(0));
// 	H[i+1,i] = w_norm;
//         // H(i+1,i) = norm( w );
// 	for (unsigned int l=0; l<n; l++) {
// 	  V[l][i+1] = w.block(0)[l] / H[i+1,i];
// 	}
// 	// V(:,i+1) = w / H(i+1,i);
// 	double temp = 0;
// 	for (unsigned int k=0; k<i-1; k++) { //                            % apply Givens rotation
// 	  temp     =  cs[k]*H[k,i] + sn[k]*H[k+1,i];
// 	  H[k+1,i] = -sn[k]*H[k,i] + cs[k]*H[k+1,i];
// 	  H[k,i]   = temp;
// 	}
// 	rotmat(H[i,i], H[i+1,i], cs[i], sn[i]);
// 	// [cs(i),sn(i)] = rotmat( H(i,i), H(i+1,i) ); % form i-th rotation matrix
// 	temp   = cs[i]*s[i]; //                       % approximate residual norm
// 	s[i+1] = -sn[i]*s[i];
// 	s[i]   = temp;
// 	H[i,i] = cs[i]*H[i,i] + sn[i]*H[i+1,i];
// 	H[i+1,i] = 0.0;
// 	error  = std::abs(s[i+1]) / bnrm2;
// 	// *****************************************************
// 	if ( error <= fem_properties.cg_tolerance ) {   //                     % update approximation
// 	  y = H(1:i,1:i) \ s(1:i); //                 % and exit
// 	  x = x + V(:,1:i)*y;
// 	  break;
// 	}
//       }

//       if ( error <= fem_properties.cg_tolerance ) break;
      
//       y = H(1:m,1:m) \ s(1:m);
//       x = x + V(:,1:m)*y; //                           % update approximation




//       r = M \ ( b-A*x );  //                            % compute residual
//       // ***************************************************
//       premultiplier = r.block(0);
//       s[i+1] = interface_norm(r.block(0));
//       //s(i+1) = norm(r);
//       error = s[i+1] / bnrm2; //   
//       //error = s(i+1) / bnrm2; // % check convergence
//       if ( error <= fem_properties.cg_tolerance ) break;
//     }
    
//     if ( error > fem_properties.cg_tolerance ) return 1;
//     else return 0; //                % converged

//     //% END of gmres.m
  return 1;
}

template unsigned int FSIProblem<2>::optimization_GMRES(unsigned int &total_solves, const unsigned int initial_timestep_number, const bool random_initial_guess, const unsigned int max_iterations);
