#include "FSI_Project.h"
#include "small_classes.h"
#include <deal.II/base/timer.h>

template <int dim>
void FSIProblem<dim>::structure_state_solve(unsigned int initialized_timestep_number) {
  // STRUCTURE SOLVER ITERATIONS
  // pcout <<"Before structure"<<std::endl;
  // solution_star.block(1)=1;
  // solution_star.block(1) = solution.block(1); 
  if (fem_properties.optimization_method.compare("DN")==0) {
    tmp.block(1) = solution.block(1);
  }
  do {
    solution_star.block(1)=solution.block(1);
    //timer.enter_subsection ("Assemble");
    assemble_structure(state, true);
    
    if (fem_properties.optimization_method.compare("DN")==0)
      system_rhs.block(1) -= stress.block(1);

    //timer.leave_subsection();
    dirichlet_boundaries((System)1,state);
    //timer.enter_subsection ("State Solve"); 
    if (timestep_number==initialized_timestep_number)
      {
	state_solver[1].initialize(system_matrix.block(1,1));
      }
    else 
      {
	state_solver[1].factorize(system_matrix.block(1,1));
      }
    solve(state_solver[1],1,state);
    //timer.leave_subsection ();
    solution_star.block(1)-=solution.block(1);
    //++total_solves;
    std::cout << "S: " << solution_star.block(1).l2_norm() << std::endl;
  } while (solution_star.block(1).l2_norm()>1e-8 && physical_properties.nonlinear_elasticity);
  solution_star.block(1) = solution.block(1); 
  // if (fem_properties.optimization_method.compare("DN")==0) {
  //   tmp.block(1) *= (1-fem_properties.steepest_descent_alpha);
  //   tmp.block(1).add(fem_properties.steepest_descent_alpha, solution.block(1));
  //   solution.block(1) = tmp.block(1);
  //   solution_star.block(1) = solution.block(1);
  // }
 
  
}


template <int dim>
void FSIProblem<dim>::assemble_structure_matrix_on_one_cell (const typename DoFHandler<dim>::active_cell_iterator& cell,
						       FullScratchData<dim>& scratch,
						       PerTaskData<dim>& data )
{
  unsigned int state=0, adjoint=1, linear=2;

  ConditionalOStream pcout(std::cout,Threads::this_thread_id()==master_thread); 
  //TimerOutput timer (pcout, TimerOutput::summary,
  //		     TimerOutput::wall_times); 
  //timer.enter_subsection ("Beginning");

  StructureRightHandSide<dim> rhs_function(physical_properties);

  StructureStressValues<dim> structure_stress_values(physical_properties);
  structure_stress_values.set_time(time);
  StructureStressValues<dim> old_structure_stress_values(physical_properties);
  old_structure_stress_values.set_time(time-time_step);

  const FEValuesExtractors::Vector displacements (0);
  const FEValuesExtractors::Vector velocities (dim);

  std::vector<Vector<double> > old_solution_values(scratch.n_q_points, Vector<double>(2*dim));
  //std::vector<Vector<double> > adjoint_rhs_values(scratch.n_face_q_points, Vector<double>(2*dim));
  //std::vector<Vector<double> > linear_rhs_values(scratch.n_face_q_points, Vector<double>(2*dim));
  std::vector<Tensor<2,dim,double>  > grad_n_star (scratch.n_q_points, Tensor<2,dim,double>());
  std::vector<Tensor<2,dim,double>  > grad_n_old (scratch.n_q_points, Tensor<2,dim,double>());
  //std::vector<Vector<double> > g_stress_values(scratch.n_face_q_points, Vector<double>(2*dim));
  std::vector<Tensor<1,dim,double>  > stress_values (2*dim, Tensor<1,dim,double>());
  Tensor<1,dim,double> old_rhs_values;
  Tensor<1,dim,double> rhs_values;

  std::vector<Tensor<1,dim,double> > 		phi_n (structure_fe.dofs_per_cell, Tensor<1,dim,double>());
  std::vector<Tensor<1,dim,double> >           phi_v       (structure_fe.dofs_per_cell, Tensor<1,dim,double>());
  std::vector<Tensor<2,dim,double> > 	        grad_phi_n (structure_fe.dofs_per_cell, Tensor<2,dim,double>());
  std::vector<Tensor<2,dim,double> > 	        F (structure_fe.dofs_per_cell, Tensor<2,dim,double>());
  std::vector<Tensor<2,dim,double> > 	        E (structure_fe.dofs_per_cell, Tensor<2,dim,double>());
  std::vector<Tensor<2,dim,double> > 	        S (structure_fe.dofs_per_cell, Tensor<2,dim,double>());
  std::vector<Tensor<2,dim,double> > 	        E2 (structure_fe.dofs_per_cell, Tensor<2,dim,double>());
  std::vector<Tensor<2,dim,double> > 	        S2 (structure_fe.dofs_per_cell, Tensor<2,dim,double>());

  scratch.fe_values.reinit(cell);
  data.cell_matrix*=0;
  data.cell_rhs*=0;
  
  Tensor<2,dim,double> Identity;
  for (unsigned int i=0; i<dim; ++i)
    Identity[i][i]=1.0;

  //timer.leave_subsection ();
  //timer.enter_subsection ("Assembly");
  if (data.assemble_matrix)
    {
      //timer.enter_subsection ("Get Data");
      scratch.fe_values.get_function_values (old_solution.block(1), old_solution_values);

      // THESE TERMS ARE DEAL.II gradients, so they are transposed. However, we want the transpose of them,
      // so we keep them AS IS.
      scratch.fe_values[displacements].get_function_gradients(old_solution.block(1),grad_n_old);
      scratch.fe_values[displacements].get_function_gradients(solution_star.block(1),grad_n_star);
     
      //timer.leave_subsection ();
      for (unsigned int q=0; q<scratch.n_q_points;
	   ++q)
	{ 
	  for (unsigned int k=0; k<dim; ++k) {
	    rhs_function.set_time(time);
	    rhs_values[k] = rhs_function.value(scratch.fe_values.quadrature_point(q), k);
	    rhs_function.set_time(time-time_step);
	    old_rhs_values[k] = rhs_function.value(scratch.fe_values.quadrature_point(q), k);
	  }

	  // All terms trailed by 2 represent the extra term from linearization
	  Tensor<2,dim,double> F_star = Identity + grad_n_star[q];
	  Tensor<2,dim,double> E_star = .5 * (transpose(F_star)*F_star - Identity);
	  Tensor<2,dim,double> E_star2 = E_star + .5 * transpose(grad_n_star[q])*grad_n_star[q]; 
	  Tensor<2,dim,double> S_star = physical_properties.lambda*trace(E_star)*Identity + 2*physical_properties.mu*E_star;
	  Tensor<2,dim,double> S_star2 = physical_properties.lambda*trace(E_star2)*Identity + 2*physical_properties.mu*E_star2;
	  double det_F_star = determinant(F_star);

	  Tensor<2,dim,double> F_old = Identity + grad_n_old[q];
	  Tensor<2,dim,double> E_old;
	  if (!physical_properties.nonlinear_elasticity) {
	    E_old = .5 * (transpose(F_old)*F_old - Identity - transpose(grad_n_old[q])*grad_n_old[q]);
	    F_old = Identity;
	  } else {
	    E_old = .5 * (transpose(F_old)*F_old - Identity);
	  }
	  Tensor<2,dim,double> S_old = physical_properties.lambda*trace(E_old)*Identity + 2*physical_properties.mu*E_old;
	  double det_F_old = determinant(F_old);
	  
	  for (unsigned int k=0; k<structure_fe.dofs_per_cell; ++k)
	    {
	      phi_n[k]	       = scratch.fe_values[displacements].value (k, q);
	      grad_phi_n[k]    = scratch.fe_values[displacements].gradient (k, q);
	      phi_v[k]         = scratch.fe_values[velocities].value (k, q);
	      F[k]             = Identity + grad_phi_n[k];
	      if (!physical_properties.nonlinear_elasticity) {
		E[k]             = .5 * (transpose(F[k])*F[k] - Identity - transpose(grad_phi_n[k])*grad_phi_n[k]); // definition of linear stress
		F_star = Identity;
	      } else {
		E[k]             = .5 * (transpose(F[k])*F[k] - Identity - transpose(grad_phi_n[k])*grad_phi_n[k]); // remove nonlinear contribution
		E[k]            +=  .5 * transpose(grad_n_star[q])*grad_phi_n[k];
		if (fem_properties.structure_newton) {
		  E2[k]          = E[k] + .5 * transpose(grad_phi_n[k])*grad_n_star[q]; // add nonlinear contribution back in
		} 
	      }
	      S[k]             = physical_properties.lambda*trace(E[k])*Identity + 2*physical_properties.mu*E[k];
	      S2[k]            = physical_properties.lambda*trace(E2[k])*Identity + 2*physical_properties.mu*E2[k];
	    }

	  for (unsigned int i=0; i<structure_fe.dofs_per_cell; ++i)
	    {
	      const unsigned int
		component_i = structure_fe.system_to_component_index(i).first;
	      for (unsigned int j=0; j<structure_fe.dofs_per_cell; ++j)
		{
		  const unsigned int
		    component_j = structure_fe.system_to_component_index(j).first;

		  if ((scratch.mode_type)==state)
		    {

		      if (component_i<dim)
			{
			  if (component_j<dim)
			    {
			      if (physical_properties.nonlinear_elasticity) {
				if (fem_properties.structure_newton) {
				  data.cell_matrix(i,j)+= (
							   scalar_product(fem_properties.structure_theta*S_star*transpose(grad_phi_n[j]), transpose(grad_phi_n[i]))
							   + scalar_product(fem_properties.structure_theta*S2[j]*transpose(F_star), transpose(grad_phi_n[i]))
							   )*scratch.fe_values.JxW(q);
				} else {
				  data.cell_matrix(i,j)+= (
							   // Formulation 1: (worst)
							   // scalar_product(.5*F_star*S[j], grad_phi_n[i])
							   // or
							   // Formulation 2: (medium)
							   // scalar_product(.5*grad_phi_n[j]*S_star, grad_phi_n[i])
							   // or
							   // Formulation 3: (best)
							   scalar_product(fem_properties.structure_theta*S_star*transpose(grad_phi_n[j]), transpose(grad_phi_n[i]))
							   + scalar_product(fem_properties.structure_theta*S[j], transpose(grad_phi_n[i]))
							   )*scratch.fe_values.JxW(q);
				}
			      } else {
				data.cell_matrix(i,j)+= fem_properties.structure_theta * scalar_product(S[j], .5*(grad_phi_n[i] + transpose(grad_phi_n[i])))
				  *scratch.fe_values.JxW(q);
			      }
			    }
			  else
			    {
			      if (fem_properties.time_dependent) {
				data.cell_matrix(i,j)+=physical_properties.rho_s/time_step*phi_n[i]*phi_v[j]*scratch.fe_values.JxW(q);
			      }
			    }
			}
		      else
			{
			  if (component_j<dim)
			    {
			      if (fem_properties.time_dependent) {
				data.cell_matrix(i,j)+=(-1./time_step*phi_v[i]*phi_n[j])
				  *scratch.fe_values.JxW(q);
			      }
			    }
			  else
			    {
			      if (fem_properties.time_dependent) {
				data.cell_matrix(i,j)+=(fem_properties.structure_theta*phi_v[i]*phi_v[j])
				  *scratch.fe_values.JxW(q);
			      } else {
				data.cell_matrix(i,j)+=phi_v[i]*phi_v[j]*scratch.fe_values.JxW(q);
			      }
			    }
			}
		    }
		  else if ((scratch.mode_type)==linear) 
		    {
		      if (component_i<dim)
			{
			  if (component_j<dim)
			    {
			      if (physical_properties.nonlinear_elasticity) {
				data.cell_matrix(i,j)+= (
							 scalar_product(fem_properties.structure_theta*S_star*transpose(grad_phi_n[j]), transpose(grad_phi_n[i]))
							 + scalar_product(fem_properties.structure_theta*S2[j]*transpose(F_star), transpose(grad_phi_n[i]))
							 )*scratch.fe_values.JxW(q);
			      } else {
				data.cell_matrix(i,j)+= fem_properties.structure_theta * scalar_product(S[j], .5*(grad_phi_n[i] + transpose(grad_phi_n[i])))
				  *scratch.fe_values.JxW(q);
			      }
			    }
			  else
			    {
			      if (fem_properties.time_dependent) {
				data.cell_matrix(i,j)+=physical_properties.rho_s/time_step*phi_n[i]*phi_v[j]*scratch.fe_values.JxW(q);
			      }
			    }
			}
		      else
			{
			  if (component_j<dim)
			    {
			      if (fem_properties.time_dependent) {
				data.cell_matrix(i,j)+=(-1./time_step*phi_v[i]*phi_n[j])
				  *scratch.fe_values.JxW(q);
			      }
			    }
			  else
			    {
			      if (fem_properties.time_dependent) {
				data.cell_matrix(i,j)+=(fem_properties.structure_theta*phi_v[i]*phi_v[j])
				  *scratch.fe_values.JxW(q);
			      } else {
				data.cell_matrix(i,j)+=phi_v[i]*phi_v[j]*scratch.fe_values.JxW(q);
			      }
			    }
			}
		    }
		  else 
		    { // scratch.mode_type==adjoint
		      if (component_i<dim)
			{
			  if (component_j<dim)
			    {
			      if (physical_properties.nonlinear_elasticity) {
				data.cell_matrix(i,j)+= (
							 scalar_product(fem_properties.structure_theta*S_star*transpose(grad_phi_n[i]), transpose(grad_phi_n[j]))
							 + scalar_product(fem_properties.structure_theta*S2[i]*transpose(F_star), transpose(grad_phi_n[j]))
							 )*scratch.fe_values.JxW(q);
			      } else {
				data.cell_matrix(i,j)+= fem_properties.structure_theta * scalar_product(S[i], .5*(grad_phi_n[j] + transpose(grad_phi_n[j])))
				  *scratch.fe_values.JxW(q);
			      }
			    }
			  else
			    {
			      if (fem_properties.time_dependent) {
				data.cell_matrix(i,j)+=-1./time_step*phi_n[i]*phi_v[j]*scratch.fe_values.JxW(q);
			      }
			    }
			}
		      else
			{
			  if (component_j<dim)
			    {
			      if (fem_properties.time_dependent) {
				data.cell_matrix(i,j)+=physical_properties.rho_s/time_step*phi_v[i]*phi_n[j]*scratch.fe_values.JxW(q);
			      }
			    }
			  else
			    {
			      if (fem_properties.time_dependent) {
				data.cell_matrix(i,j)+=(fem_properties.structure_theta*phi_v[i]*phi_v[j])
				  *scratch.fe_values.JxW(q);
			      } else {
				data.cell_matrix(i,j)+=phi_v[i]*phi_v[j]*scratch.fe_values.JxW(q);
			      }
			    }
			}
		    }
		}
	    }
	  
	  if ((scratch.mode_type)==state)
	    {
	      //timer.enter_subsection ("Rhs Assembly");
	      for (unsigned int i=0; i<structure_fe.dofs_per_cell; ++i)
		{
		  const unsigned int component_i = structure_fe.system_to_component_index(i).first;
		  Tensor<1,dim,double> old_n;
		  Tensor<1,dim,double> old_v;
		  for (unsigned int d=0; d<dim; ++d)
		    old_n[d] = old_solution_values[q](d);
		  for (unsigned int d=0; d<dim; ++d)
		    old_v[d] = old_solution_values[q](d+dim);
		  const Tensor<1,dim,double> phi_i_eta      	= scratch.fe_values[displacements].value (i, q);
		  // const Tensor<2,dim> symgrad_phi_i_eta 	= scratch.fe_values[displacements].symmetric_gradient (i, q);
		  const Tensor<2,dim,double> grad_phi_i_eta 	= scratch.fe_values[displacements].gradient (i, q);
		  // const double div_phi_i_eta 			= scratch.fe_values[displacements].divergence (i, q);
		  const Tensor<1,dim,double> phi_i_eta_dot  	= scratch.fe_values[velocities].value (i, q);
		  
		  if (component_i<dim)
		    {
		      if (physical_properties.simulation_type==0 || physical_properties.simulation_type==2) {
		      	data.cell_rhs(i) += ((1-fem_properties.structure_theta)*old_rhs_values + fem_properties.structure_theta*rhs_values) *phi_i_eta* scratch.fe_values.JxW(q);
		      } else {
			data.cell_rhs(i) += ((1-fem_properties.structure_theta)*det_F_old*old_rhs_values + fem_properties.structure_theta*det_F_star*rhs_values) *phi_i_eta* scratch.fe_values.JxW(q);
		      	// data.cell_rhs(i) += ((1-fem_properties.structure_theta)*old_rhs_values + fem_properties.structure_theta*rhs_values) *phi_i_eta* scratch.fe_values.JxW(q);
		      }
		      if (physical_properties.nonlinear_elasticity) {
			if (fem_properties.time_dependent) {
			  data.cell_rhs(i) += (physical_properties.rho_s/time_step *phi_i_eta*old_v
					     // Formulation 1: 
					     // - scalar_product(.5*F_old*S_old, grad_phi_i_eta)
					     // or
					     // Formulation 2:
					     // - scalar_product(.5*F_old*S_old, grad_phi_i_eta)
					     // - scalar_product(.5*S_star, grad_phi_i_eta)
					     // or
					     // Formulation 3:
					     - scalar_product((1-fem_properties.structure_theta)*S_old*transpose(F_old), transpose(grad_phi_i_eta))
					     )* scratch.fe_values.JxW(q);
			}
			if (fem_properties.structure_newton) {
			  data.cell_rhs(i) += ( -scalar_product(fem_properties.structure_theta*S_star, transpose(grad_phi_i_eta))
						 +  scalar_product(fem_properties.structure_theta*S_star2*transpose(F_star), transpose(grad_phi_i_eta))
						)* scratch.fe_values.JxW(q);
			}
		      } else { // linear elasticity
			if (fem_properties.time_dependent) {
			  data.cell_rhs(i) += (physical_properties.rho_s/time_step *phi_i_eta*old_v
					       -(1-fem_properties.structure_theta)*(scalar_product(S_old, .5*(grad_phi_i_eta+transpose(grad_phi_i_eta))))
					       )* scratch.fe_values.JxW(q);
			}
		      }
		    }
		  else
		    {
		      if (fem_properties.time_dependent) {
			data.cell_rhs(i) += (-(1-fem_properties.structure_theta)*phi_i_eta_dot*old_v
					     -1./time_step*phi_i_eta_dot*old_n
					     ) * scratch.fe_values.JxW(q);
		      }
		    }
		}
	      //timer.leave_subsection ();
	    }
	}
    }
  //timer.leave_subsection ();
  //timer.enter_subsection ("RHS");
  for (unsigned int face_no=0;
       face_no<GeometryInfo<dim>::faces_per_cell;
       ++face_no)
    {
      if (cell->at_boundary(face_no))
	{
	  if (structure_boundaries[cell->face(face_no)->boundary_indicator()]==Neumann)
	    {
	      if ((scratch.mode_type)==state)
		{
		  scratch.fe_face_values.reinit (cell, face_no);
		  // GET SIDE ID!
		  // scratch.structure_stress_values.gradient_list( scratch.fe_face_values.get_quadrature_points(), grad_known_stress_now);
		  for (unsigned int q=0; q<scratch.n_face_q_points; ++q)
		    for (unsigned int i=0; i<structure_fe.dofs_per_cell; ++i)
		      {
			structure_stress_values.vector_gradient(scratch.fe_face_values.quadrature_point(q),
									stress_values);
			// A TRANSPOSE IS TAKING PLACE HERE SINCE Deal.II has transposed gradient
			Tensor<2,dim,double> new_stresses;
			new_stresses[0][0]=stress_values[0][0];
			new_stresses[1][0]=stress_values[0][1];
			new_stresses[1][1]=stress_values[1][1];
			new_stresses[0][1]=stress_values[1][0];
			data.cell_rhs(i) += fem_properties.structure_theta*(scratch.fe_face_values[displacements].value (i, q)*
							new_stresses*scratch.fe_face_values.normal_vector(q) *
							scratch.fe_face_values.JxW(q));
		      }
		  // scratch.structure_stress_values.gradient_list( scratch.fe_face_values.get_quadrature_points(), grad_known_stress_old);
		  for (unsigned int q=0; q<scratch.n_face_q_points; ++q)
		    for (unsigned int i=0; i<structure_fe.dofs_per_cell; ++i)
		      {
			old_structure_stress_values.vector_gradient(scratch.fe_face_values.quadrature_point(q),
									stress_values);
			// A TRANSPOSE IS TAKING PLACE HERE SINCE Deal.II has transposed gradient
			Tensor<2,dim,double> new_stresses;
			new_stresses[0][0]=stress_values[0][0];
			new_stresses[1][0]=stress_values[0][1];
			new_stresses[1][1]=stress_values[1][1];
			new_stresses[0][1]=stress_values[1][0];
			data.cell_rhs(i) += (1-fem_properties.structure_theta)*(scratch.fe_face_values[displacements].value (i, q)*
							     new_stresses*scratch.fe_face_values.normal_vector(q) *
							     scratch.fe_face_values.JxW(q));
		      }
		}
	    }
	  // else if (structure_boundaries[cell->face(face_no)->boundary_indicator()]==Interface)
	  //   {    
	  //     scratch.fe_face_values.reinit (cell, face_no);

	  //     AssertThrow(dim==2,ExcNotImplemented()); // This scaling factor only makes sense for 1d line integrals.
	  //     std::vector<double> coordinate_transformation_multiplier_old(scratch.n_face_q_points);
	  //     std::vector<double> coordinate_transformation_multiplier_star(scratch.n_face_q_points);
	  //     if (physical_properties.nonlinear_elasticity) {
	  // 	std::vector<Tensor<2,dim,double>  > face_grad_n_star (scratch.n_face_q_points, Tensor<2,dim,double>());
	  // 	std::vector<Tensor<2,dim,double>  > face_grad_n_old (scratch.n_face_q_points, Tensor<2,dim,double>());
	  // 	scratch.fe_face_values[displacements].get_function_gradients(old_solution.block(1),face_grad_n_old);
	  // 	scratch.fe_face_values[displacements].get_function_gradients(solution_star.block(1),face_grad_n_star);
	  // 	for (unsigned int q=0; q<scratch.n_face_q_points; ++q)
	  // 	  {
	  // 	    coordinate_transformation_multiplier_old[q] = 1;//std::sqrt(std::pow(1+face_grad_n_old[q][0][0],2)+std::pow(1+face_grad_n_old[q][1][1],2));
	  // 	    coordinate_transformation_multiplier_star[q] = 1;//std::sqrt(std::pow(1+face_grad_n_star[q][0][0],2)+std::pow(1+face_grad_n_star[q][1][1],2));
	  // 	  }
	  //     } else { // linear elasticity has no spatial multiplier
	  // 	for (unsigned int q=0; q<scratch.n_face_q_points; ++q)
	  // 	  {
	  // 	    coordinate_transformation_multiplier_old[q] = 1;
	  // 	  }
	  // 	coordinate_transformation_multiplier_star = coordinate_transformation_multiplier_old;
	  //     }

	  //     if ((scratch.mode_type)==state)
	  // 	{
	  // 	  scratch.fe_face_values.get_function_values (stress.block(1), g_stress_values);

	  // 	  for (unsigned int q=0; q<scratch.n_face_q_points; ++q)
	  // 	    {
	  // 	      Tensor<1,dim,double> g_stress;
	  // 	      for (unsigned int d=0; d<dim; ++d)
	  // 		g_stress[d] = g_stress_values[q](d);
	  // 	      for (unsigned int i=0; i<structure_fe.dofs_per_cell; ++i)
	  // 		{
	  // 		  data.cell_rhs(i) += fem_properties.structure_theta*coordinate_transformation_multiplier_star[q]*(scratch.fe_face_values[displacements].value (i, q)*
	  // 					  (-g_stress) * scratch.fe_face_values.JxW(q));
	  // 		}
	  // 	    }
	  // 	  scratch.fe_face_values.get_function_values (old_stress.block(1), g_stress_values);
	  // 	  for (unsigned int q=0; q<scratch.n_face_q_points; ++q)
	  // 	    {
	  // 	      Tensor<1,dim,double> g_stress;
	  // 	      for (unsigned int d=0; d<dim; ++d)
	  // 		g_stress[d] = g_stress_values[q](d);
	  // 	      for (unsigned int i=0; i<structure_fe.dofs_per_cell; ++i)
	  // 		{
	  // 		  data.cell_rhs(i) += (1-fem_properties.structure_theta)*coordinate_transformation_multiplier_old[q]*(scratch.fe_face_values[displacements].value (i, q)*
	  // 					  (-g_stress) * scratch.fe_face_values.JxW(q));
	  // 		}
	  // 	    }
	  // 	}
	  //     else if ((scratch.mode_type)==adjoint)
	  // 	{
	  // 	  scratch.fe_face_values.get_function_values (rhs_for_adjoint.block(1), adjoint_rhs_values);

	  // 	  for (unsigned int q=0; q<scratch.n_face_q_points; ++q)
	  // 	    {
	  // 	      Tensor<1,dim,double> r;
	  // 	      if (fem_properties.adjoint_type==1)
	  // 		{
	  // 		  for (unsigned int d=0; d<dim; ++d)
	  // 		    r[d] = adjoint_rhs_values[q](d);
	  // 		  for (unsigned int i=0; i<structure_fe.dofs_per_cell; ++i)
	  // 		    {
	  // 		      data.cell_rhs(i) += fem_properties.structure_theta*coordinate_transformation_multiplier_star[q]*(scratch.fe_face_values[displacements].value (i, q)*
	  // 							   r * scratch.fe_face_values.JxW(q));
	  // 		    }
	  // 		}
	  // 	      else
	  // 		{
	  // 		  if (fem_properties.optimization_method.compare("Gradient")==0)
	  // 		    {
	  // 		      for (unsigned int d=0; d<dim; ++d)
	  // 			r[d] = adjoint_rhs_values[q](d+dim);
	  // 		      for (unsigned int i=0; i<structure_fe.dofs_per_cell; ++i)
	  // 			{
	  // 			  data.cell_rhs(i) += fem_properties.structure_theta*coordinate_transformation_multiplier_star[q]*(scratch.fe_face_values[velocities].value (i, q)*
	  // 							       r * scratch.fe_face_values.JxW(q));
	  // 			}
	  // 		    }
	  // 		  else
	  // 		    {
	  // 		      for (unsigned int d=0; d<dim; ++d)
	  // 			r[d] = adjoint_rhs_values[q](d+dim);
	  // 		      for (unsigned int i=0; i<structure_fe.dofs_per_cell; ++i)
	  // 			{
	  // 			  data.cell_rhs(i) += fem_properties.structure_theta*coordinate_transformation_multiplier_star[q]*(scratch.fe_face_values[velocities].value (i, q)*
	  // 							       r * scratch.fe_face_values.JxW(q));
	  // 			}
	  // 		    }
	  // 		}

	  // 	    }
	  // 	}
	  //     else // (scratch.mode_type)==linear
	  // 	{
	  // 	  scratch.fe_face_values.get_function_values (rhs_for_linear.block(1), linear_rhs_values);
	  // 	  for (unsigned int q=0; q<scratch.n_face_q_points; ++q)
	  // 	    {
	  // 	      Tensor<1,dim,double> h;
	  // 	      for (unsigned int d=0; d<dim; ++d)
	  // 		h[d] = linear_rhs_values[q](d);
	  // 	      for (unsigned int i=0; i<structure_fe.dofs_per_cell; ++i)
	  // 		{
	  // 		  data.cell_rhs(i) += fem_properties.structure_theta*coordinate_transformation_multiplier_star[q]*(scratch.fe_face_values[displacements].value (i, q)*
	  // 						       h * scratch.fe_face_values.JxW(q));
	  // 		}
	  // 	    }
	  // 	}
	  //   }
	}
    }
    //timer.leave_subsection ();
    
    cell->get_dof_indices (data.dof_indices);
}

template <int dim>
void FSIProblem<dim>::copy_local_structure_to_global (const PerTaskData<dim>& data )
{
  // ConditionalOStream pcout(std::cout,Threads::this_thread_id()==0);//master_thread); 
  //TimerOutput timer (pcout, TimerOutput::summary,
  //		     TimerOutput::wall_times);
  //timer.enter_subsection ("Copy");
  if (data.assemble_matrix)
    {
      structure_constraints.distribute_local_to_global (data.cell_matrix, data.cell_rhs,
  							data.dof_indices,
  							*data.global_matrix, *data.global_rhs);
    }
  else
    {
      structure_constraints.distribute_local_to_global (data.cell_rhs,
  							data.dof_indices,
  							*data.global_rhs);
    }
}

template <int dim>
void FSIProblem<dim>::assemble_structure_stresses_on_one_cell (const typename DoFHandler<dim>::active_cell_iterator& cell,
						       FullScratchData<dim>& scratch,
						       PerTaskData<dim>& data )
{
  unsigned int state=0, adjoint=1, linear=2;

  const FEValuesExtractors::Vector displacements (0);
  const FEValuesExtractors::Vector velocities (dim);

  std::vector<Vector<double> > g_stress_values(scratch.n_face_q_points, Vector<double>(2*dim));
  std::vector<Vector<double> > adjoint_rhs_values(scratch.n_face_q_points, Vector<double>(2*dim));
  std::vector<Vector<double> > linear_rhs_values(scratch.n_face_q_points, Vector<double>(2*dim));

  data.cell_rhs*=0;

  for (unsigned int face_no=0;
       face_no<GeometryInfo<dim>::faces_per_cell;
       ++face_no)
    {
      if (cell->at_boundary(face_no))
	{
	  if (structure_boundaries[cell->face(face_no)->boundary_indicator()]==Interface)
	    {    
	      scratch.fe_face_values.reinit (cell, face_no);

	      //AssertThrow(dim==2,ExcNotImplemented()); // This scaling factor only makes sense for 1d line integrals.
	      std::vector<double> coordinate_transformation_multiplier_old(scratch.n_face_q_points);
	      std::vector<double> coordinate_transformation_multiplier_star(scratch.n_face_q_points);
	      if (physical_properties.nonlinear_elasticity) {
		std::vector<Tensor<2,dim,double>  > face_grad_n_star (scratch.n_face_q_points, Tensor<2,dim,double>());
		std::vector<Tensor<2,dim,double>  > face_grad_n_old (scratch.n_face_q_points, Tensor<2,dim,double>());
		scratch.fe_face_values[displacements].get_function_gradients(old_solution.block(1),face_grad_n_old);
		scratch.fe_face_values[displacements].get_function_gradients(solution_star.block(1),face_grad_n_star);
		for (unsigned int q=0; q<scratch.n_face_q_points; ++q)
		  {
		    // when this is 1, we are not weighting the integral
		    // this is being dealt with by moving the domain before integrating, instead
		    coordinate_transformation_multiplier_old[q] = 1;//std::sqrt(std::pow(1+face_grad_n_old[q][0][0],2)+std::pow(1+face_grad_n_old[q][1][1],2));
		    coordinate_transformation_multiplier_star[q] = 1;//std::sqrt(std::pow(1+face_grad_n_star[q][0][0],2)+std::pow(1+face_grad_n_star[q][1][1],2));
		  }
	      } else { // linear elasticity has no spatial multiplier
		for (unsigned int q=0; q<scratch.n_face_q_points; ++q)
		  {
		    coordinate_transformation_multiplier_old[q] = 1;
		  }
		coordinate_transformation_multiplier_star = coordinate_transformation_multiplier_old;
	      }

	      if ((scratch.mode_type)==state)
		{
		  scratch.fe_face_values.get_function_values (stress.block(1), g_stress_values);

		  for (unsigned int q=0; q<scratch.n_face_q_points; ++q)
		    {
		      Tensor<1,dim,double> g_stress;
		      for (unsigned int d=0; d<dim; ++d)
			g_stress[d] = g_stress_values[q](d);
		      for (unsigned int i=0; i<structure_fe.dofs_per_cell; ++i)
			{
			  data.cell_rhs(i) += fem_properties.structure_theta*coordinate_transformation_multiplier_star[q]*(scratch.fe_face_values[displacements].value (i, q)*
						  (-g_stress) * scratch.fe_face_values.JxW(q));
			}
		    }
		  scratch.fe_face_values.get_function_values (old_stress.block(1), g_stress_values);
		  for (unsigned int q=0; q<scratch.n_face_q_points; ++q)
		    {
		      Tensor<1,dim,double> g_stress;
		      for (unsigned int d=0; d<dim; ++d)
			g_stress[d] = g_stress_values[q](d);
		      for (unsigned int i=0; i<structure_fe.dofs_per_cell; ++i)
			{
			  data.cell_rhs(i) += (1-fem_properties.structure_theta)*coordinate_transformation_multiplier_old[q]*(scratch.fe_face_values[displacements].value (i, q)*
						  (-g_stress) * scratch.fe_face_values.JxW(q));
			}
		    }
		}
	      else if ((scratch.mode_type)==adjoint)
		{
		  scratch.fe_face_values.get_function_values (rhs_for_adjoint.block(1), adjoint_rhs_values);

		  for (unsigned int q=0; q<scratch.n_face_q_points; ++q)
		    {
		      Tensor<1,dim,double> r;
		      if (fem_properties.adjoint_type==1)
			{
			  for (unsigned int d=0; d<dim; ++d)
			    r[d] = adjoint_rhs_values[q](d);
			  for (unsigned int i=0; i<structure_fe.dofs_per_cell; ++i)
			    {
			      data.cell_rhs(i) += fem_properties.structure_theta*coordinate_transformation_multiplier_star[q]*(scratch.fe_face_values[displacements].value (i, q)*
								   r * scratch.fe_face_values.JxW(q));
			    }
			}
		      else
			{
			  if (fem_properties.optimization_method.compare("Gradient")==0)
			    {
			      for (unsigned int d=0; d<dim; ++d)
				r[d] = adjoint_rhs_values[q](d+dim);
			      for (unsigned int i=0; i<structure_fe.dofs_per_cell; ++i)
				{
				  data.cell_rhs(i) += fem_properties.structure_theta*coordinate_transformation_multiplier_star[q]*(scratch.fe_face_values[velocities].value (i, q)*
								       r * scratch.fe_face_values.JxW(q));
				}
			    }
			  else
			    {
			      for (unsigned int d=0; d<dim; ++d)
				r[d] = adjoint_rhs_values[q](d+dim);
			      for (unsigned int i=0; i<structure_fe.dofs_per_cell; ++i)
				{
				  data.cell_rhs(i) += fem_properties.structure_theta*coordinate_transformation_multiplier_star[q]*(scratch.fe_face_values[velocities].value (i, q)*
								       r * scratch.fe_face_values.JxW(q));
				}
			    }
			}

		    }
		}
	      else // (scratch.mode_type)==linear
		{
		  scratch.fe_face_values.get_function_values (rhs_for_linear.block(1), linear_rhs_values);
		  for (unsigned int q=0; q<scratch.n_face_q_points; ++q)
		    {
		      Tensor<1,dim,double> h;
		      for (unsigned int d=0; d<dim; ++d)
			h[d] = linear_rhs_values[q](d);
		      for (unsigned int i=0; i<structure_fe.dofs_per_cell; ++i)
			{
			  data.cell_rhs(i) += fem_properties.structure_theta*coordinate_transformation_multiplier_star[q]*(scratch.fe_face_values[displacements].value (i, q)*
							       h * scratch.fe_face_values.JxW(q));
			}
		    }
		}
	    }
	}
    }
    //timer.leave_subsection ();
    
    cell->get_dof_indices (data.dof_indices);
}

template <int dim>
void FSIProblem<dim>::assemble_structure (Mode enum_, bool assemble_matrix)
{
  SparseMatrix<double> *structure_matrix;
  Vector<double> *structure_rhs;
  if (enum_==state)
    {
      structure_matrix = &system_matrix.block(1,1);
      structure_rhs = &system_rhs.block(1);
    }
  else if (enum_==adjoint)
    {
      structure_matrix = &adjoint_matrix.block(1,1);
      structure_rhs = &adjoint_rhs.block(1);
    }
  else
    {
      structure_matrix = &linear_matrix.block(1,1);
      structure_rhs = &linear_rhs.block(1);
    }

  if (assemble_matrix)
    {
      (*structure_matrix) *= 0;
    }
  (*structure_rhs) *= 0;

  // Vector<double> temporary_vector(structure_rhs->size());
  // Vector<double> forcing_terms(structure_rhs->size());

  QGauss<dim>   quadrature_formula(fem_properties.structure_degree+2);

  QGauss<dim-1> face_quadrature_formula(fem_properties.structure_degree+2);

  // if (enum_==state)
  //   {
  //     StructureRightHandSide<dim> rhs_function(physical_properties);
  //     rhs_function.set_time(time);
  //     VectorTools::create_right_hand_side(structure_dof_handler,
  // 					  QGauss<dim>(structure_fe.degree+2),
  // 					  rhs_function,
  // 					  temporary_vector);
  //     forcing_terms = temporary_vector;
  //     forcing_terms *= fem_properties.structure_theta;
  //     rhs_function.set_time(time - time_step);
  //     VectorTools::create_right_hand_side(structure_dof_handler,
  // 					  QGauss<dim>(structure_fe.degree+2),
  // 					  rhs_function,
  // 					  temporary_vector);
  //     forcing_terms.add((1-fem_properties.structure_theta), temporary_vector);
  //     *structure_rhs += forcing_terms;
  //   }


  master_thread = Threads::this_thread_id();

  PerTaskData<dim> per_task_data(structure_fe, structure_matrix, structure_rhs, assemble_matrix);
  UpdateFlags face_update_flags = update_values | update_normal_vectors | update_quadrature_points  | update_JxW_values;

  if (physical_properties.nonlinear_elasticity) {
    face_update_flags = face_update_flags | update_gradients;
  }

  FullScratchData<dim> scratch_data(structure_fe, quadrature_formula, update_values | update_gradients | update_quadrature_points | update_JxW_values,
				    face_quadrature_formula, face_update_flags, (unsigned int)enum_);

  WorkStream::run (structure_dof_handler.begin_active(),
		   structure_dof_handler.end(),
		   *this,
		   &FSIProblem<dim>::assemble_structure_matrix_on_one_cell,
		   &FSIProblem<dim>::copy_local_structure_to_global,
		   scratch_data,
		   per_task_data);

  QTrapez<dim> vertices_quadrature_formula;
  FEValues<dim> fe_vertices_values (structure_fe, vertices_quadrature_formula,
  				    update_values);

  std::vector<Vector<double> > z_vertices(vertices_quadrature_formula.size(), Vector<double>(2*dim));
  std::vector<Vector<double> > z_old_vertices(vertices_quadrature_formula.size(), Vector<double>(2*dim));
  std::set<unsigned int> visited_vertices;
  typename DoFHandler<dim>::active_cell_iterator
    cell = structure_dof_handler.begin_active(),
    endc = structure_dof_handler.end();
  for (; cell!=endc; ++cell) {
    fe_vertices_values.reinit(cell);
    if (physical_properties.nonlinear_elasticity)
      {
  	for (unsigned int i=0; i<GeometryInfo<dim>::vertices_per_cell; ++i)
  	  {
  	    if (visited_vertices.find(cell->vertex_index(i)) == visited_vertices.end())
  	      {
  		Point<dim> &v = cell->vertex(i);
  		fe_vertices_values.get_function_values(old_solution.block(1), z_vertices);
  		for (unsigned int j=0; j<dim; ++j)
  		  {
  		    v(j) += z_vertices[i](j);
  		  }
  		visited_vertices.insert(cell->vertex_index(i));
  	      }
  	  }
      }
  }

  if (fem_properties.optimization_method.compare("DN")!=0)
    {
      WorkStream::run (structure_dof_handler.begin_active(),
		       structure_dof_handler.end(),
		       *this,
		       &FSIProblem<dim>::assemble_structure_stresses_on_one_cell,
		       &FSIProblem<dim>::copy_local_structure_to_global,
		       scratch_data,
		       per_task_data);
    }

  visited_vertices.clear();
  cell = structure_dof_handler.begin_active();
  endc = structure_dof_handler.end();
  for (; cell!=endc; ++cell) {
    fe_vertices_values.reinit(cell);
    if (physical_properties.nonlinear_elasticity)
      {
  	for (unsigned int i=0; i<GeometryInfo<dim>::vertices_per_cell; ++i)
  	  {
  	    if (visited_vertices.find(cell->vertex_index(i)) == visited_vertices.end())
  	      {
  		Point<dim> &v = cell->vertex(i);
  		fe_vertices_values.get_function_values(old_solution.block(1), z_vertices);
  		for (unsigned int j=0; j<dim; ++j)
  		  {
  		    v(j) -= z_vertices[i](j);
  		  }
  		visited_vertices.insert(cell->vertex_index(i));
  	      }
  	  }
      }
  }
}
template void FSIProblem<2>::structure_state_solve(unsigned int initialized_timestep_number);

template void FSIProblem<2>::assemble_structure_matrix_on_one_cell (const DoFHandler<2>::active_cell_iterator& cell,
							     FullScratchData<2>& scratch,
							     PerTaskData<2>& data );

template void FSIProblem<2>::assemble_structure_stresses_on_one_cell (const DoFHandler<2>::active_cell_iterator& cell,
							     FullScratchData<2>& scratch,
							     PerTaskData<2>& data );

template void FSIProblem<2>::copy_local_structure_to_global (const PerTaskData<2> &data);
							     
template void FSIProblem<2>::assemble_structure (Mode enum_, bool assemble_matrix);





template void FSIProblem<3>::structure_state_solve(unsigned int initialized_timestep_number);

template void FSIProblem<3>::assemble_structure_matrix_on_one_cell (const DoFHandler<3>::active_cell_iterator& cell,
							     FullScratchData<3>& scratch,
							     PerTaskData<3>& data );

template void FSIProblem<3>::assemble_structure_stresses_on_one_cell (const DoFHandler<3>::active_cell_iterator& cell,
							     FullScratchData<3>& scratch,
							     PerTaskData<3>& data );

template void FSIProblem<3>::copy_local_structure_to_global (const PerTaskData<3> &data);
							     
template void FSIProblem<3>::assemble_structure (Mode enum_, bool assemble_matrix);


// dt = 0.0555556 h_f = 0.157135 h_s = 0.114531 L2(T) error [fluid] = 0.00118288,  L2(T) error [structure] = 8.58267e-05
//  L2(0,T;H1(t)) error [fluid] = 0.00120148,  Pressure error [fluid] = 0.0024587,  L2(0,T;H1(t)) errors [structure] = 3.52664e-05 L2(T) error [structure_vel] = 2.99115e-05
// h           fluid.vel.L2   fluid.vel.H1   fluid.press.L2   structure.displ.L2   structure.displ.H1   structure.vel.L2
// 0.353553               -              -                -                    -                    -                  -
// 0.235702            2.91           3.53             2.17                 1.25                 2.50               3.29
// 0.157135            2.49           3.54             1.85                 1.79                 2.81               3.27
// ----> data to match
