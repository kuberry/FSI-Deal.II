#include "FSI_Project.h"

template <int dim>
void FSIProblem<dim>::assemble_fluid_matrix_on_one_cell (const typename DoFHandler<dim>::active_cell_iterator& cell,
						       FullScratchData<dim>& scratch,
						       PerTaskData<dim>& data )
{
  unsigned int state=0, adjoint=1, linear=2;

  //ConditionalOStream pcout(std::cout,Threads::this_thread_id()==scratch.master_thread); 
  //TimerOutput timer (pcout, TimerOutput::summary,
  //		     TimerOutput::wall_times); 
  //timer.enter_subsection ("Beginning");

  FluidStressValues<dim> fluid_stress_values(physical_properties);
  FluidBoundaryValues<dim> fluid_boundary_values_function(physical_properties);
  fluid_boundary_values_function.set_time (time);

  const FEValuesExtractors::Vector velocities (0);
  const FEValuesExtractors::Scalar pressure (dim);

  std::vector<Vector<double> > old_solution_values(scratch.n_q_points, Vector<double>(dim+1));
  std::vector<Vector<double> > old_old_solution_values(scratch.n_q_points, Vector<double>(dim+1));
  std::vector<Vector<double> > adjoint_rhs_values(scratch.n_face_q_points, Vector<double>(dim+1));
  std::vector<Vector<double> > linear_rhs_values(scratch.n_face_q_points, Vector<double>(dim+1));
  std::vector<Vector<double> > u_star_values(scratch.n_q_points, Vector<double>(dim+1));

  std::vector<Tensor<2,dim> > grad_u_old (scratch.n_q_points);
  std::vector<Tensor<2,dim> > grad_u_star (scratch.n_q_points);
  std::vector<Tensor<2,dim> > F (scratch.n_q_points);

  std::vector<Tensor<1,dim> > stress_values (dim+1);
  Vector<double> u_true_side_values (dim+1);
  std::vector<Vector<double> > g_stress_values(scratch.n_face_q_points, Vector<double>(dim+1));
  std::vector<Vector<double> > old_solution_side_values(scratch.n_face_q_points, Vector<double>(dim+1));
  std::vector<Vector<double> > old_old_solution_side_values(scratch.n_face_q_points, Vector<double>(dim+1));
  std::vector<Vector<double> > u_star_side_values(scratch.n_face_q_points, Vector<double>(dim+1));

  std::vector<Tensor<1,dim> > 		  phi_u (fluid_fe.dofs_per_cell);
  std::vector<SymmetricTensor<2,dim> >    symgrad_phi_u (fluid_fe.dofs_per_cell);
  std::vector<Tensor<2,dim> > 		  grad_phi_u (fluid_fe.dofs_per_cell);
  std::vector<double>                     div_phi_u   (fluid_fe.dofs_per_cell);
  std::vector<double>                     phi_p       (fluid_fe.dofs_per_cell);

  scratch.fe_values.reinit(cell);
  data.cell_matrix=0;
  data.cell_rhs=0;

  if (data.assemble_matrix)
    {
      scratch.fe_values.get_function_values (old_solution.block(0), old_solution_values);
      scratch.fe_values.get_function_values (old_old_solution.block(0), old_old_solution_values);
      scratch.fe_values.get_function_values (solution_star.block(0),u_star_values);
      scratch.fe_values[velocities].get_function_gradients(old_solution.block(0),grad_u_old);
      scratch.fe_values[velocities].get_function_gradients(solution_star.block(0),grad_u_star);

      for (unsigned int q=0; q<scratch.n_q_points; ++q)
	{
	  F[q]=0;
	  F[q][0][0]=1;
	  F[q][1][1]=1;
	  double determinantJ = determinant(F[q]);
	  //std::cout << determinantJ << std::endl;
	  Tensor<2,dim> detTimesFinv;
	  detTimesFinv[0][0]=F[q][1][1];
	  detTimesFinv[0][1]=-F[q][0][1];
	  detTimesFinv[1][0]=-F[q][1][0];
	  detTimesFinv[1][1]=F[q][0][0];
	  // This should be computed at this and the previous time step so that we can have a mesh for the center

	  Tensor<1,dim> u_star, u_old, u_old_old;
	  for (unsigned int d=0; d<dim; ++d)
	    {
	      u_star[d] = u_star_values[q](d);
	      u_old[d] = old_solution_values[q](d);
	      u_old_old[d] = old_old_solution_values[q](d);
	    }

	  for (unsigned int k=0; k<fluid_fe.dofs_per_cell; ++k)
	    {
	      phi_u[k]	   = scratch.fe_values[velocities].value (k, q);
	      symgrad_phi_u[k] = scratch.fe_values[velocities].symmetric_gradient (k, q);
	      grad_phi_u[k]    = scratch.fe_values[velocities].gradient (k, q);
	      div_phi_u[k]     = scratch.fe_values[velocities].divergence (k, q);
	      phi_p[k]         = scratch.fe_values[pressure].value (k, q);
	    }
	  for (unsigned int i=0; i<fluid_fe.dofs_per_cell; ++i)
	    {
	      for (unsigned int j=0; j<fluid_fe.dofs_per_cell; ++j)
		{
		  double epsilon = 0*1e-10; // only when all Dirichlet b.c.s
		  if (physical_properties.navier_stokes)
		    {
		      if (physical_properties.stability_terms)
			{
			  if (scratch.mode_type==state)
			    {
			      if (fem_properties.newton)
				{
				  data.cell_matrix(i,j) += 0.5 * pow(fluid_theta,2) * physical_properties.rho_f * 
				    ( 
				     phi_u[j]*(transpose(detTimesFinv)*transpose(grad_u_star[q]))*phi_u[i]
				     - phi_u[j]*(transpose(detTimesFinv)*transpose(grad_phi_u[i]))*u_star
				     + u_star*(transpose(detTimesFinv)*transpose(grad_phi_u[j]))*phi_u[i]
				     - u_star*(transpose(detTimesFinv)*transpose(grad_phi_u[i]))*phi_u[j]
				      ) * scratch.fe_values.JxW(q);
				}
			      else if (fem_properties.richardson)
				{
				  data.cell_matrix(i,j) += 0.5 * pow(fluid_theta,2) * physical_properties.rho_f * 
				    (
				     (4./3*u_old_old-1./3*u_old)*(transpose(detTimesFinv)*transpose(grad_phi_u[j]))*phi_u[i]
				     - (4./3*u_old_old-1./3*u_old)*(transpose(detTimesFinv)*transpose(grad_phi_u[i]))*phi_u[j]
				     ) * scratch.fe_values.JxW(q);
				}
			      else
				{
				  data.cell_matrix(i,j) += 0.5 * pow(fluid_theta,2) * physical_properties.rho_f * 
				    (
				     phi_u[j]*(transpose(detTimesFinv)*transpose(grad_u_star[q]))*phi_u[i]
				     - phi_u[j]*(transpose(detTimesFinv)*transpose(grad_phi_u[i]))*u_star
				     ) * scratch.fe_values.JxW(q);
				}
			      data.cell_matrix(i,j) += (1-fluid_theta)*fluid_theta * physical_properties.rho_f * 
				(
				 phi_u[j]*(transpose(detTimesFinv)*transpose(grad_u_old[q]))*phi_u[i]
				 +u_old*(transpose(detTimesFinv)*transpose(grad_phi_u[j]))*phi_u[i]
				 ) * scratch.fe_values.JxW(q);
			    }
			  else if (scratch.mode_type==adjoint) 
			    {
			      data.cell_matrix(i,j) += 0.5 * pow(fluid_theta,2) * physical_properties.rho_f * 
				( 
				 phi_u[i]*(transpose(detTimesFinv)*transpose(grad_u_star[q]))*phi_u[j]
				 - phi_u[i]*(transpose(detTimesFinv)*transpose(grad_phi_u[j]))*u_star
				 + u_star*(transpose(detTimesFinv)*transpose(grad_phi_u[i]))*phi_u[j]
				 - u_star*(transpose(detTimesFinv)*transpose(grad_phi_u[j]))*phi_u[i]
				  ) * scratch.fe_values.JxW(q);
			      data.cell_matrix(i,j) += (1-fluid_theta)*fluid_theta * physical_properties.rho_f * 
				(
				 phi_u[i]*(transpose(detTimesFinv)*transpose(grad_u_old[q]))*phi_u[j]
				 +u_old*(transpose(detTimesFinv)*transpose(grad_phi_u[i]))*phi_u[j]
				 ) * scratch.fe_values.JxW(q);
			    }
			  else // scratch.mode_type==linear
			    {
			      data.cell_matrix(i,j) += 0.5 * pow(fluid_theta,2) * physical_properties.rho_f * 
				( 
				 phi_u[j]*(transpose(detTimesFinv)*transpose(grad_u_star[q]))*phi_u[i]
				 - phi_u[j]*(transpose(detTimesFinv)*transpose(grad_phi_u[i]))*u_star
				 + u_star*(transpose(detTimesFinv)*transpose(grad_phi_u[j]))*phi_u[i]
				 - u_star*(transpose(detTimesFinv)*transpose(grad_phi_u[i]))*phi_u[j]
				  ) * scratch.fe_values.JxW(q);
			      data.cell_matrix(i,j) += (1-fluid_theta)*fluid_theta * physical_properties.rho_f * 
				(
				 phi_u[j]*(transpose(detTimesFinv)*transpose(grad_u_old[q]))*phi_u[i]
				 +u_old*(transpose(detTimesFinv)*transpose(grad_phi_u[j]))*phi_u[i]
				 ) * scratch.fe_values.JxW(q);
			    }
			}
		      else
			{
			  if (scratch.mode_type==state)
			    {
			      if (fem_properties.newton)
				{
				  data.cell_matrix(i,j) += pow(fluid_theta,2) * physical_properties.rho_f * 
				    ( 
				     phi_u[j]*(transpose(detTimesFinv)*transpose(grad_u_star[q]))*phi_u[i]
				     + u_star*(transpose(detTimesFinv)*transpose(grad_phi_u[j]))*phi_u[i]
				      ) * scratch.fe_values.JxW(q);
				}
			      else if (fem_properties.richardson)
				{
				  data.cell_matrix(i,j) += pow(fluid_theta,2) * physical_properties.rho_f * 
				    (
				     (4./3*u_old_old-1./3*u_old)*(transpose(detTimesFinv)*transpose(grad_phi_u[j]))*phi_u[i]
				     ) * scratch.fe_values.JxW(q);
				}
			      else
				{
				  data.cell_matrix(i,j) += pow(fluid_theta,2) * physical_properties.rho_f * 
				    (
				     phi_u[j]*(transpose(detTimesFinv)*transpose(grad_u_star[q]))*phi_u[i]
				     ) * scratch.fe_values.JxW(q);
				}
			      data.cell_matrix(i,j) += (1-fluid_theta)*fluid_theta * physical_properties.rho_f * 
				(
				 phi_u[j]*(transpose(detTimesFinv)*transpose(grad_u_old[q]))*phi_u[i]
				 +u_old*(transpose(detTimesFinv)*transpose(grad_phi_u[j]))*phi_u[i]
				 ) * scratch.fe_values.JxW(q);
			    }
			  else if (scratch.mode_type==adjoint) 
			    {
			      data.cell_matrix(i,j) += pow(fluid_theta,2) * (physical_properties.rho_f * (phi_u[i]*(transpose(detTimesFinv)*transpose(grad_u_star[q])))*phi_u[j]
									 + physical_properties.rho_f * (u_star*(transpose(detTimesFinv)*transpose(grad_phi_u[i])))*phi_u[j])* scratch.fe_values.JxW(q);
			      data.cell_matrix(i,j) += (1-fluid_theta)*fluid_theta * physical_properties.rho_f * 
				(
				 phi_u[i]*(transpose(detTimesFinv)*transpose(grad_u_old[q]))*phi_u[j]
				 +u_old*(transpose(detTimesFinv)*transpose(grad_phi_u[i]))*phi_u[j]
				 ) * scratch.fe_values.JxW(q);
			    }
			  else // scratch.mode_type==linear
			    {
			      data.cell_matrix(i,j) += pow(fluid_theta,2) * (physical_properties.rho_f * (phi_u[j]*(transpose(detTimesFinv)*transpose(grad_u_star[q])))*phi_u[i]
									 + physical_properties.rho_f * (u_star*(transpose(detTimesFinv)*transpose(grad_phi_u[j])))*phi_u[i])* scratch.fe_values.JxW(q);
			      data.cell_matrix(i,j) += (1-fluid_theta)*fluid_theta * physical_properties.rho_f * 
				(
				 phi_u[j]*(transpose(detTimesFinv)*transpose(grad_u_old[q]))*phi_u[i]
				 +u_old*(transpose(detTimesFinv)*transpose(grad_phi_u[j]))*phi_u[i]
				 ) * scratch.fe_values.JxW(q);
			    }
			}
		    }
		  data.cell_matrix(i,j) += ( physical_properties.rho_f/time_step*phi_u[i]*phi_u[j]
					 + fluid_theta * ( 2*physical_properties.viscosity
							   *0.25*1./determinantJ
							   *scalar_product(grad_phi_u[i]*detTimesFinv+transpose(detTimesFinv)*transpose(grad_phi_u[i]),grad_phi_u[j]*detTimesFinv+transpose(detTimesFinv)*transpose(grad_phi_u[j]))
							   )		   
					 - scalar_product(grad_phi_u[i],transpose(detTimesFinv)) * phi_p[j] // (p,\div v)  momentum
					 - phi_p[i] * scalar_product(grad_phi_u[j],transpose(detTimesFinv)) // (\div u, q) mass
					 + epsilon * phi_p[i] * phi_p[j])
		    * scratch.fe_values.JxW(q);
		  //std::cout << physical_properties.rho_f * (phi_u[j]*(transpose(detTimesFinv)*transpose(grad_u_star[q])))*phi_u[i] << std::endl;
		}
	    }
	  if (scratch.mode_type==state)
	    for (unsigned int i=0; i<fluid_fe.dofs_per_cell; ++i)
	      {
		const double old_p = old_solution_values[q](dim);
		Tensor<1,dim> old_u;
		for (unsigned int d=0; d<dim; ++d)
		  old_u[d] = old_solution_values[q](d);
		const Tensor<1,dim> phi_i_s      = scratch.fe_values[velocities].value (i, q);
		//const Tensor<2,dim> symgrad_phi_i_s = scratch.fe_values[velocities].symmetric_gradient (i, q);
		//const double div_phi_i_s =  scratch.fe_values[velocities].divergence (i, q);
		const Tensor<2,dim> grad_phi_i_s = scratch.fe_values[velocities].gradient (i, q);
		const double div_phi_i_s =  scratch.fe_values[velocities].divergence (i, q);
		if (physical_properties.navier_stokes)
		  {
		    if (fem_properties.newton) 
		      {
			if (physical_properties.stability_terms)
			  {
			    data.cell_rhs(i) += 0.5 * pow(fluid_theta,2) * physical_properties.rho_f 
			      * (
				 u_star*(transpose(detTimesFinv)*transpose(grad_u_star[q]))*phi_i_s
				 - u_star*(transpose(detTimesFinv)*transpose(grad_phi_i_s))*u_star
				 ) * scratch.fe_values.JxW(q);
			  }
			else
			  {
			    data.cell_rhs(i) += pow(fluid_theta,2) * physical_properties.rho_f 
			      * (
				 u_star*(transpose(detTimesFinv)*transpose(grad_u_star[q]))*phi_i_s
				 ) * scratch.fe_values.JxW(q);
			  }
		      }
		    data.cell_rhs(i) -= pow(1-fluid_theta,2) * physical_properties.rho_f * (u_old*(transpose(detTimesFinv)*transpose(grad_u_old[q])))*phi_i_s * scratch.fe_values.JxW(q);
		  }

		    

		data.cell_rhs(i) += (physical_properties.rho_f/time_step *phi_i_s*old_u
				 + (1-fluid_theta)
				 * (-2*physical_properties.viscosity
				    *0.25/determinantJ*scalar_product(grad_u_old[q]*detTimesFinv+transpose(grad_u_old[q]*detTimesFinv),grad_phi_i_s*detTimesFinv+transpose(grad_phi_i_s*detTimesFinv))
				    //*(-2*physical_properties.viscosity
				    //*(grad_u_old[q][0][0]*symgrad_phi_i_s[0][0]
				    //+ 0.5*(grad_u_old[q][1][0]+grad_u_old[q][0][1])*(symgrad_phi_i_s[1][0]+symgrad_phi_i_s[0][1])
				    //+ grad_u_old[q][1][1]*symgrad_phi_i_s[1][1]
				    )
				 )
		  * scratch.fe_values.JxW(q);
			  
	      }
	}
    }
  for (unsigned int face_no=0;
       face_no<GeometryInfo<dim>::faces_per_cell;
       ++face_no)
    {
      if (cell->at_boundary(face_no))
	{
	  if (fluid_boundaries[cell->face(face_no)->boundary_indicator()]==Neumann)
	    {
	      if (scratch.mode_type==state)
		{
		  scratch.fe_face_values.reinit (cell, face_no);

		  if (physical_properties.navier_stokes && physical_properties.stability_terms)
		    {
		      scratch.fe_face_values.get_function_values (old_old_solution.block(0), old_old_solution_side_values);
		      scratch.fe_face_values.get_function_values (old_solution.block(0), old_solution_side_values);
		      scratch.fe_face_values.get_function_values (solution_star.block(0), u_star_side_values);
		      for (unsigned int q=0; q<scratch.n_face_q_points; ++q)
			{
			  fluid_stress_values.vector_gradient(scratch.fe_face_values.quadrature_point(q),
							      stress_values);
			  fluid_boundary_values_function.vector_value(scratch.fe_face_values.quadrature_point(q),
								      u_true_side_values);
			  Tensor<1,dim> u_old_old_side, u_old_side, u_star_side, u_true_side;
			  for (unsigned int d=0; d<dim; ++d)
			    {
			      u_old_old_side[d] = old_old_solution_side_values[q](d);
			      u_old_side[d] = old_solution_side_values[q](d);
			      u_star_side[d] = u_star_side_values[q](d);
			      u_true_side[d] = u_true_side_values(d);
			    }
			  for (unsigned int i=0; i<fluid_fe.dofs_per_cell; ++i)
			    {
			      for (unsigned int j=0; j<fluid_fe.dofs_per_cell; ++j)
				{
				  if (scratch.mode_type==state)
				    {
				      if (fem_properties.newton)
					{
					  data.cell_matrix(i,j) += 0.5 * pow(fluid_theta,2) * physical_properties.rho_f 
					    *( 
					      (scratch.fe_face_values[velocities].value (j, q)*scratch.fe_face_values.normal_vector(q))*(u_star_side*scratch.fe_face_values[velocities].value (i, q))
					      +(u_star_side*scratch.fe_face_values.normal_vector(q))*(scratch.fe_face_values[velocities].value (j, q)*scratch.fe_face_values[velocities].value (i, q))
					       ) * scratch.fe_face_values.JxW(q);
					}
				      else if (fem_properties.richardson)
					{
					  data.cell_matrix(i,j) += 0.5 * pow(fluid_theta,2) * physical_properties.rho_f
					    *(
					      ((4./3*u_old_old_side-1./3*u_old_side)*scratch.fe_face_values.normal_vector(q))*(scratch.fe_face_values[velocities].value (j, q)*scratch.fe_face_values[velocities].value (i, q))
					      ) * scratch.fe_face_values.JxW(q);
					}
				      else
					{
					  data.cell_matrix(i,j) += 0.5 * pow(fluid_theta,2) * physical_properties.rho_f 
					    *( 
					      (u_star_side*scratch.fe_face_values.normal_vector(q))*(scratch.fe_face_values[velocities].value (j, q)*scratch.fe_face_values[velocities].value (i, q))
					       ) * scratch.fe_face_values.JxW(q);

					}
				    }
				  else if (scratch.mode_type==adjoint) 
				    {
				      data.cell_matrix(i,j) += 0.5 * pow(fluid_theta,2) * physical_properties.rho_f 
					*( 
					  (scratch.fe_face_values[velocities].value (i, q)*scratch.fe_face_values.normal_vector(q))*(u_star_side*scratch.fe_face_values[velocities].value (j, q))
					  +(u_star_side*scratch.fe_face_values.normal_vector(q))*(scratch.fe_face_values[velocities].value (i, q)*scratch.fe_face_values[velocities].value (j, q))
					   ) * scratch.fe_face_values.JxW(q);
				    }
				  else // scratch.mode_type==linear
				    {
				      data.cell_matrix(i,j) += 0.5 * pow(fluid_theta,2) * physical_properties.rho_f 
					*( 
					  (scratch.fe_face_values[velocities].value (j, q)*scratch.fe_face_values.normal_vector(q))*(u_star_side*scratch.fe_face_values[velocities].value (i, q))
					  +(u_star_side*scratch.fe_face_values.normal_vector(q))*(scratch.fe_face_values[velocities].value (j, q)*scratch.fe_face_values[velocities].value (i, q))
					   ) * scratch.fe_face_values.JxW(q);
				    }
				}
			      if (fem_properties.newton) 
				{
				  data.cell_rhs(i) += 0.5 * pow(fluid_theta,2) * physical_properties.rho_f 
				    *(
				      (u_star_side*scratch.fe_face_values.normal_vector(q))*(u_star_side*scratch.fe_face_values[velocities].value (i, q))
				      ) * scratch.fe_face_values.JxW(q);
				}
			    }
			}
		    }
		  for (unsigned int q=0; q<scratch.n_face_q_points; ++q)
		    {
		      fluid_stress_values.set_time(time);
		      fluid_stress_values.vector_gradient(scratch.fe_face_values.quadrature_point(q),
							  stress_values);
		      for (unsigned int i=0; i<fluid_fe.dofs_per_cell; ++i)
			{
			  Tensor<2,dim> new_stresses;
			  new_stresses[0][0]=stress_values[0][0];
			  new_stresses[1][0]=stress_values[1][0];
			  new_stresses[1][1]=stress_values[1][1];
			  new_stresses[0][1]=stress_values[0][1];
			  data.cell_rhs(i) += fluid_theta*(scratch.fe_face_values[velocities].value (i, q)*
						       new_stresses*scratch.fe_face_values.normal_vector(q) *
						       scratch.fe_face_values.JxW(q));
			}
		      fluid_stress_values.set_time(time-time_step);
		      fluid_stress_values.vector_gradient(scratch.fe_face_values.quadrature_point(q),
							  stress_values);
		      for (unsigned int i=0; i<fluid_fe.dofs_per_cell; ++i)
			{
			  Tensor<2,dim> new_stresses;
			  new_stresses[0][0]=stress_values[0][0];
			  new_stresses[1][0]=stress_values[1][0];
			  new_stresses[1][1]=stress_values[1][1];
			  new_stresses[0][1]=stress_values[0][1];
			  data.cell_rhs(i) += (1-fluid_theta)*(scratch.fe_face_values[velocities].value (i, q)*
							   new_stresses*scratch.fe_face_values.normal_vector(q) *
							   scratch.fe_face_values.JxW(q));
			}
		    }
		}
	    }
	  else if (fluid_boundaries[cell->face(face_no)->boundary_indicator()]==Interface)
	    {
	      if (scratch.mode_type==state)
		{
		  scratch.fe_face_values.reinit (cell, face_no);
		  if ((!physical_properties.moving_domain && physical_properties.navier_stokes) && physical_properties.stability_terms)
		    {
		      scratch.fe_face_values.get_function_values (old_old_solution.block(0), old_old_solution_side_values);
		      scratch.fe_face_values.get_function_values (old_solution.block(0), old_solution_side_values);
		      scratch.fe_face_values.get_function_values (solution_star.block(0), u_star_side_values);
		      for (unsigned int q=0; q<scratch.n_face_q_points; ++q)
			{
			  fluid_boundary_values_function.vector_value(scratch.fe_face_values.quadrature_point(q),
								      u_true_side_values);
			  Tensor<1,dim> u_old_old_side, u_old_side, u_star_side, u_true_side;
			  for (unsigned int d=0; d<dim; ++d)
			    {
			      u_old_old_side[d] = old_old_solution_side_values[q](d);
			      u_old_side[d] = old_solution_side_values[q](d);
			      u_star_side[d] = u_star_side_values[q](d);
			      u_true_side[d] = u_true_side_values(d);
			    }
			  for (unsigned int i=0; i<fluid_fe.dofs_per_cell; ++i)
			    {
			      for (unsigned int j=0; j<fluid_fe.dofs_per_cell; ++j)
				{
				  if (scratch.mode_type==state)
				    {
				      if (fem_properties.newton)
					{
					  data.cell_matrix(i,j) += 0.5 * pow(fluid_theta,2) * physical_properties.rho_f 
					    *( 
					      (scratch.fe_face_values[velocities].value (j, q)*scratch.fe_face_values.normal_vector(q))*(u_star_side*scratch.fe_face_values[velocities].value (i, q))
					      +(u_star_side*scratch.fe_face_values.normal_vector(q))*(scratch.fe_face_values[velocities].value (j, q)*scratch.fe_face_values[velocities].value (i, q))
					       ) * scratch.fe_face_values.JxW(q);
					}
				      else if (fem_properties.richardson)
					{
					  data.cell_matrix(i,j) += 0.5 * pow(fluid_theta,2) * physical_properties.rho_f
					    *(
					      ((4./3*u_old_old_side-1./3*u_old_side)*scratch.fe_face_values.normal_vector(q))*(scratch.fe_face_values[velocities].value (j, q)*scratch.fe_face_values[velocities].value (i, q))
					      ) * scratch.fe_face_values.JxW(q);
					}
				      else
					{
					  data.cell_matrix(i,j) += 0.5 * pow(fluid_theta,2) * physical_properties.rho_f 
					    *( 
					      (u_star_side*scratch.fe_face_values.normal_vector(q))*(scratch.fe_face_values[velocities].value (j, q)*scratch.fe_face_values[velocities].value (i, q))
					       ) * scratch.fe_face_values.JxW(q);

					}
				    }
				  else if (scratch.mode_type==adjoint) 
				    {
				      data.cell_matrix(i,j) += 0.5 * pow(fluid_theta,2) * physical_properties.rho_f 
					*( 
					  (scratch.fe_face_values[velocities].value (i, q)*scratch.fe_face_values.normal_vector(q))*(u_star_side*scratch.fe_face_values[velocities].value (j, q))
					  +(u_star_side*scratch.fe_face_values.normal_vector(q))*(scratch.fe_face_values[velocities].value (i, q)*scratch.fe_face_values[velocities].value (j, q))
					   ) * scratch.fe_face_values.JxW(q);
				    }
				  else // scratch.mode_type==linear
				    {
				      data.cell_matrix(i,j) += 0.5 * pow(fluid_theta,2) * physical_properties.rho_f 
					*( 
					  (scratch.fe_face_values[velocities].value (j, q)*scratch.fe_face_values.normal_vector(q))*(u_star_side*scratch.fe_face_values[velocities].value (i, q))
					  +(u_star_side*scratch.fe_face_values.normal_vector(q))*(scratch.fe_face_values[velocities].value (j, q)*scratch.fe_face_values[velocities].value (i, q))
					   ) * scratch.fe_face_values.JxW(q);
				    }
				}
			      if (physical_properties.navier_stokes && physical_properties.stability_terms)
				{
				  if (fem_properties.newton) 
				    {
				      data.cell_rhs(i) += 0.5 * pow(fluid_theta,2) * physical_properties.rho_f 
					*(
					  (u_star_side*scratch.fe_face_values.normal_vector(q))*(u_star_side*scratch.fe_face_values[velocities].value (i, q))
					  ) * scratch.fe_face_values.JxW(q);
				    }
				}
			    }
			}
		    }

		  scratch.fe_face_values.get_function_values (stress.block(0), g_stress_values);
		  for (unsigned int q=0; q<scratch.n_face_q_points; ++q)
		    {
		      Tensor<1,dim> g_stress;
		      for (unsigned int d=0; d<dim; ++d)
			g_stress[d] = g_stress_values[q](d);
		      for (unsigned int i=0; i<fluid_fe.dofs_per_cell; ++i)
			{
			  data.cell_rhs(i) += fluid_theta*(scratch.fe_face_values[velocities].value (i, q)*
						       g_stress * scratch.fe_face_values.JxW(q));
			}
		    }

		  scratch.fe_face_values.get_function_values (old_stress.block(0), g_stress_values);
		  for (unsigned int q=0; q<scratch.n_face_q_points; ++q)
		    {
		      Tensor<1,dim> g_stress;
		      for (unsigned int d=0; d<dim; ++d)
			g_stress[d] = g_stress_values[q](d);
		      for (unsigned int i=0; i<fluid_fe.dofs_per_cell; ++i)
			{
			  data.cell_rhs(i) += (1-fluid_theta)*(scratch.fe_face_values[velocities].value (i, q)*
							   g_stress * scratch.fe_face_values.JxW(q));
			}
		    }
		}
	      else if (scratch.mode_type==adjoint)
		{
		  scratch.fe_face_values.reinit (cell, face_no);
		  scratch.fe_face_values.get_function_values (rhs_for_adjoint.block(0), adjoint_rhs_values);

		  for (unsigned int q=0; q<scratch.n_face_q_points; ++q)
		    {
		      Tensor<1,dim> r;
		      for (unsigned int d=0; d<dim; ++d)
			r[d] = adjoint_rhs_values[q](d);
		      for (unsigned int i=0; i<fluid_fe.dofs_per_cell; ++i)
			{
			  data.cell_rhs(i) += fluid_theta*(scratch.fe_face_values[velocities].value (i, q)*
						       r * scratch.fe_face_values.JxW(q));
			}
		    }
		}
	      else // scratch.mode_type==linear
		{
		  scratch.fe_face_values.reinit (cell, face_no);
		  scratch.fe_face_values.get_function_values (rhs_for_linear.block(0), linear_rhs_values);

		  for (unsigned int q=0; q<scratch.n_face_q_points; ++q)
		    {
		      Tensor<1,dim> h;
		      for (unsigned int d=0; d<dim; ++d)
			h[d] = linear_rhs_values[q](d);
		      for (unsigned int i=0; i<fluid_fe.dofs_per_cell; ++i)
			{
			  data.cell_rhs(i) += fluid_theta*(scratch.fe_face_values[velocities].value (i, q)*
						       h * scratch.fe_face_values.JxW(q));
			}
		    }
		}
	    }
	  else if (fluid_boundaries[cell->face(face_no)->boundary_indicator()]==Dirichlet)
	    {
	      if (physical_properties.navier_stokes && physical_properties.stability_terms)
		{
		  scratch.fe_face_values.reinit (cell, face_no);
		  for (unsigned int q=0; q<scratch.n_face_q_points; ++q)
		    {
		      fluid_boundary_values_function.vector_value(scratch.fe_face_values.quadrature_point(q),
								  u_true_side_values);
		      Tensor<1,dim> u_true_side;
		      for (unsigned int d=0; d<dim; ++d)
			{
			  u_true_side[d] = u_true_side_values(d);
			}
		      for (unsigned int i=0; i<fluid_fe.dofs_per_cell; ++i)
			{
			  data.cell_rhs(i) += 0.5 * pow(fluid_theta,2) * physical_properties.rho_f 
			    *(
			      (u_true_side*scratch.fe_face_values.normal_vector(q))*(u_true_side*scratch.fe_face_values[velocities].value (i, q))
			      ) * scratch.fe_face_values.JxW(q);
			}
		    }
		}
	    }
	}
    }
  cell->get_dof_indices (data.dof_indices);
}


template <int dim>
void FSIProblem<dim>::copy_local_fluid_to_global (const PerTaskData<dim>& data )
{
  // ConditionalOStream pcout(std::cout,Threads::this_thread_id()==0);//master_thread); 
  //TimerOutput timer (pcout, TimerOutput::summary,
  //		     TimerOutput::wall_times);
  //timer.enter_subsection ("Copy");
  if (data.assemble_matrix)
    {
      fluid_constraints.distribute_local_to_global (data.cell_matrix, data.cell_rhs,
  							data.dof_indices,
  							*data.global_matrix, *data.global_rhs);
    }
  else
    {
      fluid_constraints.distribute_local_to_global (data.cell_rhs,
  							data.dof_indices,
  							*data.global_rhs);
    }
}

template <int dim>
void FSIProblem<dim>::assemble_fluid (Mode enum_, bool assemble_matrix)
{
  SparseMatrix<double> *fluid_matrix;
  Vector<double> *fluid_rhs;
  if (enum_==state)
    {
      fluid_matrix = &system_matrix.block(0,0);
      fluid_rhs = &system_rhs.block(0);
    }
  else if (enum_==adjoint)
    {
      fluid_matrix = &adjoint_matrix.block(0,0);
      fluid_rhs = &adjoint_rhs.block(0);
    }
  else
    {
      fluid_matrix = &linear_matrix.block(0,0);
      fluid_rhs = &linear_rhs.block(0);
    }

  if (assemble_matrix)
    {
      *fluid_matrix=0;
    }
  *fluid_rhs=0;

  Vector<double> tmp;
  Vector<double> forcing_terms;

  tmp.reinit (fluid_rhs->size());
  forcing_terms.reinit (fluid_rhs->size());

  QGauss<dim>   quadrature_formula(fem_properties.fluid_degree+2);
  FEValues<dim> fe_values (fluid_fe, quadrature_formula,
			   update_values    | update_gradients  |
			   update_quadrature_points | update_JxW_values);

  QGauss<dim-1> face_quadrature_formula(fem_properties.fluid_degree+2);
  FEFaceValues<dim> fe_face_values (fluid_fe, face_quadrature_formula,
				    update_values    | update_normal_vectors |
				    update_quadrature_points  | update_JxW_values);

  if (enum_==state)
    {
      FluidRightHandSide<dim> rhs_function(physical_properties);
      rhs_function.set_time(time);
      VectorTools::create_right_hand_side(fluid_dof_handler,
					  QGauss<dim>(fluid_fe.degree+2),
					  rhs_function,
					  tmp);
      forcing_terms = tmp;
      forcing_terms *= fluid_theta;
      rhs_function.set_time(time - time_step);
      VectorTools::create_right_hand_side(fluid_dof_handler,
					  QGauss<dim>(fluid_fe.degree+2),
					  rhs_function,
					  tmp);
      forcing_terms.add((1 - fluid_theta), tmp);
      *fluid_rhs += forcing_terms;
    }

  master_thread = Threads::this_thread_id();

  PerTaskData<dim> per_task_data(fluid_fe, fluid_matrix, fluid_rhs, assemble_matrix);
  FullScratchData<dim> scratch_data(fluid_fe, quadrature_formula, update_values | update_gradients | update_quadrature_points | update_JxW_values,
					  face_quadrature_formula, update_values | update_normal_vectors | update_quadrature_points  | update_JxW_values,
					  (unsigned int)enum_);
 
  WorkStream::run (fluid_dof_handler.begin_active(),
  		   fluid_dof_handler.end(),
  		   *this,
  		   &FSIProblem<dim>::assemble_fluid_matrix_on_one_cell,
  		   &FSIProblem<dim>::copy_local_fluid_to_global,
  		   scratch_data,
  		   per_task_data);
}

template void FSIProblem<2>::assemble_fluid_matrix_on_one_cell (const DoFHandler<2>::active_cell_iterator& cell,
							     FullScratchData<2>& scratch,
							     PerTaskData<2>& data );

template void FSIProblem<2>::copy_local_fluid_to_global (const PerTaskData<2> &data);

template void FSIProblem<2>::assemble_fluid (Mode enum_, bool assemble_matrix);



// h           fluid.vel.L2   fluid.vel.H1   fluid.press.L2   structure.displ.L2   structure.displ.H1   structure.vel.L2
// 0.353553               -              -                -                    -                    -                  -
// 0.235702            3.57           2.37             2.15                 3.84                 3.04               3.15
// 0.157135            3.67           2.63             2.03                 3.76                 2.71               3.10
