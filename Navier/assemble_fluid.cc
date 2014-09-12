#include "FSI_Project.h"

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

  const FEValuesExtractors::Vector velocities (0);
  const FEValuesExtractors::Scalar pressure (dim);

  Vector<double> tmp;
  Vector<double> forcing_terms;

  tmp.reinit (fluid_rhs->size());
  forcing_terms.reinit (fluid_rhs->size());

  if (assemble_matrix)
    {
      *fluid_matrix=0;
    }
  *fluid_rhs=0;

  QGauss<dim>   quadrature_formula(fem_properties.fluid_degree+2);
  QGauss<dim-1> face_quadrature_formula(fem_properties.fluid_degree+2);

  FEValues<dim> fe_values (fluid_fe, quadrature_formula,
			   update_values    |
			   update_quadrature_points  |
			   update_JxW_values |
			   update_gradients);

  FEFaceValues<dim> fe_face_values (fluid_fe, face_quadrature_formula,
				    update_values    | update_normal_vectors |
				    update_quadrature_points  | update_JxW_values);

  const unsigned int   dofs_per_cell   = fluid_fe.dofs_per_cell;
  const unsigned int   n_q_points      = quadrature_formula.size();
  const unsigned int   n_face_q_points = face_quadrature_formula.size();
  FullMatrix<double>   local_matrix (dofs_per_cell, dofs_per_cell);
  Vector<double>       local_rhs (dofs_per_cell);

  std::vector<Vector<double> > old_solution_values(n_q_points, Vector<double>(dim+1));
  std::vector<Vector<double> > old_old_solution_values(n_q_points, Vector<double>(dim+1));
  std::vector<Vector<double> > adjoint_rhs_values(n_face_q_points, Vector<double>(dim+1));
  std::vector<Vector<double> > linear_rhs_values(n_face_q_points, Vector<double>(dim+1));
  std::vector<Vector<double> > u_star_values(n_q_points, Vector<double>(dim+1));

  std::vector<Tensor<2,dim> > grad_u_old (n_q_points);
  std::vector<Tensor<2,dim> > grad_u_star (n_q_points);
  std::vector<Tensor<2,dim> > F (n_q_points);

  std::vector<types::global_dof_index> local_dof_indices (dofs_per_cell);

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

  FluidStressValues<dim> fluid_stress_values(physical_properties);
  FluidBoundaryValues<dim> fluid_boundary_values_function(physical_properties);
  fluid_boundary_values_function.set_time (time);

  std::vector<Tensor<1,dim> > stress_values (dim+1);
  Vector<double> u_true_side_values (dim+1);
  std::vector<Vector<double> > g_stress_values(n_face_q_points, Vector<double>(dim+1));
  std::vector<Vector<double> > old_solution_side_values(n_face_q_points, Vector<double>(dim+1));
  std::vector<Vector<double> > old_old_solution_side_values(n_face_q_points, Vector<double>(dim+1));
  std::vector<Vector<double> > u_star_side_values(n_face_q_points, Vector<double>(dim+1));

  std::vector<Tensor<1,dim> > 		  phi_u (dofs_per_cell);
  std::vector<SymmetricTensor<2,dim> >      symgrad_phi_u (dofs_per_cell);
  std::vector<Tensor<2,dim> > 		  grad_phi_u (dofs_per_cell);
  std::vector<double>                       div_phi_u   (dofs_per_cell);
  std::vector<double>                       phi_p       (dofs_per_cell);

  double length = 0;
  double residual = 0;


  typename DoFHandler<dim>::active_cell_iterator
    cell = fluid_dof_handler.begin_active(),
    endc = fluid_dof_handler.end();
  //if (enum_==state)
  for (; cell!=endc; ++cell)
    {
      fe_values.reinit (cell);
      local_matrix = 0;
      local_rhs = 0;

      if (assemble_matrix)
	{
	  fe_values.get_function_values (old_solution.block(0), old_solution_values);
	  fe_values.get_function_values (old_old_solution.block(0), old_old_solution_values);
	  fe_values.get_function_values (solution_star.block(0),u_star_values);
	  fe_values[velocities].get_function_gradients(old_solution.block(0),grad_u_old);
	  fe_values[velocities].get_function_gradients(solution_star.block(0),grad_u_star);

	  for (unsigned int q=0; q<n_q_points; ++q)
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

	      for (unsigned int k=0; k<dofs_per_cell; ++k)
		{
		  phi_u[k]	   = fe_values[velocities].value (k, q);
		  symgrad_phi_u[k] = fe_values[velocities].symmetric_gradient (k, q);
		  grad_phi_u[k]    = fe_values[velocities].gradient (k, q);
		  div_phi_u[k]     = fe_values[velocities].divergence (k, q);
		  phi_p[k]         = fe_values[pressure].value (k, q);
		}
	      for (unsigned int i=0; i<dofs_per_cell; ++i)
		{
		  for (unsigned int j=0; j<dofs_per_cell; ++j)
		    {
		      double epsilon = 0*1e-10; // only when all Dirichlet b.c.s
		      if (physical_properties.navier_stokes)
			{
			  if (physical_properties.stability_terms)
			    {
			      if (enum_==state)
				{
				  if (fem_properties.newton)
				    {
				      local_matrix(i,j) += 0.5 * pow(fluid_theta,2) * physical_properties.rho_f * 
					( 
					 phi_u[j]*(transpose(detTimesFinv)*transpose(grad_u_star[q]))*phi_u[i]
					 - phi_u[j]*(transpose(detTimesFinv)*transpose(grad_phi_u[i]))*u_star
					 + u_star*(transpose(detTimesFinv)*transpose(grad_phi_u[j]))*phi_u[i]
					 - u_star*(transpose(detTimesFinv)*transpose(grad_phi_u[i]))*phi_u[j]
					  ) * fe_values.JxW(q);
				    }
				  else if (fem_properties.richardson)
				    {
				      local_matrix(i,j) += 0.5 * pow(fluid_theta,2) * physical_properties.rho_f * 
					(
					 (4./3*u_old_old-1./3*u_old)*(transpose(detTimesFinv)*transpose(grad_phi_u[j]))*phi_u[i]
					 - (4./3*u_old_old-1./3*u_old)*(transpose(detTimesFinv)*transpose(grad_phi_u[i]))*phi_u[j]
					 ) * fe_values.JxW(q);
				    }
				  else
				    {
				      local_matrix(i,j) += 0.5 * pow(fluid_theta,2) * physical_properties.rho_f * 
					(
					 phi_u[j]*(transpose(detTimesFinv)*transpose(grad_u_star[q]))*phi_u[i]
					 - phi_u[j]*(transpose(detTimesFinv)*transpose(grad_phi_u[i]))*u_star
					 ) * fe_values.JxW(q);
				    }
				  local_matrix(i,j) += (1-fluid_theta)*fluid_theta * physical_properties.rho_f * 
				    (
				     phi_u[j]*(transpose(detTimesFinv)*transpose(grad_u_old[q]))*phi_u[i]
				     +u_old*(transpose(detTimesFinv)*transpose(grad_phi_u[j]))*phi_u[i]
				     ) * fe_values.JxW(q);
				}
			      else if (enum_==adjoint) 
				{
				  local_matrix(i,j) += 0.5 * pow(fluid_theta,2) * physical_properties.rho_f * 
				    ( 
				     phi_u[i]*(transpose(detTimesFinv)*transpose(grad_u_star[q]))*phi_u[j]
				     - phi_u[i]*(transpose(detTimesFinv)*transpose(grad_phi_u[j]))*u_star
				     + u_star*(transpose(detTimesFinv)*transpose(grad_phi_u[i]))*phi_u[j]
				     - u_star*(transpose(detTimesFinv)*transpose(grad_phi_u[j]))*phi_u[i]
				      ) * fe_values.JxW(q);
				  local_matrix(i,j) += (1-fluid_theta)*fluid_theta * physical_properties.rho_f * 
				    (
				     phi_u[i]*(transpose(detTimesFinv)*transpose(grad_u_old[q]))*phi_u[j]
				     +u_old*(transpose(detTimesFinv)*transpose(grad_phi_u[i]))*phi_u[j]
				     ) * fe_values.JxW(q);
				}
			      else // enum_==linear
				{
				  local_matrix(i,j) += 0.5 * pow(fluid_theta,2) * physical_properties.rho_f * 
				    ( 
				     phi_u[j]*(transpose(detTimesFinv)*transpose(grad_u_star[q]))*phi_u[i]
				     - phi_u[j]*(transpose(detTimesFinv)*transpose(grad_phi_u[i]))*u_star
				     + u_star*(transpose(detTimesFinv)*transpose(grad_phi_u[j]))*phi_u[i]
				     - u_star*(transpose(detTimesFinv)*transpose(grad_phi_u[i]))*phi_u[j]
				      ) * fe_values.JxW(q);
				  local_matrix(i,j) += (1-fluid_theta)*fluid_theta * physical_properties.rho_f * 
				    (
				     phi_u[j]*(transpose(detTimesFinv)*transpose(grad_u_old[q]))*phi_u[i]
				     +u_old*(transpose(detTimesFinv)*transpose(grad_phi_u[j]))*phi_u[i]
				     ) * fe_values.JxW(q);
				}
			    }
			  else
			    {
			      if (enum_==state)
				{
				  if (fem_properties.newton)
				    {
				      local_matrix(i,j) += pow(fluid_theta,2) * physical_properties.rho_f * 
					( 
					 phi_u[j]*(transpose(detTimesFinv)*transpose(grad_u_star[q]))*phi_u[i]
					 + u_star*(transpose(detTimesFinv)*transpose(grad_phi_u[j]))*phi_u[i]
					  ) * fe_values.JxW(q);
				    }
				  else if (fem_properties.richardson)
				    {
				      local_matrix(i,j) += pow(fluid_theta,2) * physical_properties.rho_f * 
					(
					 (4./3*u_old_old-1./3*u_old)*(transpose(detTimesFinv)*transpose(grad_phi_u[j]))*phi_u[i]
					 ) * fe_values.JxW(q);
				    }
				  else
				    {
				      local_matrix(i,j) += pow(fluid_theta,2) * physical_properties.rho_f * 
					(
					 phi_u[j]*(transpose(detTimesFinv)*transpose(grad_u_star[q]))*phi_u[i]
					 ) * fe_values.JxW(q);
				    }
				  local_matrix(i,j) += (1-fluid_theta)*fluid_theta * physical_properties.rho_f * 
				    (
				     phi_u[j]*(transpose(detTimesFinv)*transpose(grad_u_old[q]))*phi_u[i]
				     +u_old*(transpose(detTimesFinv)*transpose(grad_phi_u[j]))*phi_u[i]
				     ) * fe_values.JxW(q);
				}
			      else if (enum_==adjoint) 
				{
				  local_matrix(i,j) += pow(fluid_theta,2) * (physical_properties.rho_f * (phi_u[i]*(transpose(detTimesFinv)*transpose(grad_u_star[q])))*phi_u[j]
									     + physical_properties.rho_f * (u_star*(transpose(detTimesFinv)*transpose(grad_phi_u[i])))*phi_u[j])* fe_values.JxW(q);
				  local_matrix(i,j) += (1-fluid_theta)*fluid_theta * physical_properties.rho_f * 
				    (
				     phi_u[i]*(transpose(detTimesFinv)*transpose(grad_u_old[q]))*phi_u[j]
				     +u_old*(transpose(detTimesFinv)*transpose(grad_phi_u[i]))*phi_u[j]
				     ) * fe_values.JxW(q);
				}
			      else // enum_==linear
				{
				  local_matrix(i,j) += pow(fluid_theta,2) * (physical_properties.rho_f * (phi_u[j]*(transpose(detTimesFinv)*transpose(grad_u_star[q])))*phi_u[i]
									     + physical_properties.rho_f * (u_star*(transpose(detTimesFinv)*transpose(grad_phi_u[j])))*phi_u[i])* fe_values.JxW(q);
				  local_matrix(i,j) += (1-fluid_theta)*fluid_theta * physical_properties.rho_f * 
				    (
				     phi_u[j]*(transpose(detTimesFinv)*transpose(grad_u_old[q]))*phi_u[i]
				     +u_old*(transpose(detTimesFinv)*transpose(grad_phi_u[j]))*phi_u[i]
				     ) * fe_values.JxW(q);
				}
			    }
			}
		      local_matrix(i,j) += ( physical_properties.rho_f/time_step*phi_u[i]*phi_u[j]
					     + fluid_theta * ( 2*physical_properties.viscosity
							       *0.25*1./determinantJ
							       *scalar_product(grad_phi_u[i]*detTimesFinv+transpose(detTimesFinv)*transpose(grad_phi_u[i]),grad_phi_u[j]*detTimesFinv+transpose(detTimesFinv)*transpose(grad_phi_u[j]))
							       )		   
					     - scalar_product(grad_phi_u[i],transpose(detTimesFinv)) * phi_p[j] // (p,\div v)  momentum
					     - phi_p[i] * scalar_product(grad_phi_u[j],transpose(detTimesFinv)) // (\div u, q) mass
					     + epsilon * phi_p[i] * phi_p[j])
			* fe_values.JxW(q);
		      //std::cout << physical_properties.rho_f * (phi_u[j]*(transpose(detTimesFinv)*transpose(grad_u_star[q])))*phi_u[i] << std::endl;
		    }
		}
	      if (enum_==state)
		for (unsigned int i=0; i<dofs_per_cell; ++i)
		  {
		    const double old_p = old_solution_values[q](dim);
		    Tensor<1,dim> old_u;
		    for (unsigned int d=0; d<dim; ++d)
		      old_u[d] = old_solution_values[q](d);
		    const Tensor<1,dim> phi_i_s      = fe_values[velocities].value (i, q);
		    //const Tensor<2,dim> symgrad_phi_i_s = fe_values[velocities].symmetric_gradient (i, q);
		    //const double div_phi_i_s =  fe_values[velocities].divergence (i, q);
		    const Tensor<2,dim> grad_phi_i_s = fe_values[velocities].gradient (i, q);
		    const double div_phi_i_s =  fe_values[velocities].divergence (i, q);
		    if (physical_properties.navier_stokes)
		      {
			if (fem_properties.newton) 
			  {
			    if (physical_properties.stability_terms)
			      {
				local_rhs(i) += 0.5 * pow(fluid_theta,2) * physical_properties.rho_f 
				  * (
				     u_star*(transpose(detTimesFinv)*transpose(grad_u_star[q]))*phi_i_s
				     - u_star*(transpose(detTimesFinv)*transpose(grad_phi_i_s))*u_star
				     ) * fe_values.JxW(q);
			      }
			    else
			      {
				local_rhs(i) += pow(fluid_theta,2) * physical_properties.rho_f 
				  * (
				     u_star*(transpose(detTimesFinv)*transpose(grad_u_star[q]))*phi_i_s
				     ) * fe_values.JxW(q);
			      }
			  }
			local_rhs(i) -= pow(1-fluid_theta,2) * physical_properties.rho_f * (u_old*(transpose(detTimesFinv)*transpose(grad_u_old[q])))*phi_i_s * fe_values.JxW(q);
		      }

		    

		    local_rhs(i) += (physical_properties.rho_f/time_step *phi_i_s*old_u
				     + (1-fluid_theta)
				     * (-2*physical_properties.viscosity
					*0.25/determinantJ*scalar_product(grad_u_old[q]*detTimesFinv+transpose(grad_u_old[q]*detTimesFinv),grad_phi_i_s*detTimesFinv+transpose(grad_phi_i_s*detTimesFinv))
					//*(-2*physical_properties.viscosity
					//*(grad_u_old[q][0][0]*symgrad_phi_i_s[0][0]
					//+ 0.5*(grad_u_old[q][1][0]+grad_u_old[q][0][1])*(symgrad_phi_i_s[1][0]+symgrad_phi_i_s[0][1])
					//+ grad_u_old[q][1][1]*symgrad_phi_i_s[1][1]
					)
				     )
		      * fe_values.JxW(q);
			  
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
		  if (enum_==state)
		    {
		      fe_face_values.reinit (cell, face_no);

		      if (physical_properties.navier_stokes && physical_properties.stability_terms)
			{
			  fe_face_values.get_function_values (old_old_solution.block(0), old_old_solution_side_values);
			  fe_face_values.get_function_values (old_solution.block(0), old_solution_side_values);
			  fe_face_values.get_function_values (solution_star.block(0), u_star_side_values);
			  for (unsigned int q=0; q<n_face_q_points; ++q)
			    {
			      fluid_stress_values.vector_gradient(fe_face_values.quadrature_point(q),
								  stress_values);
			      fluid_boundary_values_function.vector_value(fe_face_values.quadrature_point(q),
									  u_true_side_values);
			      Tensor<1,dim> u_old_old_side, u_old_side, u_star_side, u_true_side;
			      for (unsigned int d=0; d<dim; ++d)
				{
				  u_old_old_side[d] = old_old_solution_side_values[q](d);
				  u_old_side[d] = old_solution_side_values[q](d);
				  u_star_side[d] = u_star_side_values[q](d);
				  u_true_side[d] = u_true_side_values(d);
				}
			      for (unsigned int i=0; i<dofs_per_cell; ++i)
				{
				  for (unsigned int j=0; j<dofs_per_cell; ++j)
				    {
				      if (enum_==state)
					{
					  if (fem_properties.newton)
					    {
					      local_matrix(i,j) += 0.5 * pow(fluid_theta,2) * physical_properties.rho_f 
						*( 
						  (fe_face_values[velocities].value (j, q)*fe_face_values.normal_vector(q))*(u_star_side*fe_face_values[velocities].value (i, q))
						  +(u_star_side*fe_face_values.normal_vector(q))*(fe_face_values[velocities].value (j, q)*fe_face_values[velocities].value (i, q))
						   ) * fe_face_values.JxW(q);
					    }
					  else if (fem_properties.richardson)
					    {
					      local_matrix(i,j) += 0.5 * pow(fluid_theta,2) * physical_properties.rho_f
						*(
						  ((4./3*u_old_old_side-1./3*u_old_side)*fe_face_values.normal_vector(q))*(fe_face_values[velocities].value (j, q)*fe_face_values[velocities].value (i, q))
						  ) * fe_face_values.JxW(q);
					    }
					  else
					    {
					      local_matrix(i,j) += 0.5 * pow(fluid_theta,2) * physical_properties.rho_f 
						*( 
						  (u_star_side*fe_face_values.normal_vector(q))*(fe_face_values[velocities].value (j, q)*fe_face_values[velocities].value (i, q))
						   ) * fe_face_values.JxW(q);

					    }
					}
				      else if (enum_==adjoint) 
					{
					  local_matrix(i,j) += 0.5 * pow(fluid_theta,2) * physical_properties.rho_f 
					    *( 
					      (fe_face_values[velocities].value (i, q)*fe_face_values.normal_vector(q))*(u_star_side*fe_face_values[velocities].value (j, q))
					      +(u_star_side*fe_face_values.normal_vector(q))*(fe_face_values[velocities].value (i, q)*fe_face_values[velocities].value (j, q))
					       ) * fe_face_values.JxW(q);
					}
				      else // enum_==linear
					{
					  local_matrix(i,j) += 0.5 * pow(fluid_theta,2) * physical_properties.rho_f 
					    *( 
					      (fe_face_values[velocities].value (j, q)*fe_face_values.normal_vector(q))*(u_star_side*fe_face_values[velocities].value (i, q))
					      +(u_star_side*fe_face_values.normal_vector(q))*(fe_face_values[velocities].value (j, q)*fe_face_values[velocities].value (i, q))
					       ) * fe_face_values.JxW(q);
					}
				    }
				  if (fem_properties.newton) 
				    {
				      local_rhs(i) += 0.5 * pow(fluid_theta,2) * physical_properties.rho_f 
					*(
					  (u_star_side*fe_face_values.normal_vector(q))*(u_star_side*fe_face_values[velocities].value (i, q))
					  ) * fe_face_values.JxW(q);
				    }
				}
			    }
			}
		      for (unsigned int q=0; q<n_face_q_points; ++q)
			{
			  fluid_stress_values.set_time(time);
			  fluid_stress_values.vector_gradient(fe_face_values.quadrature_point(q),
							      stress_values);
			  for (unsigned int i=0; i<dofs_per_cell; ++i)
			    {
			      Tensor<2,dim> new_stresses;
			      new_stresses[0][0]=stress_values[0][0];
			      new_stresses[1][0]=stress_values[1][0];
			      new_stresses[1][1]=stress_values[1][1];
			      new_stresses[0][1]=stress_values[0][1];
			      local_rhs(i) += fluid_theta*(fe_face_values[velocities].value (i, q)*
							   new_stresses*fe_face_values.normal_vector(q) *
							   fe_face_values.JxW(q));
			    }
			  fluid_stress_values.set_time(time-time_step);
			  fluid_stress_values.vector_gradient(fe_face_values.quadrature_point(q),
							      stress_values);
			  for (unsigned int i=0; i<dofs_per_cell; ++i)
			    {
			      Tensor<2,dim> new_stresses;
			      new_stresses[0][0]=stress_values[0][0];
			      new_stresses[1][0]=stress_values[1][0];
			      new_stresses[1][1]=stress_values[1][1];
			      new_stresses[0][1]=stress_values[0][1];
			      local_rhs(i) += (1-fluid_theta)*(fe_face_values[velocities].value (i, q)*
							       new_stresses*fe_face_values.normal_vector(q) *
							       fe_face_values.JxW(q));
			    }
			}
		    }
		}
	      else if (fluid_boundaries[cell->face(face_no)->boundary_indicator()]==Interface)
		{
		  if (enum_==state)
		    {
		      fe_face_values.reinit (cell, face_no);
		      if ((!physical_properties.moving_domain && physical_properties.navier_stokes) && physical_properties.stability_terms)
			{
			  fe_face_values.get_function_values (old_old_solution.block(0), old_old_solution_side_values);
			  fe_face_values.get_function_values (old_solution.block(0), old_solution_side_values);
			  fe_face_values.get_function_values (solution_star.block(0), u_star_side_values);
			  for (unsigned int q=0; q<n_face_q_points; ++q)
			    {
			      fluid_boundary_values_function.vector_value(fe_face_values.quadrature_point(q),
									  u_true_side_values);
			      Tensor<1,dim> u_old_old_side, u_old_side, u_star_side, u_true_side;
			      for (unsigned int d=0; d<dim; ++d)
				{
				  u_old_old_side[d] = old_old_solution_side_values[q](d);
				  u_old_side[d] = old_solution_side_values[q](d);
				  u_star_side[d] = u_star_side_values[q](d);
				  u_true_side[d] = u_true_side_values(d);
				}
			      for (unsigned int i=0; i<dofs_per_cell; ++i)
				{
				  for (unsigned int j=0; j<dofs_per_cell; ++j)
				    {
				      if (enum_==state)
					{
					  if (fem_properties.newton)
					    {
					      local_matrix(i,j) += 0.5 * pow(fluid_theta,2) * physical_properties.rho_f 
						*( 
						  (fe_face_values[velocities].value (j, q)*fe_face_values.normal_vector(q))*(u_star_side*fe_face_values[velocities].value (i, q))
						  +(u_star_side*fe_face_values.normal_vector(q))*(fe_face_values[velocities].value (j, q)*fe_face_values[velocities].value (i, q))
						   ) * fe_face_values.JxW(q);
					    }
					  else if (fem_properties.richardson)
					    {
					      local_matrix(i,j) += 0.5 * pow(fluid_theta,2) * physical_properties.rho_f
						*(
						  ((4./3*u_old_old_side-1./3*u_old_side)*fe_face_values.normal_vector(q))*(fe_face_values[velocities].value (j, q)*fe_face_values[velocities].value (i, q))
						  ) * fe_face_values.JxW(q);
					    }
					  else
					    {
					      local_matrix(i,j) += 0.5 * pow(fluid_theta,2) * physical_properties.rho_f 
						*( 
						  (u_star_side*fe_face_values.normal_vector(q))*(fe_face_values[velocities].value (j, q)*fe_face_values[velocities].value (i, q))
						   ) * fe_face_values.JxW(q);

					    }
					}
				      else if (enum_==adjoint) 
					{
					  local_matrix(i,j) += 0.5 * pow(fluid_theta,2) * physical_properties.rho_f 
					    *( 
					      (fe_face_values[velocities].value (i, q)*fe_face_values.normal_vector(q))*(u_star_side*fe_face_values[velocities].value (j, q))
					      +(u_star_side*fe_face_values.normal_vector(q))*(fe_face_values[velocities].value (i, q)*fe_face_values[velocities].value (j, q))
					       ) * fe_face_values.JxW(q);
					}
				      else // enum_==linear
					{
					  local_matrix(i,j) += 0.5 * pow(fluid_theta,2) * physical_properties.rho_f 
					    *( 
					      (fe_face_values[velocities].value (j, q)*fe_face_values.normal_vector(q))*(u_star_side*fe_face_values[velocities].value (i, q))
					      +(u_star_side*fe_face_values.normal_vector(q))*(fe_face_values[velocities].value (j, q)*fe_face_values[velocities].value (i, q))
					       ) * fe_face_values.JxW(q);
					}
				    }
				  if (physical_properties.navier_stokes && physical_properties.stability_terms)
				    {
				      if (fem_properties.newton) 
					{
					  local_rhs(i) += 0.5 * pow(fluid_theta,2) * physical_properties.rho_f 
					    *(
					      (u_star_side*fe_face_values.normal_vector(q))*(u_star_side*fe_face_values[velocities].value (i, q))
					      ) * fe_face_values.JxW(q);
					}
				    }
				}
			    }
			}

		      fe_face_values.get_function_values (stress.block(0), g_stress_values);
		      for (unsigned int q=0; q<n_face_q_points; ++q)
			{
			  Tensor<1,dim> g_stress;
			  for (unsigned int d=0; d<dim; ++d)
			    g_stress[d] = g_stress_values[q](d);
			  for (unsigned int i=0; i<dofs_per_cell; ++i)
			    {
			      local_rhs(i) += fluid_theta*(fe_face_values[velocities].value (i, q)*
							   g_stress * fe_face_values.JxW(q));
			    }
			}

		      fe_face_values.get_function_values (old_stress.block(0), g_stress_values);
		      for (unsigned int q=0; q<n_face_q_points; ++q)
			{
			  Tensor<1,dim> g_stress;
			  for (unsigned int d=0; d<dim; ++d)
			    g_stress[d] = g_stress_values[q](d);
			  for (unsigned int i=0; i<dofs_per_cell; ++i)
			    {
			      local_rhs(i) += (1-fluid_theta)*(fe_face_values[velocities].value (i, q)*
							       g_stress * fe_face_values.JxW(q));
			    }
			}
		    }
		  else if (enum_==adjoint)
		    {
		      fe_face_values.reinit (cell, face_no);
		      fe_face_values.get_function_values (rhs_for_adjoint.block(0), adjoint_rhs_values);

		      for (unsigned int q=0; q<n_face_q_points; ++q)
			{
			  Tensor<1,dim> r;
			  for (unsigned int d=0; d<dim; ++d)
			    r[d] = adjoint_rhs_values[q](d);
			  for (unsigned int i=0; i<dofs_per_cell; ++i)
			    {
			      local_rhs(i) += fluid_theta*(fe_face_values[velocities].value (i, q)*
							   r * fe_face_values.JxW(q));
			    }
			  length += fe_face_values.JxW(q);
			  residual += 0.5 * r * r * fe_face_values.JxW(q);
			}
		    }
		  else // enum_==linear
		    {
		      fe_face_values.reinit (cell, face_no);
		      fe_face_values.get_function_values (rhs_for_linear.block(0), linear_rhs_values);

		      for (unsigned int q=0; q<n_face_q_points; ++q)
			{
			  Tensor<1,dim> h;
			  for (unsigned int d=0; d<dim; ++d)
			    h[d] = linear_rhs_values[q](d);
			  for (unsigned int i=0; i<dofs_per_cell; ++i)
			    {
			      local_rhs(i) += fluid_theta*(fe_face_values[velocities].value (i, q)*
							   h * fe_face_values.JxW(q));
			    }
			}
		    }
		}
	      else if (fluid_boundaries[cell->face(face_no)->boundary_indicator()]==Dirichlet)
		{
		  if (physical_properties.navier_stokes && physical_properties.stability_terms)
		    {
		      fe_face_values.reinit (cell, face_no);
		      for (unsigned int q=0; q<n_face_q_points; ++q)
			{
			  fluid_boundary_values_function.vector_value(fe_face_values.quadrature_point(q),
								      u_true_side_values);
			  Tensor<1,dim> u_true_side;
			  for (unsigned int d=0; d<dim; ++d)
			    {
			      u_true_side[d] = u_true_side_values(d);
			    }
			  for (unsigned int i=0; i<dofs_per_cell; ++i)
			    {
			      local_rhs(i) += 0.5 * pow(fluid_theta,2) * physical_properties.rho_f 
				*(
				  (u_true_side*fe_face_values.normal_vector(q))*(u_true_side*fe_face_values[velocities].value (i, q))
				  ) * fe_face_values.JxW(q);
			    }
			}
		    }
		}
	    }
	}
      cell->get_dof_indices (local_dof_indices);
      if (assemble_matrix)
	{
	  fluid_constraints.distribute_local_to_global (local_matrix, local_rhs,
							local_dof_indices,
							*fluid_matrix, *fluid_rhs);
	}
      else
	{
	  fluid_constraints.distribute_local_to_global (local_rhs,
							local_dof_indices,
							*fluid_rhs);
	}
    }
}

template void FSIProblem<2>::assemble_fluid (Mode enum_, bool assemble_matrix);
