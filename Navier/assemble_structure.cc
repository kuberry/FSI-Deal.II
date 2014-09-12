#include "FSI_Project.h"

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
      *structure_matrix=0;
    }
  *structure_rhs=0;

  Vector<double> tmp;
  Vector<double> forcing_terms;

  tmp.reinit (structure_rhs->size());
  forcing_terms.reinit (structure_rhs->size());

  tmp=0;
  forcing_terms=0;

  QGauss<dim>   quadrature_formula(fem_properties.structure_degree+2);
  FEValues<dim> fe_values (structure_fe, quadrature_formula,
			   update_values   | update_gradients |
			   update_quadrature_points | update_JxW_values);

  QGauss<dim-1> face_quadrature_formula(fem_properties.structure_degree+2);
  FEFaceValues<dim> fe_face_values (structure_fe, face_quadrature_formula,
				    update_values    | update_normal_vectors |
				    update_quadrature_points  | update_JxW_values);

  const unsigned int   dofs_per_cell = structure_fe.dofs_per_cell;
  const unsigned int   n_q_points    = quadrature_formula.size();
  const unsigned int   n_face_q_points = face_quadrature_formula.size();
  FullMatrix<double>   local_matrix (dofs_per_cell, dofs_per_cell);
  Vector<double>       local_rhs (dofs_per_cell);
  std::vector<types::global_dof_index> local_dof_indices (dofs_per_cell);

  std::vector<Vector<double> > old_solution_values(n_q_points, Vector<double>(2*dim));
  std::vector<Vector<double> > adjoint_rhs_values(n_face_q_points, Vector<double>(2*dim));
  std::vector<Vector<double> > linear_rhs_values(n_face_q_points, Vector<double>(2*dim));
  std::vector<Tensor<2,dim> > grad_n (n_q_points);

  if (enum_==state)
    {
      StructureRightHandSide<dim> rhs_function(physical_properties);
      rhs_function.set_time(time);
      VectorTools::create_right_hand_side(structure_dof_handler,
					  QGauss<dim>(structure_fe.degree+2),
					  rhs_function,
					  tmp);
      forcing_terms = tmp;
      forcing_terms *= 0.5;
      rhs_function.set_time(time - time_step);
      VectorTools::create_right_hand_side(structure_dof_handler,
					  QGauss<dim>(structure_fe.degree+2),
					  rhs_function,
					  tmp);
      forcing_terms.add(0.5, tmp);
      *structure_rhs += forcing_terms;
    }

  StructureStressValues<dim> structure_stress_values(physical_properties);
  std::vector<Tensor<1,dim> > stress_values (3);
  std::vector<Vector<double> > g_stress_values(n_face_q_points, Vector<double>(2*dim));

  std::vector<Tensor<1,dim> > 		phi_n (dofs_per_cell);
  std::vector<SymmetricTensor<2,dim> > 	symgrad_phi_n (dofs_per_cell);
  std::vector<double>                  	div_phi_n   (dofs_per_cell);
  std::vector<Tensor<1,dim> >           	phi_v       (dofs_per_cell);

  const FEValuesExtractors::Vector displacements (0);
  const FEValuesExtractors::Vector velocities (dim);
  typename DoFHandler<dim>::active_cell_iterator cell = structure_dof_handler.begin_active(),
    endc = structure_dof_handler.end();
  for (; cell!=endc; ++cell)
    {
      fe_values.reinit (cell);
      local_matrix = 0;
      local_rhs = 0;
      if (assemble_matrix)
	{
	  fe_values.get_function_values (old_solution.block(1), old_solution_values);
	  fe_values[displacements].get_function_gradients(old_solution.block(1),grad_n);
	  for (unsigned int q_point=0; q_point<n_q_points;
	       ++q_point)
	    {
	      for (unsigned int k=0; k<dofs_per_cell; ++k)
		{
		  phi_n[k]		   = fe_values[displacements].value (k, q_point);
		  symgrad_phi_n[k] = fe_values[displacements].symmetric_gradient (k, q_point);
		  div_phi_n[k]     = fe_values[displacements].divergence (k, q_point);
		  phi_v[k]         = fe_values[velocities].value (k, q_point);
		}
	      for (unsigned int i=0; i<dofs_per_cell; ++i)
		{
		  const unsigned int
		    component_i = structure_fe.system_to_component_index(i).first;
		  for (unsigned int j=0; j<dofs_per_cell; ++j)
		    {
		      const unsigned int
			component_j = structure_fe.system_to_component_index(j).first;

		      if (enum_==state || enum_==linear)
			{

			  if (component_i<dim)
			    {
			      if (component_j<dim)
				{
				  local_matrix(i,j)+=(.5*	( 2*physical_properties.mu*symgrad_phi_n[i] * symgrad_phi_n[j]
								  + physical_properties.lambda*div_phi_n[i] * div_phi_n[j]))
				    *fe_values.JxW(q_point);
				}
			      else
				{
				  local_matrix(i,j)+=physical_properties.rho_s/time_step*phi_n[i]*phi_v[j]*fe_values.JxW(q_point);
				}
			    }
			  else
			    {
			      if (component_j<dim)
				{
				  local_matrix(i,j)+=(-1./time_step*phi_v[i]*phi_n[j])
				    *fe_values.JxW(q_point);
				}
			      else
				{
				  local_matrix(i,j)+=(0.5*phi_v[i]*phi_v[j])
				    *fe_values.JxW(q_point);
				}
			    }
			}
		      else // enum_==adjoint
			{
			  if (component_i<dim)
			    {
			      if (component_j<dim)
				{
				  local_matrix(i,j)+=(.5*	( 2*physical_properties.mu*symgrad_phi_n[i] * symgrad_phi_n[j]
								  + physical_properties.lambda*div_phi_n[i] * div_phi_n[j]))
				    *fe_values.JxW(q_point);
				}
			      else
				{
				  local_matrix(i,j)+=-1./time_step*phi_n[i]*phi_v[j]*fe_values.JxW(q_point);
				}
			    }
			  else
			    {
			      if (component_j<dim)
				{
				  local_matrix(i,j)+=physical_properties.rho_s/time_step*phi_v[i]*phi_n[j]*fe_values.JxW(q_point);
				}
			      else
				{
				  local_matrix(i,j)+=(0.5*phi_v[i]*phi_v[j])
				    *fe_values.JxW(q_point);
				}
			    }
			}
		    }
		}
	      if (enum_==state)
		{
		  for (unsigned int i=0; i<dofs_per_cell; ++i)
		    {
		      const unsigned int component_i = structure_fe.system_to_component_index(i).first;
		      Tensor<1,dim> old_n;
		      Tensor<1,dim> old_v;
		      for (unsigned int d=0; d<dim; ++d)
			old_n[d] = old_solution_values[q_point](d);
		      for (unsigned int d=0; d<dim; ++d)
			old_v[d] = old_solution_values[q_point](d+dim);
		      const Tensor<1,dim> phi_i_eta      	= fe_values[displacements].value (i, q_point);
		      const Tensor<2,dim> symgrad_phi_i_eta 	= fe_values[displacements].symmetric_gradient (i, q_point);
		      const double div_phi_i_eta 			= fe_values[displacements].divergence (i, q_point);
		      const Tensor<1,dim> phi_i_eta_dot  	= fe_values[velocities].value (i, q_point);
		      if (component_i<dim)
			{
			  local_rhs(i) += (physical_properties.rho_s/time_step *phi_i_eta*old_v
					   +0.5*(-2*physical_properties.mu*(scalar_product(grad_n[q_point],symgrad_phi_i_eta))
						 -physical_properties.lambda*((grad_n[q_point][0][0]+grad_n[q_point][1][1])*div_phi_i_eta))
					   )
			    * fe_values.JxW(q_point);
			}
		      else
			{
			  local_rhs(i) += (-0.5*phi_i_eta_dot*old_v
					   -1./time_step*phi_i_eta_dot*old_n
					   )
			    * fe_values.JxW(q_point);
			}
		    }
		}
	    }
	}
      unsigned int total_loops;
      if (enum_==state)
	{
	  total_loops = 2;
	}
      else
	{
	  total_loops = 1;
	}
      for (unsigned int i=0; i<total_loops; ++i)
	{
	  double multiplier;
	  Vector<double> *stress_vector;
	  if (i==0)
	    {
	      structure_stress_values.set_time(time);
	      multiplier=structure_theta;
	      stress_vector=&stress.block(1);
	    }
	  else
	    {
	      structure_stress_values.set_time(time-time_step);
	      multiplier=(1-structure_theta);
	      stress_vector=&old_stress.block(1);
	    }

	  for (unsigned int face_no=0;
	       face_no<GeometryInfo<dim>::faces_per_cell;
	       ++face_no)
	    {
	      if (cell->at_boundary(face_no))
		{
		  if (structure_boundaries[cell->face(face_no)->boundary_indicator()]==Neumann)
		    {
		      if (enum_==state)
			{
			  fe_face_values.reinit (cell, face_no);
			  // GET SIDE ID!

			  for (unsigned int q=0; q<n_face_q_points; ++q)
			    for (unsigned int i=0; i<dofs_per_cell; ++i)
			      {
				structure_stress_values.vector_gradient(fe_face_values.quadrature_point(q),
									stress_values);
				Tensor<2,dim> new_stresses;
				new_stresses[0][0]=stress_values[0][0];
				new_stresses[1][0]=stress_values[1][0];
				new_stresses[1][1]=stress_values[1][1];
				new_stresses[0][1]=stress_values[0][1];
				local_rhs(i) += multiplier*(fe_face_values[displacements].value (i, q)*
							    new_stresses*fe_face_values.normal_vector(q) *
							    fe_face_values.JxW(q));
			      }
			}
		    }
		  else if (structure_boundaries[cell->face(face_no)->boundary_indicator()]==Interface)
		    {
		      if (enum_==state)
			{
			  fe_face_values.reinit (cell, face_no);
			  fe_face_values.get_function_values (*stress_vector, g_stress_values);

			  for (unsigned int q=0; q<n_face_q_points; ++q)
			    {
			      Tensor<1,dim> g_stress;
			      for (unsigned int d=0; d<dim; ++d)
				g_stress[d] = g_stress_values[q](d);
			      for (unsigned int i=0; i<dofs_per_cell; ++i)
				{
				  local_rhs(i) += multiplier*(fe_face_values[displacements].value (i, q)*
							      (-g_stress) * fe_face_values.JxW(q));
				}
			    }
			}
		      else if (enum_==adjoint)
			{
			  fe_face_values.reinit (cell, face_no);
			  fe_face_values.get_function_values (rhs_for_adjoint.block(1), adjoint_rhs_values);
			  for (unsigned int q=0; q<n_face_q_points; ++q)
			    {
			      Tensor<1,dim> r;
			      if (fem_properties.adjoint_type==1)
				{
				  for (unsigned int d=0; d<dim; ++d)
				    r[d] = adjoint_rhs_values[q](d);
				  for (unsigned int i=0; i<dofs_per_cell; ++i)
				    {
				      local_rhs(i) += structure_theta*(fe_face_values[displacements].value (i, q)*
								       r * fe_face_values.JxW(q));
				    }
				}
			      else
				{
				  if (fem_properties.optimization_method.compare("Gradient")==0)
				    {
				      for (unsigned int d=0; d<dim; ++d)
					r[d] = adjoint_rhs_values[q](d+dim);
				      for (unsigned int i=0; i<dofs_per_cell; ++i)
					{
					  local_rhs(i) += structure_theta*(fe_face_values[velocities].value (i, q)*
									   r * fe_face_values.JxW(q));
					}
				    }
				  else
				    {
				      for (unsigned int d=0; d<dim; ++d)
					r[d] = adjoint_rhs_values[q](d+dim);
				      for (unsigned int i=0; i<dofs_per_cell; ++i)
					{
					  local_rhs(i) += structure_theta*(fe_face_values[velocities].value (i, q)*
									   r * fe_face_values.JxW(q));
					}
				    }
				}

			    }
			}
		      else // enum_==linear
			{
			  fe_face_values.reinit (cell, face_no);
			  fe_face_values.get_function_values (rhs_for_linear.block(1), linear_rhs_values);
			  for (unsigned int q=0; q<n_face_q_points; ++q)
			    {
			      Tensor<1,dim> h;
			      for (unsigned int d=0; d<dim; ++d)
				h[d] = linear_rhs_values[q](d);
			      for (unsigned int i=0; i<dofs_per_cell; ++i)
				{
				  local_rhs(i) += structure_theta*(fe_face_values[displacements].value (i, q)*
								   h * fe_face_values.JxW(q));
				}
			    }
			}
		    }
		}
	    }
	}
      cell->get_dof_indices (local_dof_indices);
      if (assemble_matrix)
	{
	  structure_constraints.distribute_local_to_global (local_matrix, local_rhs,
							    local_dof_indices,
							    *structure_matrix, *structure_rhs);
	}
      else
	{
	  structure_constraints.distribute_local_to_global (local_rhs,
							    local_dof_indices,
							    *structure_rhs);
	}
    }
}


template void FSIProblem<2>::assemble_structure (Mode enum_, bool assemble_matrix);
