#include "FSI_Project.h"

template <int dim>
void FSIProblem<dim>::assemble_ale (Mode enum_, bool assemble_matrix)
{
  SparseMatrix<double> *ale_matrix;
  Vector<double> *ale_rhs;
  if (enum_==state)
    {
      ale_matrix = &system_matrix.block(2,2);
      ale_rhs = &system_rhs.block(2);
    }
  else if (enum_==adjoint)
    {
      ale_matrix = &adjoint_matrix.block(2,2);
      ale_rhs = &adjoint_rhs.block(2);
    }
  else
    {
      ale_matrix = &linear_matrix.block(2,2);
      ale_rhs = &linear_rhs.block(2);
    }

  if (assemble_matrix)
    {
      *ale_matrix=0;
    }
  *ale_rhs=0;
  QGauss<dim>   quadrature_formula(fem_properties.fluid_degree+2);
  FEValues<dim> fe_values (ale_fe, quadrature_formula,
			   update_values   | update_gradients |
			   update_quadrature_points | update_JxW_values);
  const unsigned int   dofs_per_cell = ale_fe.dofs_per_cell;
  const unsigned int   n_q_points    = quadrature_formula.size();
  FullMatrix<double>   local_matrix (dofs_per_cell, dofs_per_cell);
  Vector<double>       local_rhs (dofs_per_cell);
  std::vector<types::global_dof_index> local_dof_indices (dofs_per_cell);
  std::vector<Tensor<2,dim> > 	grad_phi_n (dofs_per_cell);

  const FEValuesExtractors::Vector displacements (0);
  typename DoFHandler<dim>::active_cell_iterator cell = ale_dof_handler.begin_active(),
    endc = ale_dof_handler.end();
  for (; cell!=endc; ++cell)
    {
      fe_values.reinit (cell);
      local_matrix = 0;
      local_rhs = 0;
      for (unsigned int q_point=0; q_point<n_q_points;
	   ++q_point)
	{
	  for (unsigned int k=0; k<dofs_per_cell; ++k)
	    {
	      grad_phi_n[k] = fe_values[displacements].gradient(k, q_point);
	    }
	  for (unsigned int i=0; i<dofs_per_cell; ++i)
	    {
	      for (unsigned int j=0; j<dofs_per_cell; ++j)
		{
		  local_matrix(i,j)+=scalar_product(grad_phi_n[i],grad_phi_n[j])*fe_values.JxW(q_point);
		}
	    }
	}
      cell->get_dof_indices (local_dof_indices);
      if (assemble_matrix)
	{
	  ale_constraints.distribute_local_to_global (local_matrix, local_rhs,
						      local_dof_indices,
						      *ale_matrix, *ale_rhs);
	}
      else
	{
	  ale_constraints.distribute_local_to_global (local_rhs,
						      local_dof_indices,
						      *ale_rhs);
	}

    }
}


template void FSIProblem<2>::assemble_ale (Mode enum_, bool assemble_matrix);
