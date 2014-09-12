#include "FSI_Project.h"
#include "small_classes.h"


template <int dim>
void FSIProblem<dim>::assemble_ale_matrix_on_one_cell (const typename DoFHandler<dim>::active_cell_iterator& cell,
						       ScratchData<dim>& scratch,
						       PerTaskData<dim>& data
				      )//,
				      // unsigned int n_q_points,
				      // unsigned int dofs_per_cell)
{
  // // rhs_function.value_list (scratch.fe_values.get_quadrature_points,
  // // 			   scratch.rhs_values);
  // const FEValuesExtractors::Vector displacements (0);
  // std::vector<Tensor<2,dim> > 	grad_phi_n (dofs_per_cell);
  // scratch.fe_values.reinit(cell);
  // for (unsigned int q_point=0; q_point<n_q_points;
  //      ++q_point)
  //   {
  //     for (unsigned int k=0; k<dofs_per_cell; ++k)
  // 	{
  // 	  grad_phi_n[k] = scratch.fe_values[displacements].gradient(k, q_point);
  // 	}
  //     for (unsigned int i=0; i<dofs_per_cell; ++i)
  // 	{
  // 	  for (unsigned int j=0; j<dofs_per_cell; ++j)
  // 	    {
  // 	      data.cell_matrix(i,j)+=scalar_product(grad_phi_n[i],grad_phi_n[j])*scratch.fe_values.JxW(q_point);
  // 	    }
  // 	}
  //   }
}

template <int dim>
void FSIProblem<dim>::copy_local_matrix_to_global (const PerTaskData<dim>& data )//, 
//unsigned int dofs_per_cell) //, 
						   // SparseMatrix<double>* global_matrix, 
						   // Vector<double>* global_rhs)
{

  // for (unsigned int i=0; i<dofs_per_cell; ++i)
  //   for (unsigned int j=0; j<dofs_per_cell; ++j)
  //     global_matrix->add (data.dof_indices[i], data.dof_indices[j], data.cell_matrix(i,j));


  //for (unsigned int i=0; i<data.dofs_per_cell; ++i)
    // global_rhs->add (data.dof_indices, data.cell_rhs);
}


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

  // std::vector<types::global_dof_index> local_dof_indices (dofs_per_cell);


  PerTaskData<dim> per_task_data(ale_fe);//, assemble_matrix, ale_matrix, ale_rhs);
  ScratchData<dim> scratch_data(ale_fe, quadrature_formula, update_values | update_gradients | update_quadrature_points | update_JxW_values);

  if (assemble_matrix)
    {
    //   WorkStream::run (ale_dof_handler.begin_active(),
    // 		       ale_dof_handler.end(),
    // 		       std_cxx1x::bind(&assemble_ale_matrix_on_one_cell<dim>,
    // 				       std_cxx1x::_1,
    // 				       std_cxx1x::_3,
    // 				       std_cxx1x::_2,
    // 				       quadrature_formula.size(),
    // 				       ale_fe.dofs_per_cell
    // 				       ),
    // 		       std_cxx1x::bind(&copy_local_matrix_to_global<dim>,
    // 				       std_cxx1x::_3,
    // 				       ale_fe.dofs_per_cell
    // 				       ),
    // 		       scratch_data,
    // 		       per_task_data);
    // }

// ,
// 				       ale_matrix,
// 				       ale_rhs

  // WorkStream::run (ale_dof_handler.begin_active(),
  // 		   ale_dof_handler.end(),
  // 		   &FSIProblem<dim>::assemble_ale_matrix_on_one_cell,
  // 		   &FSIProblem<dim>::copy_local_matrix_to_global,
  // 		   per_task_data);

  WorkStream::run (ale_dof_handler.begin_active(),
  		   ale_dof_handler.end(),
  		   *this,
  		   &FSIProblem<dim>::assemble_ale_matrix_on_one_cell,
  		   &FSIProblem<dim>::copy_local_matrix_to_global,
		   scratch_data,
  		   per_task_data);
    }

}

// template void FSIProblem<2>::assemble_ale_matrix_on_one_cell (const typename DoFHandler<2>::active_cell_iterator &cell,
// 							      ScratchData<2> &scratch,
// 							      PerTaskData<2> &data);
// template void FSIProblem<2>::copy_local_matrix_to_global (const PerTaskData<2> &data);

// template struct PerTaskData<2>;

template void FSIProblem<2>::assemble_ale_matrix_on_one_cell (const typename DoFHandler<2>::active_cell_iterator &cell,
							      ScratchData<2> &scratch,
							      PerTaskData<2> &data );//,
							      // unsigned int n_q_points,
							      // unsigned int dofs_per_cell);
template void FSIProblem<2>::copy_local_matrix_to_global (const PerTaskData<2> &data);//, 
 //unsigned int dofs_per_cell);//, 
						 // SparseMatrix<double>* global_matrix, 
						 // Vector<double>* global_rhs);

template void FSIProblem<2>::assemble_ale (Mode enum_, bool assemble_matrix);
