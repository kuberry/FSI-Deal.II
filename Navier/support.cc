#include "FSI_Project.h"

template <int dim>
void FSIProblem<dim>::build_adjoint_rhs()
{

  if (fem_properties.adjoint_type==1)
    {
      // here we build the rhs_for_adjoint vector from state variable information
      // build rhs of fluid adjoint problem
      // [u^n - (n^n-n^{n-1})/delta t]
      tmp=0;
      rhs_for_adjoint=0;
      transfer_interface_dofs(solution,rhs_for_adjoint,1,0,Displacement);
      rhs_for_adjoint.block(0)*=-1./time_step;
      transfer_interface_dofs(old_solution,tmp,1,0,Displacement);
      rhs_for_adjoint.block(0).add(1./time_step,tmp.block(0));
      tmp=0;
      transfer_interface_dofs(solution,tmp,0,0);
      rhs_for_adjoint.block(0)+=tmp.block(0);
      // build rhs of structure adjoint problem
      transfer_interface_dofs(rhs_for_adjoint,rhs_for_adjoint,0,1,Displacement);
      rhs_for_adjoint.block(1)*=-1./time_step;
    }
  else
    {
      rhs_for_adjoint=0;
      transfer_interface_dofs(solution,rhs_for_adjoint,1,0,Velocity);
      rhs_for_adjoint.block(0)*=-1;
      tmp=0;
      transfer_interface_dofs(solution,tmp,0,0);
      rhs_for_adjoint.block(0)+=tmp.block(0);
      // build rhs of structure adjoint problem
      transfer_interface_dofs(rhs_for_adjoint,rhs_for_adjoint,0,1,Velocity);
      rhs_for_adjoint.block(1)*=-1;
    }
}

template <int dim>
double FSIProblem<dim>::interface_error()
{
  QGauss<dim-1> face_quadrature_formula(fem_properties.fluid_degree+2);
  FEFaceValues<dim> fe_face_values (fluid_fe, face_quadrature_formula,
				    update_values    | update_normal_vectors |
				    update_quadrature_points  | update_JxW_values);
  const unsigned int   n_face_q_points = face_quadrature_formula.size();

  std::vector<Vector<double> > error_values(n_face_q_points, Vector<double>(dim+1));
  std::vector<Vector<double> > stress_values(n_face_q_points, Vector<double>(dim+1));

  double functional = 0;
  double penalty_functional = 0;

  typename DoFHandler<dim>::active_cell_iterator
    cell = fluid_dof_handler.begin_active(),
    endc = fluid_dof_handler.end();
  for (; cell!=endc; ++cell)
    {
      for (unsigned int face_no=0;
	   face_no<GeometryInfo<dim>::faces_per_cell;
	   ++face_no)
	{
	  if (cell->at_boundary(face_no))
	    {
	      if (fluid_boundaries[cell->face(face_no)->boundary_indicator()]==Interface)
		{
		  fe_face_values.reinit (cell, face_no);
		  fe_face_values.get_function_values (rhs_for_adjoint.block(0), error_values);
		  fe_face_values.get_function_values (stress.block(0), stress_values);

		  for (unsigned int q=0; q<n_face_q_points; ++q)
		    {
		      Tensor<1,dim> error;
		      Tensor<1,dim> g_stress;
		      for (unsigned int d=0; d<dim; ++d)
			{
			  error[d] = error_values[q](d);
			  g_stress[d] = stress_values[q](d);
			}
		      functional += 0.5 * error * error * fe_face_values.JxW(q);
		      penalty_functional += fem_properties.penalty_epsilon * 0.5 * g_stress * g_stress * fe_face_values.JxW(q); 
		    }
		}
	    }
	}
    }
  return functional+penalty_functional;
}

template <int dim>
double FSIProblem<dim>::interface_norm(Vector<double>   &values)
{
  QGauss<dim-1> face_quadrature_formula(fem_properties.fluid_degree+2);
  FEFaceValues<dim> fe_face_values (fluid_fe, face_quadrature_formula,
				    update_values    | update_normal_vectors |
				    update_quadrature_points  | update_JxW_values);
  const unsigned int   n_face_q_points = face_quadrature_formula.size();

  std::vector<Vector<double> > actual_values(n_face_q_points, Vector<double>(dim+1));
  std::vector<Vector<double> > premult_values(n_face_q_points, Vector<double>(dim+1));

  double functional = 0;

  typename DoFHandler<dim>::active_cell_iterator
    cell = fluid_dof_handler.begin_active(),
    endc = fluid_dof_handler.end();
  for (; cell!=endc; ++cell)
    {
      for (unsigned int face_no=0;
	   face_no<GeometryInfo<dim>::faces_per_cell;
	   ++face_no)
	{
	  if (cell->at_boundary(face_no))
	    {
	      if (fluid_boundaries[cell->face(face_no)->boundary_indicator()]==Interface)
		{
		  fe_face_values.reinit (cell, face_no);
		  fe_face_values.get_function_values (values, actual_values);
		  fe_face_values.get_function_values (premultiplier.block(0), premult_values);

		  for (unsigned int q=0; q<n_face_q_points; ++q)
		    {
		      Tensor<1,dim> pval, val;
		      for (unsigned int d=0; d<dim; ++d)
			{
			  pval[d] = premult_values[q](d);
			  val[d] = actual_values[q](d);
			}
		      functional += pval * val * fe_face_values.JxW(q); 
		    }
		}
	    }
	}
    }
  return functional;
}



template void FSIProblem<2>::build_adjoint_rhs();
template double FSIProblem<2>::interface_error();
template double FSIProblem<2>::interface_norm(Vector<double>   &values);
