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
Tensor<1,dim> FSIProblem<dim>::lift_and_drag_fluid()
{
  AssertThrow(physical_properties.simulation_type==3, ExcNotImplemented());
  const FEValuesExtractors::Vector velocities (0);
  const FEValuesExtractors::Scalar pressure (dim);

  QGauss<dim-1> face_quadrature_formula(fem_properties.fluid_degree+2);
  FEFaceValues<dim> fe_face_values (fluid_fe, face_quadrature_formula,
				    update_values    | update_normal_vectors | update_gradients |
				    update_quadrature_points  | update_JxW_values);
  const unsigned int   n_face_q_points = face_quadrature_formula.size();

  std::vector<Tensor<2,dim> > grad_u(n_face_q_points);
  std::vector<double> p(n_face_q_points);

  Tensor<1,dim> functional;
    Tensor<1,dim> temp_functional;
  Tensor<2,dim> Identity;
  AssertThrow(dim==2,ExcNotImplemented());
  for (unsigned int i=0; i<2; ++i) Identity[i][i]=1;

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
	      // std::cout << (unsigned int)(cell->face(face_no)->boundary_indicator()) << std::endl;
	      // std::cout << cell->face(face_no)->center() << std::endl;

	      double multiplier=1;
	      unsigned int bc = (unsigned int)(cell->face(face_no)->boundary_indicator());
	      //if (bc<8) multiplier *= -1;
	      if ((unsigned int)(cell->face(face_no)->boundary_indicator())>=5 /* circle + interface */) // <---- For fluid tests
		//if (cell->face(face_no)->boundary_indicator()==8 /* circle + interface */) // <---- The good one
		//if (fluid_boundaries[cell->face(face_no)->boundary_indicator()]==Interface)
		{
		  fe_face_values.reinit (cell, face_no);
		  fe_face_values[velocities].get_function_gradients(solution.block(0),grad_u);
		  fe_face_values[pressure].get_function_values(solution.block(0),p);
		  for (unsigned int q=0; q<n_face_q_points; ++q)
		    {
		      // std::cout << physical_properties.viscosity << std::endl;
		      // std::cout << "grad_u[q] " << grad_u[q] << std::endl;
		      // std::cout << "p[q] " << p[q] << std::endl;
		      temp_functional = multiplier * (2*physical_properties.viscosity*.5*(transpose(grad_u[q]) + grad_u[q]) - p[q]*Identity) * fe_face_values.normal_vector(q) * fe_face_values.JxW(q); 
		      // std::cout << "normal: " << fe_face_values.normal_vector(q) << std::endl;
		      // std::cout << "center " <<  cell->face(face_no)->center() << std::endl;
		      // std::cout << (unsigned int)(cell->face(face_no)->boundary_indicator()) << std::endl;
		      // temp_functional[0]=std::abs(temp_functional[0]);
		      // temp_functional[1]=std::abs(temp_functional[1]);
		      functional += temp_functional;
		    }
		}
	    }
	}
    }
  return functional;
}

template <int dim>
Tensor<1,dim> FSIProblem<dim>::lift_and_drag_structure()
{
  AssertThrow(physical_properties.simulation_type==3, ExcNotImplemented());
  const FEValuesExtractors::Vector displacements (0);

  QGauss<dim-1> face_quadrature_formula(fem_properties.structure_degree+2);
  FEFaceValues<dim> fe_face_values (structure_fe, face_quadrature_formula,
				    update_values    | update_normal_vectors |
				    update_quadrature_points  | update_JxW_values);
  const unsigned int   n_face_q_points = face_quadrature_formula.size();

  std::vector<Tensor<2,dim> > grad_n(n_face_q_points);
  std::vector<Tensor<2,dim> > F(n_face_q_points);
  std::vector<Tensor<2,dim> > E(n_face_q_points);
  

  Tensor<1,dim> functional;

  typename DoFHandler<dim>::active_cell_iterator
    cell = structure_dof_handler.begin_active(),
    endc = structure_dof_handler.end();
  for (; cell!=endc; ++cell)
    {
      for (unsigned int face_no=0;
	   face_no<GeometryInfo<dim>::faces_per_cell;
	   ++face_no)
	{
	  if (cell->at_boundary(face_no))
	    {
	      if (cell->face(face_no)->boundary_indicator()>=1 && cell->face(face_no)->boundary_indicator()<=4 /* interface */)
		{
		  fe_face_values.reinit (cell, face_no);
		  fe_face_values[displacements].get_function_gradients(solution.block(1),grad_n);

		  for (unsigned int q=0; q<n_face_q_points; ++q)
		    {
		      Tensor<2,dim> F = grad_n[q];
		      Tensor<2,dim> Identity;
		      for (unsigned int k=0; k<dim; ++k) {
			Identity[k][k] = 1;
		      }
		      F += Identity;
		      Tensor<2,dim> E = .5*(transpose(F)*F - Identity);
		      Tensor<2,dim> S = physical_properties.lambda*trace(E)*Identity + 2*physical_properties.mu*E;

		      functional += (F*S) * fe_face_values.normal_vector(q) * fe_face_values.JxW(q); 
		    }
		}
	    }
	}
    }
  return functional;
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
template Tensor<1,2> FSIProblem<2>::lift_and_drag_fluid();
template Tensor<1,2> FSIProblem<2>::lift_and_drag_structure();
template double FSIProblem<2>::interface_error();
template double FSIProblem<2>::interface_norm(Vector<double>   &values);
