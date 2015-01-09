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
void FSIProblem<dim>::get_fluid_stress()
{
  tmp.block(0)=0;
  AssertThrow(fem_properties.optimization_method.compare("DN")==0, ExcNotImplemented());
  const FEValuesExtractors::Vector velocities (0);
  const FEValuesExtractors::Scalar pressure (dim);

  QGauss<dim-1> face_quadrature_formula(fem_properties.fluid_degree+2);
  FEFaceValues<dim> fe_face_values (fluid_fe, face_quadrature_formula,
				    update_values    | update_normal_vectors | update_gradients |
				    update_quadrature_points  | update_JxW_values);
  const unsigned int   n_face_q_points = face_quadrature_formula.size();

  std::vector<Tensor<2,dim,double> > grad_u(n_face_q_points, Tensor<2,dim,double>());
  std::vector<double> p(n_face_q_points);

  Tensor<1,dim,double> functional;
  Tensor<2,dim,double> Identity;
  AssertThrow(dim==2,ExcNotImplemented());
  for (unsigned int i=0; i<2; ++i) Identity[i][i]=1;

  std::vector<types::global_dof_index> dof_indices(fluid_fe.dofs_per_cell);
  Vector<double> cell_rhs(fluid_fe.dofs_per_cell);

  typename DoFHandler<dim>::active_cell_iterator
    cell = fluid_dof_handler.begin_active(),
    endc = fluid_dof_handler.end();
  for (; cell!=endc; ++cell)
    {
      cell_rhs *= 0;
      for (unsigned int face_no=0;
	   face_no<GeometryInfo<dim>::faces_per_cell;
	   ++face_no)
	{
	  if (cell->at_boundary(face_no))
	    {
	      if (fluid_boundaries[cell->face(face_no)->boundary_indicator()]==Interface)
		{
		  fe_face_values.reinit (cell, face_no);
		  fe_face_values[velocities].get_function_gradients(solution.block(0),grad_u);
		  fe_face_values[pressure].get_function_values(solution.block(0),p);
		  for (unsigned int q=0; q<n_face_q_points; ++q)
		    {
		      for (unsigned int i=0; i<fluid_fe.dofs_per_cell; ++i)
			{
			  cell_rhs(i) += ((2*physical_properties.viscosity*.5*(transpose(grad_u[q]) + grad_u[q]) - p[q]*Identity) * fe_face_values.normal_vector(q) * fe_face_values[velocities].value (i, q)) * fe_face_values.JxW(q); 
			}
		    }
		}
	    }
	}
      cell->get_dof_indices (dof_indices);
      fluid_constraints.distribute_local_to_global (cell_rhs,
      						    dof_indices,
      						    tmp);
    }
}

template <int dim>
Tensor<1,dim,double> FSIProblem<dim>::lift_and_drag_fluid()
{
  AssertThrow(physical_properties.simulation_type==3, ExcNotImplemented());
  const FEValuesExtractors::Vector velocities (0);
  const FEValuesExtractors::Scalar pressure (dim);

  QGauss<dim-1> face_quadrature_formula(fem_properties.fluid_degree+2);
  FEFaceValues<dim> fe_face_values (fluid_fe, face_quadrature_formula,
				    update_values    | update_normal_vectors | update_gradients |
				    update_quadrature_points  | update_JxW_values);
  const unsigned int   n_face_q_points = face_quadrature_formula.size();

  std::vector<Tensor<2,dim,double> > grad_u(n_face_q_points, Tensor<2,dim,double>());
  std::vector<double> p(n_face_q_points);

  Tensor<1,dim,double> functional;
  Tensor<2,dim,double> Identity;
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
	      if ((unsigned int)(cell->face(face_no)->boundary_indicator())==8 /* circle + interface */) // <---- For fluid tests
		//if (cell->face(face_no)->boundary_indicator()==8 /* circle + interface */) // <---- The good one
		{
		  fe_face_values.reinit (cell, face_no);
		  fe_face_values[velocities].get_function_gradients(solution.block(0),grad_u);
		  fe_face_values[pressure].get_function_values(solution.block(0),p);
		  for (unsigned int q=0; q<n_face_q_points; ++q)
		    {
		      functional += (2*physical_properties.viscosity*.5*(transpose(grad_u[q]) + grad_u[q]) - p[q]*Identity) * fe_face_values.normal_vector(q) * fe_face_values.JxW(q); 
		    }
		}
	    }
	}
    }
  return functional;
}

template <int dim>
Tensor<1,dim,double> FSIProblem<dim>::lift_and_drag_structure()
{
  AssertThrow(physical_properties.simulation_type==3, ExcNotImplemented());
  const FEValuesExtractors::Vector displacements (0);

  QGauss<dim-1> face_quadrature_formula(fem_properties.structure_degree+2);
  FEFaceValues<dim> fe_face_values (structure_fe, face_quadrature_formula,
				    update_values    | update_normal_vectors | update_gradients |
				    update_quadrature_points  | update_JxW_values);
  const unsigned int   n_face_q_points = face_quadrature_formula.size();

  std::vector<Tensor<2,dim,double> > grad_n(n_face_q_points, Tensor<2,dim,double>());

  Tensor<1,dim,double> functional;
  Tensor<2,dim,double> Identity;
  AssertThrow(dim==2,ExcNotImplemented());
  for (unsigned int i=0; i<2; ++i) Identity[i][i]=1;

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
	      if ((unsigned int)(cell->face(face_no)->boundary_indicator())>=1 && (unsigned int)(cell->face(face_no)->boundary_indicator())<=3 /* interface */)
		{
		  fe_face_values.reinit (cell, face_no);
		  fe_face_values[displacements].get_function_gradients(solution.block(1),grad_n);

		  for (unsigned int q=0; q<n_face_q_points; ++q)
		    {
		      Tensor<2,dim,double> F = grad_n[q];
		      F += Identity;
		      Tensor<2,dim,double> E = .5*(transpose(F)*F - Identity);
		      Tensor<2,dim,double> S = physical_properties.lambda*trace(E)*Identity + 2*physical_properties.mu*E;

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
		  if (fem_properties.optimization_method.compare("DN")!=0) {
		    fe_face_values.get_function_values (stress.block(0), stress_values);
		  }

		  for (unsigned int q=0; q<n_face_q_points; ++q)
		    {
		      Tensor<1,dim,double> error;
		      Tensor<1,dim,double> g_stress;
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
		      Tensor<1,dim,double> pval, val;
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
template void FSIProblem<2>::get_fluid_stress();
template Tensor<1,2,double> FSIProblem<2>::lift_and_drag_fluid();
template Tensor<1,2,double> FSIProblem<2>::lift_and_drag_structure();
template double FSIProblem<2>::interface_error();
template double FSIProblem<2>::interface_norm(Vector<double>   &values);
