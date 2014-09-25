#include "FSI_Project.h"
#include "small_classes.h"
#include <deal.II/base/timer.h>

template <int dim>
void FSIProblem<dim>::assemble_structure_matrix_on_one_cell (const typename DoFHandler<dim>::active_cell_iterator& cell,
						       FullScratchData<dim>& scratch,
						       PerTaskData<dim>& data )
{
  unsigned int state=0, adjoint=1, linear=2;

  //ConditionalOStream pcout(std::cout,Threads::this_thread_id()==scratch.master_thread); 
  //TimerOutput timer (pcout, TimerOutput::summary,
  //		     TimerOutput::wall_times); 
  //timer.enter_subsection ("Beginning");


  StructureStressValues<dim> structure_stress_values(physical_properties);
  structure_stress_values.set_time(time);
  StructureStressValues<dim> old_structure_stress_values(physical_properties);
  old_structure_stress_values.set_time(time-time_step);

  const FEValuesExtractors::Vector displacements (0);
  const FEValuesExtractors::Vector velocities (dim);

  std::vector<Vector<double> > old_solution_values(scratch.n_q_points, Vector<double>(2*dim));
  std::vector<Vector<double> > adjoint_rhs_values(scratch.n_face_q_points, Vector<double>(2*dim));
  std::vector<Vector<double> > linear_rhs_values(scratch.n_face_q_points, Vector<double>(2*dim));
  std::vector<Tensor<2,dim>  > grad_n (scratch.n_q_points);
  std::vector<Vector<double> > g_stress_values(scratch.n_face_q_points, Vector<double>(2*dim));
  std::vector<Tensor<1,dim>  > stress_values (2*dim);

  std::vector<Tensor<1,dim> > 		phi_n (structure_fe.dofs_per_cell);
  std::vector<SymmetricTensor<2,dim> > 	symgrad_phi_n (structure_fe.dofs_per_cell);
  std::vector<double>                  	div_phi_n   (structure_fe.dofs_per_cell);
  std::vector<Tensor<1,dim> >           phi_v       (structure_fe.dofs_per_cell);
  std::vector<Tensor<2,dim> > 	grad_phi_n (structure_fe.dofs_per_cell);
  
  std::vector<Tensor<1,dim> > grad_known_stress_now (scratch.n_face_q_points,Tensor<1,dim>(2*dim));
  std::vector<Tensor<1,dim> > grad_known_stress_old (scratch.n_face_q_points,Tensor<1,dim>(2*dim));

  scratch.fe_values.reinit(cell);
  data.cell_matrix=0;
  data.cell_rhs=0;
  
  //timer.leave_subsection ();
  //timer.enter_subsection ("Assembly");
  if (data.assemble_matrix)
    {
      //timer.enter_subsection ("Get Data");
      scratch.fe_values.get_function_values (old_solution.block(1), old_solution_values);
      scratch.fe_values[displacements].get_function_gradients(old_solution.block(1),grad_n);
      //timer.leave_subsection ();
      for (unsigned int q_point=0; q_point<scratch.n_q_points;
	   ++q_point)
	{
	  //timer.enter_subsection ("Preload Gradients");
	  for (unsigned int k=0; k<structure_fe.dofs_per_cell; ++k)
	    {
	      phi_n[k]	       = scratch.fe_values[displacements].value (k, q_point);
	      symgrad_phi_n[k] = scratch.fe_values[displacements].symmetric_gradient (k, q_point);
	      div_phi_n[k]     = scratch.fe_values[displacements].divergence (k, q_point);
	      phi_v[k]         = scratch.fe_values[velocities].value (k, q_point);
	    }
	  //timer.leave_subsection ();
	  for (unsigned int i=0; i<structure_fe.dofs_per_cell; ++i)
	    {
	      const unsigned int
		component_i = structure_fe.system_to_component_index(i).first;
	      for (unsigned int j=0; j<structure_fe.dofs_per_cell; ++j)
		{
		  const unsigned int
		    component_j = structure_fe.system_to_component_index(j).first;

		  if ((scratch.mode_type)==state || (scratch.mode_type)==linear)
		    {

		      if (component_i<dim)
			{
			  if (component_j<dim)
			    {
			      data.cell_matrix(i,j)+=(.5*	( 2*physical_properties.mu*symgrad_phi_n[i] * symgrad_phi_n[j]
							  + physical_properties.lambda*div_phi_n[i] * div_phi_n[j]))
				*scratch.fe_values.JxW(q_point);
			    }
			  else
			    {
			      data.cell_matrix(i,j)+=physical_properties.rho_s/time_step*phi_n[i]*phi_v[j]*scratch.fe_values.JxW(q_point);
			    }
			}
		      else
			{
			  if (component_j<dim)
			    {
			      data.cell_matrix(i,j)+=(-1./time_step*phi_v[i]*phi_n[j])
				*scratch.fe_values.JxW(q_point);
			    }
			  else
			    {
			      data.cell_matrix(i,j)+=(0.5*phi_v[i]*phi_v[j])
				*scratch.fe_values.JxW(q_point);
			    }
			}
		    }
		  else // scratch.mode_type==adjoint
		    {
		      if (component_i<dim)
			{
			  if (component_j<dim)
			    {
			      data.cell_matrix(i,j)+=(.5*	( 2*physical_properties.mu*symgrad_phi_n[i] * symgrad_phi_n[j]
							  + physical_properties.lambda*div_phi_n[i] * div_phi_n[j]))
				*scratch.fe_values.JxW(q_point);
			    }
			  else
			    {
			      data.cell_matrix(i,j)+=-1./time_step*phi_n[i]*phi_v[j]*scratch.fe_values.JxW(q_point);
			    }
			}
		      else
			{
			  if (component_j<dim)
			    {
			      data.cell_matrix(i,j)+=physical_properties.rho_s/time_step*phi_v[i]*phi_n[j]*scratch.fe_values.JxW(q_point);
			    }
			  else
			    {
			      data.cell_matrix(i,j)+=(0.5*phi_v[i]*phi_v[j])
				*scratch.fe_values.JxW(q_point);
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
		  Tensor<1,dim> old_n;
		  Tensor<1,dim> old_v;
		  for (unsigned int d=0; d<dim; ++d)
		    old_n[d] = old_solution_values[q_point](d);
		  for (unsigned int d=0; d<dim; ++d)
		    old_v[d] = old_solution_values[q_point](d+dim);
		  const Tensor<1,dim> phi_i_eta      	= scratch.fe_values[displacements].value (i, q_point);
		  const Tensor<2,dim> symgrad_phi_i_eta 	= scratch.fe_values[displacements].symmetric_gradient (i, q_point);
		  const double div_phi_i_eta 			= scratch.fe_values[displacements].divergence (i, q_point);
		  const Tensor<1,dim> phi_i_eta_dot  	= scratch.fe_values[velocities].value (i, q_point);
		  if (component_i<dim)
		    {
		      data.cell_rhs(i) += (physical_properties.rho_s/time_step *phi_i_eta*old_v
				       +0.5*(-2*physical_properties.mu*(scalar_product(grad_n[q_point],symgrad_phi_i_eta))
					     -physical_properties.lambda*((grad_n[q_point][0][0]+grad_n[q_point][1][1])*div_phi_i_eta))
				       )
			* scratch.fe_values.JxW(q_point);
		    }
		  else
		    {
		      data.cell_rhs(i) += (-0.5*phi_i_eta_dot*old_v
				       -1./time_step*phi_i_eta_dot*old_n
				       )
			* scratch.fe_values.JxW(q_point);
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
			Tensor<2,dim> new_stresses;
			new_stresses[0][0]=stress_values[0][0];
			new_stresses[1][0]=stress_values[1][0];
			new_stresses[1][1]=stress_values[1][1];
			new_stresses[0][1]=stress_values[0][1];
			data.cell_rhs(i) += .5*(scratch.fe_face_values[displacements].value (i, q)*
							new_stresses*scratch.fe_face_values.normal_vector(q) *
							scratch.fe_face_values.JxW(q));
		      }
		  // scratch.structure_stress_values.gradient_list( scratch.fe_face_values.get_quadrature_points(), grad_known_stress_old);
		  for (unsigned int q=0; q<scratch.n_face_q_points; ++q)
		    for (unsigned int i=0; i<structure_fe.dofs_per_cell; ++i)
		      {
			old_structure_stress_values.vector_gradient(scratch.fe_face_values.quadrature_point(q),
									stress_values);
			Tensor<2,dim> new_stresses;
			new_stresses[0][0]=stress_values[0][0];
			new_stresses[1][0]=stress_values[1][0];
			new_stresses[1][1]=stress_values[1][1];
			new_stresses[0][1]=stress_values[0][1];
			data.cell_rhs(i) += (1-.5)*(scratch.fe_face_values[displacements].value (i, q)*
							     new_stresses*scratch.fe_face_values.normal_vector(q) *
							     scratch.fe_face_values.JxW(q));
		      }
		}
	    }
	  else if (structure_boundaries[cell->face(face_no)->boundary_indicator()]==Interface)
	    {
	      if ((scratch.mode_type)==state)
		{
		  scratch.fe_face_values.reinit (cell, face_no);
		  scratch.fe_face_values.get_function_values (stress.block(1), g_stress_values);
		  for (unsigned int q=0; q<scratch.n_face_q_points; ++q)
		    {
		      Tensor<1,dim> g_stress;
		      for (unsigned int d=0; d<dim; ++d)
			g_stress[d] = g_stress_values[q](d);
		      for (unsigned int i=0; i<structure_fe.dofs_per_cell; ++i)
			{
			  data.cell_rhs(i) += .5*(scratch.fe_face_values[displacements].value (i, q)*
							       (-g_stress) * scratch.fe_face_values.JxW(q));
			}
		    }
		  scratch.fe_face_values.get_function_values (old_stress.block(1), g_stress_values);
		  for (unsigned int q=0; q<scratch.n_face_q_points; ++q)
		    {
		      Tensor<1,dim> g_stress;
		      for (unsigned int d=0; d<dim; ++d)
			g_stress[d] = g_stress_values[q](d);
		      for (unsigned int i=0; i<structure_fe.dofs_per_cell; ++i)
			{
			  data.cell_rhs(i) += .5*(scratch.fe_face_values[displacements].value (i, q)*
								   (-g_stress) * scratch.fe_face_values.JxW(q));
			}
		    }
		}
	      else if ((scratch.mode_type)==adjoint)
		{
		  scratch.fe_face_values.reinit (cell, face_no);
		  scratch.fe_face_values.get_function_values (rhs_for_adjoint.block(1), adjoint_rhs_values);
		  for (unsigned int q=0; q<scratch.n_face_q_points; ++q)
		    {
		      Tensor<1,dim> r;
		      if (fem_properties.adjoint_type==1)
			{
			  for (unsigned int d=0; d<dim; ++d)
			    r[d] = adjoint_rhs_values[q](d);
			  for (unsigned int i=0; i<structure_fe.dofs_per_cell; ++i)
			    {
			      data.cell_rhs(i) += .5*(scratch.fe_face_values[displacements].value (i, q)*
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
				  data.cell_rhs(i) += .5*(scratch.fe_face_values[velocities].value (i, q)*
								       r * scratch.fe_face_values.JxW(q));
				}
			    }
			  else
			    {
			      for (unsigned int d=0; d<dim; ++d)
				r[d] = adjoint_rhs_values[q](d+dim);
			      for (unsigned int i=0; i<structure_fe.dofs_per_cell; ++i)
				{
				  data.cell_rhs(i) += .5*(scratch.fe_face_values[velocities].value (i, q)*
								       r * scratch.fe_face_values.JxW(q));
				}
			    }
			}

		    }
		}
	      else // (scratch.mode_type)==linear
		{
		  scratch.fe_face_values.reinit (cell, face_no);
		  scratch.fe_face_values.get_function_values (rhs_for_linear.block(1), linear_rhs_values);
		  for (unsigned int q=0; q<scratch.n_face_q_points; ++q)
		    {
		      Tensor<1,dim> h;
		      for (unsigned int d=0; d<dim; ++d)
			h[d] = linear_rhs_values[q](d);
		      for (unsigned int i=0; i<structure_fe.dofs_per_cell; ++i)
			{
			  data.cell_rhs(i) += .5*(scratch.fe_face_values[displacements].value (i, q)*
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


  master_thread = Threads::this_thread_id();

  PerTaskData<dim> per_task_data(structure_fe, structure_matrix, structure_rhs, assemble_matrix);
  FullScratchData<dim> scratch_data(structure_fe, quadrature_formula, update_values | update_gradients | update_quadrature_points | update_JxW_values,
					  face_quadrature_formula, update_values | update_normal_vectors | update_quadrature_points  | update_JxW_values,
					  (unsigned int)enum_);
 
  WorkStream::run (structure_dof_handler.begin_active(),
  		   structure_dof_handler.end(),
  		   *this,
  		   &FSIProblem<dim>::assemble_structure_matrix_on_one_cell,
  		   &FSIProblem<dim>::copy_local_structure_to_global,
  		   scratch_data,
  		   per_task_data);

  
}

template void FSIProblem<2>::assemble_structure_matrix_on_one_cell (const DoFHandler<2>::active_cell_iterator& cell,
							     FullScratchData<2>& scratch,
							     PerTaskData<2>& data );

template void FSIProblem<2>::copy_local_structure_to_global (const PerTaskData<2> &data);
							     
template void FSIProblem<2>::assemble_structure (Mode enum_, bool assemble_matrix);


// dt = 0.0555556 h_f = 0.157135 h_s = 0.114531 L2(T) error [fluid] = 0.00118288,  L2(T) error [structure] = 8.58267e-05
//  L2(0,T;H1(t)) error [fluid] = 0.00120148,  Pressure error [fluid] = 0.0024587,  L2(0,T;H1(t)) errors [structure] = 3.52664e-05 L2(T) error [structure_vel] = 2.99115e-05
// h           fluid.vel.L2   fluid.vel.H1   fluid.press.L2   structure.displ.L2   structure.displ.H1   structure.vel.L2
// 0.353553               -              -                -                    -                    -                  -
// 0.235702            2.91           3.53             2.17                 1.25                 2.50               3.29
// 0.157135            2.49           3.54             1.85                 1.79                 2.81               3.27
// ----> data to match
