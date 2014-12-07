#include "FSI_Project.h"
#include <deal.II/grid/grid_out.h>

template <int dim>
void FSIProblem<dim>::output_results () const
{
  /* To see the true solution
   * - This requires removing 'const from this function where it is declared and defined.
   * FluidBoundaryValues<dim> fluid_boundary_values(fem_prop);
   * fluid_boundary_values.set_time(time);
   * VectorTools::interpolate(fluid_dof_handler,fluid_boundary_values,
   *			                          solution.block(0));
   */
  std::vector<std::vector<std::string> > solution_names(3);
  switch (dim)
    {
    case 2:
      solution_names[0].push_back ("u_x");
      solution_names[0].push_back ("u_y");
      solution_names[0].push_back ("p");
      solution_names[1].push_back ("n_x");
      solution_names[1].push_back ("n_y");
      solution_names[1].push_back ("v_x");
      solution_names[1].push_back ("v_y");
      solution_names[2].push_back ("a_x");
      solution_names[2].push_back ("a_y");
      break;

    case 3:
      solution_names[0].push_back ("u_x");
      solution_names[0].push_back ("u_y");
      solution_names[0].push_back ("u_z");
      solution_names[0].push_back ("p");
      solution_names[1].push_back ("n_x");
      solution_names[1].push_back ("n_y");
      solution_names[1].push_back ("n_z");
      solution_names[1].push_back ("v_x");
      solution_names[1].push_back ("v_y");
      solution_names[1].push_back ("v_z");
      solution_names[2].push_back ("a_x");
      solution_names[2].push_back ("a_y");
      solution_names[2].push_back ("a_z");
      break;

    default:
      AssertThrow (false, ExcNotImplemented());
    }
  DataOut<dim> fluid_data_out, structure_data_out;
  fluid_data_out.add_data_vector (fluid_dof_handler,solution.block(0), solution_names[0]);
  fluid_data_out.add_data_vector (ale_dof_handler,solution.block(2), solution_names[2]);
  structure_data_out.add_data_vector (structure_dof_handler,solution.block(1), solution_names[1]);
  fluid_data_out.build_patches (fem_properties.fluid_degree-1);
  structure_data_out.build_patches (fem_properties.structure_degree+1);
  const std::string fluid_filename = "fluid-" +
    Utilities::int_to_string (timestep_number, 4) +
    ".vtk";
  const std::string structure_filename = "structure-" +
    Utilities::int_to_string (timestep_number, 4) +
    ".vtk";
  std::ofstream fluid_output (fluid_filename.c_str());
  std::ofstream structure_output (structure_filename.c_str());
  fluid_data_out.write_vtk (fluid_output);
  structure_data_out.write_vtk (structure_output);
  // const std::string fluid_mesh_filename = "fluid-" +
  //   Utilities::int_to_string (timestep_number, 3) +
  //   ".vtk";
  // std::ofstream mesh_out (fluid_mesh_filename.c_str());
  // GridOut grid_out;
  // grid_out.write_vtk (fluid_triangulation, mesh_out);
}

template <int dim>
void FSIProblem<dim>::compute_error ()
{
  QTrapez<1>     q_trapez;
  QIterated<dim> quadrature (q_trapez, 3);

  Vector<double> fluid_cellwise_errors (fluid_triangulation.n_active_cells());
  FluidBoundaryValues<dim> fluid_exact_solution(physical_properties);
  fluid_exact_solution.set_time(time);
  std::pair<unsigned int,unsigned int> fluid_indices(0,dim);
  ComponentSelectFunction<dim> fluid_velocity_mask(fluid_indices,dim+1);

  // Calculate l^inf(l2) error of fluid velocity
  fluid_cellwise_errors=0;
  VectorTools::integrate_difference (fluid_dof_handler, solution.block(0), fluid_exact_solution,
				     fluid_cellwise_errors, quadrature,
				     VectorTools::L2_norm,&fluid_velocity_mask);
  errors.fluid_velocity_L2_Error = std::max(errors.fluid_velocity_L2_Error,fluid_cellwise_errors.l2_norm());

  // Calculate l2(h1) error of fluid velocity
  fluid_cellwise_errors=0;
  VectorTools::integrate_difference (fluid_dof_handler, solution.block(0), fluid_exact_solution,
				     fluid_cellwise_errors, quadrature, VectorTools::H1_norm,&fluid_velocity_mask);
  errors.fluid_velocity_H1_Error += fluid_cellwise_errors.l2_norm();


  Vector<double> structure_cellwise_errors (structure_triangulation.n_active_cells());
  StructureBoundaryValues<dim> structure_exact_solution(physical_properties);
  structure_exact_solution.set_time(time);

  std::pair<unsigned int,unsigned int> structure_displacement_indices(0,dim);
  std::pair<unsigned int,unsigned int> structure_velocity_indices(dim,2*dim);
  ComponentSelectFunction<dim> structure_displacement_mask(structure_displacement_indices,2*dim);
  ComponentSelectFunction<dim> structure_velocity_mask(structure_velocity_indices,2*dim);

  // Calculate l2(l2) error of structure displacements over time steps
  structure_cellwise_errors = 0;
  VectorTools::integrate_difference (structure_dof_handler, solution.block(1), structure_exact_solution,
				     structure_cellwise_errors, quadrature,
				     VectorTools::L2_norm,&structure_displacement_mask);
  errors.structure_displacement_L2_Error += structure_cellwise_errors.l2_norm();

  // Calculate l2(h1) error of structure displacements over time steps
  structure_cellwise_errors = 0;
  VectorTools::integrate_difference (structure_dof_handler, solution.block(1), structure_exact_solution,
				     structure_cellwise_errors, quadrature, VectorTools::H1_norm,&structure_displacement_mask);
  errors.structure_displacement_H1_Error += structure_cellwise_errors.l2_norm();

  // Calculate l^inf(l2) error of structure velocities
  VectorTools::integrate_difference (structure_dof_handler, solution.block(1), structure_exact_solution,
				     structure_cellwise_errors, quadrature,
				     VectorTools::L2_norm,&structure_velocity_mask);
  errors.structure_velocity_L2_Error = std::max(errors.structure_velocity_L2_Error, structure_cellwise_errors.l2_norm());

  if (std::fabs(time-fem_properties.T)<1e-13)
    {
      // Calculate pressure error only at last time step
      ComponentSelectFunction<dim> fluid_pressure_mask(dim,dim+1);

      // Set time step appropriately and then calculate error at current time step in pressure
      fluid_exact_solution.set_time(time-(1-fem_properties.fluid_theta)*time_step);
      fluid_cellwise_errors=0;
      VectorTools::integrate_difference (fluid_dof_handler, solution.block(0), fluid_exact_solution,
					 fluid_cellwise_errors, quadrature,
					 VectorTools::L2_norm,&fluid_pressure_mask);
      //errors.fluid_pressure_L2_Error=std::max(errors.fluid_pressure_L2_Error,fluid_cellwise_errors.l2_norm());
      errors.fluid_pressure_L2_Error=fluid_cellwise_errors.l2_norm();

      AssertThrow(errors.fluid_velocity_L2_Error>=0 && errors.fluid_velocity_H1_Error>=0 && errors.fluid_pressure_L2_Error>=0
		  && errors.structure_displacement_L2_Error>=0 && errors.structure_displacement_H1_Error>=0 && errors.structure_velocity_L2_Error>=0,ExcIO());
      errors.fluid_velocity_H1_Error *= time_step;
      errors.structure_displacement_H1_Error *= time_step;
      errors.structure_displacement_L2_Error *= time_step;

      std::cout << "dt = " << time_step
		<< " h_f = " << fluid_triangulation.begin_active()->diameter() << " h_s = " << structure_triangulation.begin_active()->diameter()
		<< " L2(T) error [fluid] = " << errors.fluid_velocity_L2_Error << ", "<< " L2(T) error [structure] = " << errors.structure_displacement_L2_Error << std::endl
		<< " L2(0,T;H1(t)) error [fluid] = " << errors.fluid_velocity_H1_Error << ", "
		<< " Pressure error [fluid] = " << errors.fluid_pressure_L2_Error << ", "
		<< " L2(0,T;H1(t)) errors [structure] = " << errors.structure_displacement_H1_Error << " L2(T) error [structure_vel] = " << errors.structure_velocity_L2_Error << std::endl;

      errors.fluid_active_cells=fluid_triangulation.n_active_cells();
      errors.structure_active_cells=structure_triangulation.n_active_cells();
      errors.fluid_velocity_dofs = dofs_per_block[0];//*timestep_number;
      errors.fluid_pressure_dofs = dofs_per_block[1];//*timestep_number;
      errors.structure_displacement_dofs = dofs_per_block[2];//*timestep_number;
      errors.structure_velocity_dofs = dofs_per_block[3];//*timestep_number;

      std::vector<double> L2_error_array(4);
      L2_error_array[0]=errors.fluid_velocity_L2_Error;
      L2_error_array[1]=errors.fluid_pressure_L2_Error;
      L2_error_array[2]=errors.structure_displacement_L2_Error;
      L2_error_array[3]=errors.structure_velocity_L2_Error;

      std::vector<double> H1_error_array(2);
      H1_error_array[0]=errors.fluid_velocity_H1_Error;
      H1_error_array[1]=errors.structure_displacement_H1_Error;

      // Write the error to errors.dat file
      std::vector<std::string> subsystem(2);
      subsystem[0]="fluid"; subsystem[1]="structure";
      std::vector<std::vector<std::string> > variable(2,std::vector<std::string>(2));
      variable[0][0]="vel";variable[0][1]="press";
      variable[1][0]="displ";variable[1][1]="vel";
      std::vector<unsigned int> show_errors(4,1);
      show_errors[0]=2;show_errors[2]=2;

      std::ofstream error_data;
      error_data.open("errors.dat");
      for (unsigned int i=0; i<subsystem.size(); ++i)
	{
	  for (unsigned int j=0; j<variable.size(); ++j)
	    {
	      error_data << subsystem[i] << "." << variable[i][j] << ".dofs:";
	      if (fem_properties.convergence_mode=="space")
		{
		  if (j==0)
		    {
		      error_data << fluid_triangulation.begin_active()->diameter() << std::endl;
		    }
		  else
		    {
		      error_data << structure_triangulation.begin_active()->diameter() << std::endl;
		    }
		}
	      else
		{
		  error_data << timestep_number << std::endl;
		}
	      for (unsigned int k=0; k<show_errors[2*i+j]; ++k)
		{
		  error_data << subsystem[i] << "." << variable[i][j] << ".";
		  if (k==0)
		    {
		      error_data << "L2:" << L2_error_array[2*i+j] << std::endl;
		    }
		  else
		    {
		      error_data << "H1:" << H1_error_array[i] << std::endl;
		    }
		}
	    }
	}
      error_data.close();
    }
}

template void FSIProblem<2>::output_results () const;
template void FSIProblem<2>::compute_error ();
