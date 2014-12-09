#include "FSI_Project.h"
#include <deal.II/grid/grid_in.h>

template <int dim>
void FSIProblem<dim>::dirichlet_boundaries (System system, Mode enum_)
{
  const FEValuesExtractors::Vector velocities (0);
  const FEValuesExtractors::Vector displacements (0);
  const FEValuesExtractors::Vector ale_displacement (0);

  if (enum_==state)
    {
      if (system==Fluid)
	{
	  FluidBoundaryValues<dim> fluid_boundary_values_function(physical_properties);
	  fluid_boundary_values_function.set_time (time);
	  std::map<types::global_dof_index,double> fluid_boundary_values;
	  for (unsigned int i=0; i<4; ++i)
	    {
	      if (fluid_boundaries[i]==Dirichlet)
		{
		  if (physical_properties.simulation_type!=1)
		    {
		      VectorTools::interpolate_boundary_values (fluid_dof_handler,
								i,
								fluid_boundary_values_function,
								fluid_boundary_values,
								fluid_fe.component_mask(velocities));
		    }
		  else
		    {
		      VectorTools::interpolate_boundary_values (fluid_dof_handler,
								i,
								ZeroFunction<dim>(dim+1),
								fluid_boundary_values,
								fluid_fe.component_mask(velocities));
		    }
		}
	    }
	  MatrixTools::apply_boundary_values (fluid_boundary_values,
					      system_matrix.block(0,0),
					      solution.block(0),
					      system_rhs.block(0));//,
	  //!physical_properties.stability_terms);
	}
      else if (system==Structure)
	{
	  StructureBoundaryValues<dim> structure_boundary_values_function(physical_properties);
	  structure_boundary_values_function.set_time (time);
	  std::map<types::global_dof_index,double> structure_boundary_values;
	  for (unsigned int i=0; i<4; ++i)
	    {
	      if (structure_boundaries[i]==Dirichlet)
		{
		  VectorTools::interpolate_boundary_values (structure_dof_handler,
							    i,
							    structure_boundary_values_function,
							    structure_boundary_values);
		}
	    }
	  MatrixTools::apply_boundary_values (structure_boundary_values,
					      system_matrix.block(1,1),
					      solution.block(1),
					      system_rhs.block(1));
	}
      else
	{
	  std::map<types::global_dof_index,double> ale_dirichlet_boundary_values;
	  std::map<types::global_dof_index,double> ale_interface_boundary_values;
	  for (unsigned int i=0; i<dofs_per_big_block[2]; ++i) // loops over nodes local to ale
	    {
	      if (a2n.count(i)) // lookup key for certain ale dof
		{
		  ale_interface_boundary_values.insert(std::pair<unsigned int,double>(i,solution.block(1)[a2n[i]]));
		}
	    }
	  for (unsigned int i=0; i<4; ++i)
	    {
	      if (ale_boundaries[i]==Dirichlet)
		{
		  VectorTools::interpolate_boundary_values (ale_dof_handler,
							    i,
							    ZeroFunction<dim>(dim),
							    ale_dirichlet_boundary_values,
							    ale_fe.component_mask(ale_displacement));
		}
	    }
	  MatrixTools::apply_boundary_values (ale_dirichlet_boundary_values,
					      system_matrix.block(2,2),
					      solution.block(2),
					      system_rhs.block(2));
	  MatrixTools::apply_boundary_values (ale_interface_boundary_values,
					      system_matrix.block(2,2),
					      solution.block(2),
					      system_rhs.block(2));
	}
    }
  else //  enum_==adjoint or enum_==linear
    {
      if (system==Fluid)
	{
	  std::map<types::global_dof_index,double> fluid_boundary_values;
	  for (unsigned int i=0; i<4; ++i)
	    {
	      if (fluid_boundaries[i]==Dirichlet)// non interface or Neumann sides
		{
		  VectorTools::interpolate_boundary_values (fluid_dof_handler,
							    i,
							    ZeroFunction<dim>(dim+1),
							    fluid_boundary_values,
							    fluid_fe.component_mask(velocities));
		}
	    }
	  if (enum_==adjoint)
	    {
	      MatrixTools::apply_boundary_values (fluid_boundary_values,
						  adjoint_matrix.block(0,0),
						  adjoint_solution.block(0),
						  adjoint_rhs.block(0));
	    }
	  else
	    { 
	      MatrixTools::apply_boundary_values (fluid_boundary_values,
						  linear_matrix.block(0,0),
						  linear_solution.block(0),
						  linear_rhs.block(0));
	    }
	}
      else if (system==Structure)
	{
	  std::map<types::global_dof_index,double> structure_boundary_values;
	  for (unsigned int i=0; i<4; ++i)
	    {
	      if (structure_boundaries[i]==Dirichlet)// non interface or Neumann sides
		{
		  VectorTools::interpolate_boundary_values (structure_dof_handler,
							    i,
							    ZeroFunction<dim>(2*dim),
							    structure_boundary_values,
							    structure_fe.component_mask(displacements));
		}
	    }
	  if (enum_==adjoint)
	    {
	      MatrixTools::apply_boundary_values (structure_boundary_values,
						  adjoint_matrix.block(1,1),
						  adjoint_solution.block(1),
						  adjoint_rhs.block(1));
	    }
	  else
	    {
	      MatrixTools::apply_boundary_values (structure_boundary_values,
						  linear_matrix.block(1,1),
						  linear_solution.block(1),
						  linear_rhs.block(1));
	    }
	}
      else
	{
	  std::map<types::global_dof_index,double> ale_boundary_values;
	  for (unsigned int i=0; i<4; ++i)
	    {
	      if (ale_boundaries[i]==Dirichlet || ale_boundaries[i]==Interface)// non interface or Neumann sides
		{
		  VectorTools::interpolate_boundary_values (ale_dof_handler,
							    i,
							    ZeroFunction<dim>(dim),
							    ale_boundary_values);
		}
	    }
	  if (enum_==adjoint)
	    {
	      MatrixTools::apply_boundary_values (ale_boundary_values,
						  adjoint_matrix.block(2,2),
						  adjoint_solution.block(2),
						  adjoint_rhs.block(2));
	    }	
	  else
	    {
	      MatrixTools::apply_boundary_values (ale_boundary_values,
						  linear_matrix.block(2,2),
						  linear_solution.block(2),
						  linear_rhs.block(2));
	    }
	}
    }
}

template <int dim>
void FSIProblem<dim>::setup_system ()
{
  AssertThrow(dim==2,ExcNotImplemented());
  if (physical_properties.simulation_type == 0 || physical_properties.simulation_type == 2) {
    Point<2> fluid_bottom_left(0,0), fluid_top_right(fem_properties.fluid_width,fem_properties.fluid_height);
    Point<2> structure_bottom_left(0,fem_properties.fluid_height),
      structure_top_right(fem_properties.structure_width,fem_properties.fluid_height+fem_properties.structure_height);
    std::vector<double> x_scales(fem_properties.nx_f,fem_properties.fluid_width/((double)fem_properties.nx_f));
    std::vector<double> f_y_scales(fem_properties.ny_f,fem_properties.fluid_height/((double)fem_properties.ny_f));
    std::vector<double> s_y_scales(fem_properties.ny_s,fem_properties.structure_height/((double)fem_properties.ny_s));

    std::vector<std::vector<double> > f_scales(2),s_scales(2);
    f_scales[0]=x_scales;f_scales[1]=f_y_scales;
    s_scales[0]=x_scales;s_scales[1]=s_y_scales;
    GridGenerator::subdivided_hyper_rectangle (fluid_triangulation,f_scales,fluid_bottom_left,fluid_top_right,false);
    GridGenerator::subdivided_hyper_rectangle (structure_triangulation,s_scales,structure_bottom_left,structure_top_right,false);

    // Structure sits on top of fluid
    AssertThrow(fem_properties.nx_f==fem_properties.nx_s,ExcNotImplemented()); // Checks that the interface edges are equally refined
    AssertThrow(std::fabs(fem_properties.fluid_width-fem_properties.structure_width)<1e-15,ExcNotImplemented());

    for (unsigned int i=0; i<4; ++i)
      {
	if (physical_properties.simulation_type == 0) {
	  if (i==1||i==3) fluid_boundaries.insert(std::pair<unsigned int, BoundaryCondition>(i,Neumann));
	  else if (i==2) fluid_boundaries.insert(std::pair<unsigned int, BoundaryCondition>(i,Interface));
	  else fluid_boundaries.insert(std::pair<unsigned int, BoundaryCondition>(i,Dirichlet));
	} else { // simulation_type ==2
	  if (i==0) fluid_boundaries.insert(std::pair<unsigned int, BoundaryCondition>(i,Neumann));
	  else fluid_boundaries.insert(std::pair<unsigned int, BoundaryCondition>(i,Dirichlet));
	}
      }

    for (unsigned int i=0; i<4; ++i)
      {
	if (physical_properties.simulation_type == 0) {
	  if (i==0) structure_boundaries.insert(std::pair<unsigned int, BoundaryCondition>(i,Interface));
	  else if (i==1||i==3) structure_boundaries.insert(std::pair<unsigned int, BoundaryCondition>(i,Neumann));
	  else structure_boundaries.insert(std::pair<unsigned int, BoundaryCondition>(i,Dirichlet));
	} else { // simulation_type ==2
	  if (i==1||i==3) structure_boundaries.insert(std::pair<unsigned int, BoundaryCondition>(i,Neumann));
	  else structure_boundaries.insert(std::pair<unsigned int, BoundaryCondition>(i,Dirichlet));
	}
      }
    for (unsigned int i=0; i<4; ++i)
      {
	if (physical_properties.simulation_type == 0) {
	  if (i==2) ale_boundaries.insert(std::pair<unsigned int, BoundaryCondition>(i,Interface));
	  else if (i==0) ale_boundaries.insert(std::pair<unsigned int, BoundaryCondition>(i,Dirichlet));
	  else ale_boundaries.insert(std::pair<unsigned int, BoundaryCondition>(i,Neumann));
	} else { // simulation_type ==2
	  ale_boundaries.insert(std::pair<unsigned int, BoundaryCondition>(i,Dirichlet));
	}
      }
  } else if (physical_properties.simulation_type == 1) {
    Point<2> fluid_bottom_left(0,0), fluid_top_right(fem_properties.fluid_width,fem_properties.fluid_height);
    Point<2> structure_bottom_left(0,fem_properties.fluid_height),
      structure_top_right(fem_properties.structure_width,fem_properties.fluid_height+fem_properties.structure_height);
    std::vector<double> x_scales(fem_properties.nx_f,fem_properties.fluid_width/((double)fem_properties.nx_f));
    std::vector<double> f_y_scales(fem_properties.ny_f,fem_properties.fluid_height/((double)fem_properties.ny_f));
    std::vector<double> s_y_scales(fem_properties.ny_s,fem_properties.structure_height/((double)fem_properties.ny_s));

    std::vector<std::vector<double> > f_scales(2),s_scales(2);
    f_scales[0]=x_scales;f_scales[1]=f_y_scales;
    s_scales[0]=x_scales;s_scales[1]=s_y_scales;
    std::vector<unsigned int > f_reps(2),s_reps(2);
    f_reps[0]=fem_properties.nx_f; f_reps[1]=fem_properties.ny_f;
    s_reps[0]=fem_properties.nx_s; s_reps[1]=fem_properties.ny_s;

    GridGenerator::subdivided_hyper_rectangle (fluid_triangulation,f_reps,fluid_bottom_left,fluid_top_right,false);
    GridGenerator::subdivided_hyper_rectangle (structure_triangulation,s_reps,structure_bottom_left,structure_top_right,false);

    // Structure sits on top of fluid
    AssertThrow(fem_properties.nx_f==fem_properties.nx_s,ExcNotImplemented()); // Checks that the interface edges are equally refined
    AssertThrow(std::fabs(fem_properties.fluid_width-fem_properties.structure_width)<1e-15,ExcNotImplemented());

    for (unsigned int i=0; i<4; ++i)
      {
	if (i==1) fluid_boundaries.insert(std::pair<unsigned int, BoundaryCondition>(i,DoNothing));
	else if (i==3) fluid_boundaries.insert(std::pair<unsigned int, BoundaryCondition>(i,Neumann));
	else if (i==2) fluid_boundaries.insert(std::pair<unsigned int, BoundaryCondition>(i,Interface));
	else fluid_boundaries.insert(std::pair<unsigned int, BoundaryCondition>(i,Dirichlet));
	fluid_interface_boundaries.insert(2);
      }

    for (unsigned int i=0; i<4; ++i)
      {
	if (i==0) structure_boundaries.insert(std::pair<unsigned int, BoundaryCondition>(i,Interface));
	else if (i==1||i==3) structure_boundaries.insert(std::pair<unsigned int, BoundaryCondition>(i,Dirichlet));
	else structure_boundaries.insert(std::pair<unsigned int, BoundaryCondition>(i,Neumann));
	structure_interface_boundaries.insert(0);
      }
    for (unsigned int i=0; i<4; ++i)
      {
	if (i==2) ale_boundaries.insert(std::pair<unsigned int, BoundaryCondition>(i,Interface));
	else ale_boundaries.insert(std::pair<unsigned int, BoundaryCondition>(i,Dirichlet));
      }
  } else if (physical_properties.simulation_type == 3) {
    GridIn<2> gridin_fluid;
    gridin_fluid.attach_triangulation(fluid_triangulation);
    std::ifstream f_fluid("HronTurek-Fluid.msh");
    gridin_fluid.read_msh(f_fluid);

    GridIn<2> gridin_structure;
    gridin_structure.attach_triangulation(structure_triangulation);
    std::ifstream f_structure("HronTurek-Structure.msh");
    gridin_structure.read_msh(f_structure);

    for (unsigned int i=1; i<=8; ++i)
      {
	// 1- bottom, 3- top, 4- left, 8- circle
	if (i==1 || i==3 || i==4 || i==8) fluid_boundaries.insert(std::pair<unsigned int, BoundaryCondition>(i,Dirichlet));
	// 2- right
	else if (i==2) fluid_boundaries.insert(std::pair<unsigned int, BoundaryCondition>(i,DoNothing));
	else if (i>=5 || i<=7) fluid_boundaries.insert(std::pair<unsigned int, BoundaryCondition>(i,Dirichlet)); // Interface
	else AssertThrow(false, ExcNotImplemented()); // There should be no other boundary option
	//fluid_interface_boundaries.insert();
      }

    for (unsigned int i=1; i<=4; ++i)
      {
	// 1- bottom, 2- right, 3- top
	if (i>=1 && i<=3) structure_boundaries.insert(std::pair<unsigned int, BoundaryCondition>(i,Dirichlet)); // Interface
	// 4- left (against cylinder)
	else if (i==4) structure_boundaries.insert(std::pair<unsigned int, BoundaryCondition>(i,Dirichlet));
	else AssertThrow(false, ExcNotImplemented()); // There should be no other boundary option
	//structure_interface_boundaries.insert();
      }
    for (unsigned int i=1; i<=8; ++i)
      {
	if (i>=5 || i<=7) ale_boundaries.insert(std::pair<unsigned int, BoundaryCondition>(i,Dirichlet)); // Interface
	else if (i>=1 && i<=8) ale_boundaries.insert(std::pair<unsigned int, BoundaryCondition>(i,Dirichlet));
	else AssertThrow(false, ExcNotImplemented()); // There should be no other boundary option
      }
  } else {
    AssertThrow(false,ExcNotImplemented());
  }

  // std::vector<Point<dim> > fluid_face_centers, temp_structure_face_centers(structure_triangulation.n_active_cells());
  // std::vector<bool> temp_structure_interface_cells(structure_triangulation.n_active_cells());
  // std::vector<unsigned int> temp_structure_interface_faces(structure_triangulation.n_active_cells());
  // std::vector<Point<dim> > structure_face_centers(fluid_interface_faces.size());

  if (physical_properties.simulation_type < 3) {
    // All of these cases have the idea of the fluid placed below the structure and both as rectangles

    // we need to track cells, faces, and temporarily the centers for the faces
    // also, we will initially have a temp_* vectors that we will rearrange to match the order of the fluid

    //unsigned int ind=0;
    for (typename Triangulation<dim>::active_cell_iterator
	   cell = fluid_triangulation.begin_active(); cell != fluid_triangulation.end(); ++cell)
      {
	for (unsigned int f=0; f<GeometryInfo<dim>::faces_per_cell; ++f)
	  if (cell->face(f)->at_boundary())
	    {
	      if (std::fabs(cell->face(f)->center()[1])<1e-5*(1./fem_properties.ny_f))
		{ // BOTTOM OF FLUID BOUNDARY
		  cell->face(f)->set_all_boundary_indicators(0);
		}
	      else if (std::fabs(cell->face(f)->center()[0])<1e-5*(1./fem_properties.nx_f))
		{ // LEFT SIDE OF FLUID BOUNDARY
		  cell->face(f)->set_all_boundary_indicators(3);
		}
	      else if (std::fabs(cell->face(f)->center()[0]-fem_properties.fluid_width)<1e-5*(1./fem_properties.nx_f))
		{ // RIGHT SIDE OF FLUID BOUNDARY
		  cell->face(f)->set_all_boundary_indicators(1);
		}
	      else if (std::fabs(cell->face(f)->center()[1]-fem_properties.fluid_height)<1e-5*1./fem_properties.ny_f)
		{ // ON THE INTERFACE
		  cell->face(f)->set_all_boundary_indicators(2);
		  // fluid_interface_cells.push_back(ind);
		  // fluid_interface_faces.push_back(f);
		  // fluid_face_centers.push_back(cell->face(f)->center());
		}
	    }
	// ++ind;
      }

    // ind=0;
    for (typename Triangulation<dim>::active_cell_iterator
	   cell = structure_triangulation.begin_active(); cell != structure_triangulation.end(); ++cell)
      {
	for (unsigned int f=0; f<GeometryInfo<dim>::faces_per_cell; ++f)
	  if (cell->face(f)->at_boundary())
	    {
	      if (std::fabs(cell->face(f)->center()[1]-(fem_properties.fluid_height+fem_properties.structure_height))<1e-5*(1./fem_properties.ny_s))
		{ // TOP OF STRUCTURE BOUNDARY
		  cell->face(f)->set_all_boundary_indicators(2);
		}
	      else if (std::fabs(cell->face(f)->center()[0])<1e-5*(1./fem_properties.nx_s))
		{ // LEFT SIDE OF STRUCTURE BOUNDARY
		  cell->face(f)->set_all_boundary_indicators(3);
		}
	      else if (std::fabs(cell->face(f)->center()[0]-fem_properties.structure_width)<1e-5*(1./fem_properties.nx_s))
		{ // RIGHT SIDE OF STRUCTURE BOUNDARY
		  cell->face(f)->set_all_boundary_indicators(1);
		}
	      else if (std::fabs(cell->face(f)->center()[1]-fem_properties.fluid_height)<1e-5*1./fem_properties.ny_s)
		{ // INTERFACE BOUNDARY
		  cell->face(f)->set_all_boundary_indicators(0);
		  // temp_structure_interface_cells[ind]=true;
		  // temp_structure_interface_faces[ind]=f;
		  // temp_structure_face_centers[ind]=cell->face(f)->center();
		}
	    }
	// ++ind;
      }

  } else if (physical_properties.simulation_type == 3) {
    // unsigned int ind=0;
    // for (typename Triangulation<dim>::active_cell_iterator
    // 	   cell = fluid_triangulation.begin_active(); cell != fluid_triangulation.end(); ++cell)
    //   {
    // 	for (unsigned int f=0; f<GeometryInfo<dim>::faces_per_cell; ++f)
    // 	  if (cell->face(f)->at_boundary())
    // 	    {
    // 	      if (cell->face(f)->boundary_indicator()>=5 || cell->face(f)->boundary_indicator()<=7) {
    // 		fluid_interface_cells.push_back(ind);
    // 		fluid_interface_faces.push_back(f);
    // 		fluid_face_centers.push_back(cell->face(f)->center());
    // 	      }
    // 	    }
    // 	++ind;
    //   }

    // ind=0;
    // for (typename Triangulation<dim>::active_cell_iterator
    // 	   cell = structure_triangulation.begin_active(); cell != structure_triangulation.end(); ++cell)
    //   {
    // 	for (unsigned int f=0; f<GeometryInfo<dim>::faces_per_cell; ++f)
    // 	  if (cell->face(f)->at_boundary())
    // 	    {
    // 	      if (cell->face(f)->boundary_indicator()<=3) {
    // 		temp_structure_interface_cells[ind]=true;
    // 		temp_structure_interface_faces[ind]=f;
    // 		temp_structure_face_centers[ind]=cell->face(f)->center();
    // 	      }
    // 	    }
    // 	++ind;
    //   }
  } else {
    AssertThrow(false, ExcNotImplemented());
  }

  // for (unsigned int i=0; i<temp_structure_face_centers.size(); ++i)
  //   std::cout << temp_structure_face_centers[i] << ",";
  // std::cout << std::endl;
  // //std::cout << fluid_face_centers.size() << std::endl;
  // std::cout << "Made it to matching " << std::endl;
  // find the matching cells and edges between the two subproblems

  // structure_interface_cells.resize(fluid_interface_cells.size());
  // structure_interface_faces.resize(fluid_interface_cells.size());

  // for (unsigned int i=0; i < fluid_interface_cells.size(); ++i)
  //   {
  //     unsigned int j=0;
  //     for (typename Triangulation<dim>::active_cell_iterator
  //            cell = structure_triangulation.begin_active();
  // 	   cell != structure_triangulation.end(); ++cell)
  //       {
  // 	  std::cout << temp_structure_face_centers[j] << std::endl;
  // 	  if (temp_structure_interface_cells[j] && fluid_face_centers[i].distance(temp_structure_face_centers[j])<1e-13)
  // 	    {
  // 	      structure_interface_cells[i]=j;
  // 	      structure_interface_faces[i]=temp_structure_interface_faces[j];
  // 	      structure_face_centers[i]=temp_structure_face_centers[j];
  // 	    }
  // 	  ++j;
  //       }
  //   }

  std::cout << "Number of active cells: "
	    << "fluid: " << fluid_triangulation.n_active_cells()
	    << " structure: " << structure_triangulation.n_active_cells()
	    << std::endl;

  fluid_dof_handler.distribute_dofs (fluid_fe);
  structure_dof_handler.distribute_dofs (structure_fe);
  ale_dof_handler.distribute_dofs (ale_fe);
  std::vector<unsigned int> fluid_block_component (dim+1,0);
  fluid_block_component[dim] = 1;
  DoFRenumbering::component_wise (fluid_dof_handler, fluid_block_component);

  std::vector<unsigned int> structure_block_component (2*dim,0);
  for (unsigned int i=dim; i<2*dim; ++i)
    structure_block_component[i] = 1;
  DoFRenumbering::component_wise (structure_dof_handler, structure_block_component);

  std::vector<unsigned int> ale_block_component (dim,0);
  DoFRenumbering::component_wise (ale_dof_handler, ale_block_component);

  {
    AssertThrow(n_blocks==5,ExcNotImplemented());

    std::vector<types::global_dof_index> fluid_dofs_per_block (2);
    DoFTools::count_dofs_per_block (fluid_dof_handler, fluid_dofs_per_block, fluid_block_component);
    dofs_per_block[0]=fluid_dofs_per_block[0];
    dofs_per_block[1]=fluid_dofs_per_block[1];

    std::vector<types::global_dof_index> structure_dofs_per_block (2);
    DoFTools::count_dofs_per_block (structure_dof_handler, structure_dofs_per_block, structure_block_component);
    dofs_per_block[2]=structure_dofs_per_block[0];
    dofs_per_block[3]=structure_dofs_per_block[1];

    std::vector<types::global_dof_index> ale_dofs_per_block (1);
    DoFTools::count_dofs_per_block (ale_dof_handler, ale_dofs_per_block, ale_block_component);
    dofs_per_block[4]=ale_dofs_per_block[0];
  }


  std::cout << "Number of degrees of freedom: "
	    << fluid_dof_handler.n_dofs() + structure_dof_handler.n_dofs() + ale_dof_handler.n_dofs()
	    << " (" << dofs_per_block[0] << '+' << dofs_per_block[1]
	    << '+' << dofs_per_block[2] << '+' << dofs_per_block[3] << '+' << dofs_per_block[4] << ')'
	    << std::endl;

  //BlockCompressedSparsityPattern csp_alt (n_big_blocks,n_big_blocks);
  BlockCompressedSimpleSparsityPattern csp (n_big_blocks,n_big_blocks);

  dofs_per_big_block.push_back(dofs_per_block[0]+dofs_per_block[1]);
  dofs_per_big_block.push_back(dofs_per_block[2]+dofs_per_block[3]);
  dofs_per_big_block.push_back(dofs_per_block[4]);

  // if (physical_properties.stability_terms) 
  //   {
  //     for (unsigned int i=0; i<n_big_blocks; ++i)
  // 	for (unsigned int j=0; j<n_big_blocks; ++j)
  // 	  csp_alt.block(i,j).reinit (dofs_per_big_block[i], dofs_per_big_block[j]);

  //     csp_alt.collect_sizes();

  //     DoFTools::make_sparsity_pattern (fluid_dof_handler, csp_alt.block(0,0), fluid_constraints, true);
  //     DoFTools::make_sparsity_pattern (structure_dof_handler, csp_alt.block(1,1), structure_constraints, true);
  //     DoFTools::make_sparsity_pattern (ale_dof_handler, csp_alt.block(2,2), ale_constraints, true);

  //     fluid_constraints.condense(csp_alt.block(0,0));
  //     structure_constraints.condense(csp_alt.block(1,1));
  //     ale_constraints.condense(csp_alt.block(2,2));

  //     sparsity_pattern.copy_from (csp_alt);
  //   }
  // else
  //   {
      for (unsigned int i=0; i<n_big_blocks; ++i)
	for (unsigned int j=0; j<n_big_blocks; ++j)
	  csp.block(i,j).reinit (dofs_per_big_block[i], dofs_per_big_block[j]);

      csp.collect_sizes();

      DoFTools::make_sparsity_pattern (fluid_dof_handler, csp.block(0,0), fluid_constraints, false);
      DoFTools::make_sparsity_pattern (structure_dof_handler, csp.block(1,1), structure_constraints, false);
      DoFTools::make_sparsity_pattern (ale_dof_handler, csp.block(2,2), ale_constraints, false);

      sparsity_pattern.copy_from (csp);
    // }


  system_matrix.reinit (sparsity_pattern);
  adjoint_matrix.reinit (sparsity_pattern);
  linear_matrix.reinit (sparsity_pattern);

  solution.reinit (n_big_blocks);
  solution_star.reinit (n_big_blocks);
  rhs_for_adjoint.reinit(n_big_blocks);
  rhs_for_adjoint_s.reinit(n_big_blocks);
  rhs_for_linear.reinit(n_big_blocks);
  rhs_for_linear_h.reinit(n_big_blocks);
  rhs_for_linear_p.reinit(n_big_blocks);
  rhs_for_linear_Ap.reinit(n_big_blocks);
  rhs_for_linear_Ap_s.reinit(n_big_blocks);
  premultiplier.reinit(n_big_blocks);
  adjoint_solution.reinit (n_big_blocks);
  linear_solution.reinit (n_big_blocks);
  tmp.reinit (n_big_blocks);
  tmp2.reinit (n_big_blocks);
  old_solution.reinit (n_big_blocks);
  old_old_solution.reinit (n_big_blocks);
  system_rhs.reinit (n_big_blocks);
  adjoint_rhs.reinit (n_big_blocks);
  linear_rhs.reinit (n_big_blocks);
  stress.reinit (n_big_blocks);
  old_stress.reinit (n_big_blocks);
  mesh_displacement_star.reinit (n_big_blocks);
  mesh_displacement_star_old.reinit (n_big_blocks);
  old_mesh_displacement.reinit (n_big_blocks);
  mesh_velocity.reinit (n_big_blocks);
  for (unsigned int i=0; i<n_big_blocks; ++i)
    {
      solution.block(i).reinit (dofs_per_big_block[i]);
      solution_star.block(i).reinit (dofs_per_big_block[i]);
      rhs_for_adjoint.block(i).reinit(dofs_per_big_block[i]);
      rhs_for_adjoint_s.block(i).reinit(dofs_per_big_block[i]);
      rhs_for_linear.block(i).reinit(dofs_per_big_block[i]);
      rhs_for_linear_h.block(i).reinit(dofs_per_big_block[i]);
      rhs_for_linear_p.block(i).reinit(dofs_per_big_block[i]);
      rhs_for_linear_Ap.block(i).reinit(dofs_per_big_block[i]);
      rhs_for_linear_Ap_s.block(i).reinit(dofs_per_big_block[i]);
      premultiplier.block(i).reinit(dofs_per_big_block[i]);
      adjoint_solution.block(i).reinit (dofs_per_big_block[i]);
      linear_solution.block(i).reinit (dofs_per_big_block[i]);
      tmp.block(i).reinit (dofs_per_big_block[i]);
      tmp2.block(i).reinit (dofs_per_big_block[i]);
      old_solution.block(i).reinit (dofs_per_big_block[i]);
      old_old_solution.block(i).reinit (dofs_per_big_block[i]);
      system_rhs.block(i).reinit (dofs_per_big_block[i]);
      adjoint_rhs.block(i).reinit (dofs_per_big_block[i]);
      linear_rhs.block(i).reinit (dofs_per_big_block[i]);
      stress.block(i).reinit (dofs_per_big_block[i]);
      old_stress.block(i).reinit (dofs_per_big_block[i]);
      mesh_displacement_star.block(i).reinit (dofs_per_big_block[i]);
      mesh_displacement_star_old.block(i).reinit (dofs_per_big_block[i]);
      old_mesh_displacement.block(i).reinit (dofs_per_big_block[i]);
      mesh_velocity.block(i).reinit (dofs_per_big_block[i]);
    }
  solution.collect_sizes ();
  solution_star.collect_sizes ();
  rhs_for_adjoint.collect_sizes ();
  rhs_for_adjoint_s.collect_sizes ();
  rhs_for_linear.collect_sizes ();
  rhs_for_linear_h.collect_sizes ();
  rhs_for_linear_p.collect_sizes ();
  rhs_for_linear_Ap.collect_sizes ();
  rhs_for_linear_Ap_s.collect_sizes ();
  premultiplier.collect_sizes ();
  adjoint_solution.collect_sizes ();
  linear_solution.collect_sizes ();
  tmp.collect_sizes ();
  tmp2.collect_sizes ();
  old_solution.collect_sizes ();
  old_old_solution.collect_sizes ();
  system_rhs.collect_sizes ();
  adjoint_rhs.collect_sizes ();
  linear_rhs.collect_sizes ();
  stress.collect_sizes ();
  old_stress.collect_sizes ();
  mesh_displacement_star.collect_sizes ();
  mesh_displacement_star.collect_sizes ();
  old_mesh_displacement.collect_sizes ();
  mesh_velocity.collect_sizes ();

  fluid_constraints.close ();
  structure_constraints.close ();
  ale_constraints.close ();
}


template void FSIProblem<2>::dirichlet_boundaries (System system, Mode enum_);
template void FSIProblem<2>::setup_system ();
