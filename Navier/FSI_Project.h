#ifndef FSI_PROJECT_H
#define FSI_PROJECT_H

/* ---------------------------------------------------------------------
 *  Time Dependent FSI Problem with ALE on Fluid Domain

 * ---------------------------------------------------------------------
 *
 * Originally authored by Wolfgang Bangerth, Texas A&M University, 2006
 * and amended significantly by Paul Kuberry, Clemson University, 2014
 */


#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/convergence_table.h>
#include <deal.II/lac/block_vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/block_sparse_matrix.h>
#include <deal.II/lac/sparse_direct.h>
#include <deal.II/lac/constraint_matrix.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/fe_field_function.h>
#include <deal.II/base/utilities.h>
#include <deal.II/base/parameter_handler.h>
#include <deal.II/base/function_parser.h>

#include <deal.II/base/work_stream.h>
#include <deal.II/base/multithread_info.h>

#include <fstream>
#include <iostream>
#include <boost/timer.hpp>

#include "parameters.h"
#include "small_classes.h"
#include "data1.h"

using namespace dealii;

template <int dim>
class FSIProblem
{
 public:
  //static void
  //declare_parameters (ParameterHandler & prm);
  FSIProblem (ParameterHandler & prm);
  void run ();

  ~FSIProblem ();


 private:
  enum Mode
  {
    state,
    adjoint,
    linear
  };
  enum BoundaryCondition
  {
    Dirichlet,
    Neumann,
    Interface,
    DoNothing
  };
  enum System
  {
    Fluid,
    Structure,
    ALE
  };
  enum Optimization
  {
    CG,
    Gradient
  };
  enum StructureComponent
  {
    Displacement,
    Velocity,
    NotSet
  };

  void assemble_fluid (Mode enum_, bool assemble_matrix);
  void assemble_fluid_matrix_on_one_cell (const typename DoFHandler<dim>::active_cell_iterator& cell,
							     FluidScratchData<dim>& scratch,
							     PerTaskData<dim>& data );
  void copy_local_fluid_to_global (const PerTaskData<dim> &data);

  void assemble_structure(Mode enum_, bool assemble_matrix);
  void assemble_structure_matrix_on_one_cell (const typename DoFHandler<dim>::active_cell_iterator& cell,
							     FullScratchData<dim>& scratch,
							     PerTaskData<dim>& data );
  void copy_local_structure_to_global (const PerTaskData<dim> &data);

  void assemble_ale(Mode enum_, bool assemble_matrix);
  void assemble_ale_matrix_on_one_cell (const typename DoFHandler<dim>::active_cell_iterator &cell,
					BaseScratchData<dim> &scratch,
					PerTaskData<dim> &data);
  void copy_local_ale_to_global (const PerTaskData<dim> &data);

  void build_adjoint_rhs();
  double interface_error();
  double interface_norm(Vector<double>   &values);
  void dirichlet_boundaries(System system, Mode enum_);
  void build_dof_mapping();
  void transfer_interface_dofs(BlockVector<double> & solution_1, BlockVector<double> & solution_2, unsigned int from, unsigned int to, StructureComponent structure_var_1=NotSet, StructureComponent structure_var_2=NotSet);
  void transfer_all_dofs(BlockVector<double> & solution_1, BlockVector<double> & solution_2, unsigned int from, unsigned int to);
  void setup_system ();
  void solve (const SparseDirectUMFPACK& direct_solver, const int block_num, Mode enum_);
  void output_results () const;
  void compute_error ();

  Triangulation<dim>   	fluid_triangulation, structure_triangulation;
  FESystem<dim>  	    	fluid_fe, structure_fe, ale_fe;
  DoFHandler<dim>      	fluid_dof_handler, structure_dof_handler, ale_dof_handler;

  ConstraintMatrix fluid_constraints, structure_constraints, ale_constraints;

  BlockSparsityPattern       sparsity_pattern;
  BlockSparseMatrix<double>  system_matrix;
  BlockSparseMatrix<double>  adjoint_matrix;
  BlockSparseMatrix<double>  linear_matrix;

  BlockVector<double>       	solution;
  BlockVector<double>       	solution_star;
  BlockVector<double>		rhs_for_adjoint;
  BlockVector<double>		rhs_for_adjoint_s;
  BlockVector<double>		rhs_for_linear;
  BlockVector<double>		rhs_for_linear_h;
  BlockVector<double>		rhs_for_linear_p;
  BlockVector<double>		rhs_for_linear_Ap_s;
  BlockVector<double>		premultiplier;
  BlockVector<double>		adjoint_solution;
  BlockVector<double>		linear_solution;
  BlockVector<double>		tmp, tmp2;
  BlockVector<double>       	old_solution;
  BlockVector<double>       	old_old_solution;
  BlockVector<double>       	system_rhs;
  BlockVector<double>       	adjoint_rhs;    
  BlockVector<double>       	linear_rhs;
  BlockVector<double>		stress;
  BlockVector<double>		old_stress;
  BlockVector<double>		mesh_displacement_star;
  BlockVector<double>		mesh_displacement_star_old;
  BlockVector<double>		old_mesh_displacement;
  BlockVector<double>		mesh_velocity;

  double time, time_step;
  unsigned int timestep_number;
  const double fluid_theta;
  const double structure_theta;
  Parameters::ComputationData errors;
  const unsigned int n_blocks;
  const unsigned int n_big_blocks;
  std::vector<unsigned int> dofs_per_block;
  std::vector<unsigned int> dofs_per_big_block;
  Parameters::SimulationProperties fem_properties;
  Parameters::PhysicalProperties physical_properties;
  std::vector<unsigned int> fluid_interface_cells, fluid_interface_faces;
  std::vector<unsigned int> structure_interface_cells, structure_interface_faces;
  std::map<unsigned int, unsigned int> f2n, n2f, f2v, v2f, n2a, a2n, a2v, v2a, a2f, f2a, n2v, v2n, a2f_all, f2a_all;
  std::map<unsigned int, BoundaryCondition> fluid_boundaries, structure_boundaries, ale_boundaries;
  std::vector<SparseDirectUMFPACK > state_solver,  adjoint_solver,  linear_solver;

  unsigned int master_thread;
  bool update_domain;
};




template <int dim>
FSIProblem<dim>::FSIProblem (ParameterHandler & prm_) :
fluid_fe (FE_Q<dim>(prm_.get_integer("fluid velocity degree")), dim,
	  FE_Q<dim>(prm_.get_integer("fluid pressure degree")), 1),
  structure_fe (FE_Q<dim>(prm_.get_integer("structure degree")), dim,
		FE_Q<dim>(prm_.get_integer("structure degree")), dim),
  ale_fe (FE_Q<dim>(prm_.get_integer("ale degree")), dim),
  fluid_dof_handler (fluid_triangulation),
  structure_dof_handler (structure_triangulation),
  ale_dof_handler (fluid_triangulation),
  time_step ((prm_.get_double("T")-prm_.get_double("t0"))/prm_.get_integer("number of time steps")),
  timestep_number(0),
  fluid_theta(prm_.get_double("fluid theta")),
  structure_theta(prm_.get_double("structure theta")),
  errors(),
  n_blocks(5),
  n_big_blocks(3),
  dofs_per_block(5),
  state_solver(3),  
  adjoint_solver(3),
  linear_solver(3)
{
  fem_properties.fluid_degree		= prm_.get_integer("fluid velocity degree");
  fem_properties.pressure_degree	= prm_.get_integer("fluid pressure degree");
  fem_properties.structure_degree	= prm_.get_integer("structure degree");
  fem_properties.ale_degree		= prm_.get_integer("ale degree");
  // Time Parameters
  fem_properties.t0                     = prm_.get_double("t0");
  fem_properties.T			= prm_.get_double("T");
  fem_properties.n_time_steps		= prm_.get_integer("number of time steps");
  fem_properties.fluid_theta		= prm_.get_double("fluid theta");
  fem_properties.structure_theta	= prm_.get_double("structure theta");
  // Domain Parameters
  fem_properties.fluid_width		= prm_.get_double("fluid width");
  fem_properties.fluid_height		= prm_.get_double("fluid height");
  fem_properties.structure_width	= prm_.get_double("structure width");
  fem_properties.structure_height	= prm_.get_double("structure height");
  fem_properties.nx_f			= prm_.get_integer("nx fluid");
  fem_properties.ny_f			= prm_.get_integer("ny fluid");
  fem_properties.nx_s			= prm_.get_integer("nx structure");
  fem_properties.ny_s			= prm_.get_integer("ny structure");
  // Output Parameters
  fem_properties.make_plots		= prm_.get_bool("make plots");
  fem_properties.print_error		= prm_.get_bool("output error");
  fem_properties.convergence_mode	= prm_.get("convergence method");
  // Optimization Parameters
  fem_properties.jump_tolerance		= prm_.get_double("jump tolerance");
  fem_properties.cg_tolerance		= prm_.get_double("cg tolerance");
  fem_properties.steepest_descent_alpha = prm_.get_double("steepest descent alpha");
  fem_properties.penalty_epsilon	= prm_.get_double("penalty epsilon");
  fem_properties.max_optimization_iterations = prm_.get_integer("max optimization iterations");
  fem_properties.true_control           = prm_.get_bool("true control");
  fem_properties.optimization_method    = prm_.get("optimization method");
  fem_properties.adjoint_type           = prm_.get_integer("adjoint type");
  // Solver Parameters
  fem_properties.richardson		= prm_.get_bool("richardson");
  fem_properties.newton 		= prm_.get_bool("newton");
  physical_properties.moving_domain	= prm_.get_bool("moving domain");
  physical_properties.move_domain	= prm_.get_bool("move domain");

  // Problem Parameters
  physical_properties.simulation_type   = prm_.get_integer("simulation type");
  physical_properties.viscosity		= prm_.get_double("viscosity");
  physical_properties.lambda		= prm_.get_double("lambda");
  physical_properties.mu		= prm_.get_double("mu");
  physical_properties.nu		= prm_.get_double("nu");
  physical_properties.navier_stokes     = prm_.get_bool("navier stokes");
  physical_properties.stability_terms   = prm_.get_bool("stability terms");
  if (std::fabs(physical_properties.lambda)<1e-13) // Lambda is to be computed
    {
      physical_properties.lambda	= 2*physical_properties.mu*physical_properties.nu/(1-2*physical_properties.nu);
    }
  else if (std::fabs(physical_properties.mu)<1e-13) // Mu is to be computed
    {
      physical_properties.mu		= physical_properties.lambda*(1-2*physical_properties.nu)/2*physical_properties.nu;
    }
  //else  We don't need to compute anything
  physical_properties.rho_f				= prm_.get_double("fluid rho");
  physical_properties.rho_s				= prm_.get_double("structure rho");
  physical_properties.n_fourier_coeffs	= prm_.get_integer("number fourier coefficients");
}

template <int dim>
FSIProblem<dim>::~FSIProblem ()
{
  fluid_dof_handler.clear();
  structure_dof_handler.clear();
}

#endif
