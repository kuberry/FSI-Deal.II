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
    Interface
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
  void assemble_structure(Mode enum_, bool assemble_matrix);
  void assemble_ale(Mode enum_, bool assemble_matrix);
  void assemble_ale_matrix_on_one_cell (const typename DoFHandler<dim>::active_cell_iterator &cell,
					ScratchData<dim> &scratch,
					PerTaskData<dim> &data);
  /* 							 unsigned int n_q_points, */
  /* 							 unsigned int dofs_per_cell); */
  void copy_local_matrix_to_global (const PerTaskData<dim> &data);
  /* 						     unsigned int dofs_per_cell,  */
  /* 						     SparseMatrix<double>& global_matrix,  */
  /* 						     Vector<double>& global_rhs); */
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
  BlockVector<double>		mesh_displacement;
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


template <int dim>
class FluidStressValues : public Function<dim>
{
 public:
  Parameters::PhysicalProperties physical_properties;
  FluidStressValues (const Parameters::PhysicalProperties & physical_properties_) : Function<dim>(dim+1), physical_properties(physical_properties_)  {}

  virtual double value (const Point<dim>   &p,
			const unsigned int  component = 0) const;
  virtual Tensor<1,dim> gradient (const Point<dim>   &p,
				  const unsigned int  component = 0) const;
};

template <int dim>
double FluidStressValues<dim>::value (const Point<dim>  &p,
				      const unsigned int component) const
{
  // This function can be deleted later. It is just for interpolating g on the boundary
  /*
   * u1=cos(t + x)*sin(t + y) + cos(t + y)*sin(t + x);
   * u2=- cos(t + x)*sin(t + y) - cos(t + y)*sin(t + x);
   * p=2*lambda*cos(t + x)*sin(t + y) - 2*viscosity*(cos(t + x)*cos(t + y) - sin(t + x)*sin(t + y));
   * 2*viscosity*(diff(u1,x)*n1+0.5*(diff(u1,y)+diff(u2,x))*n2)-p*n1*u1
   * 2*viscosity*(diff(u2,y)*n2+0.5*(diff(u1,y)+diff(u2,x))*n1)-p*n2*u2
   *
   */
  Tensor<1,dim> result;
  const double t = this->get_time();
  const double x = p[0];
  const double y = p[1];
  const double u1_x = cos(t + x)*cos(t + y) - sin(t + x)*sin(t + y);
  const double u2_y = sin(t + x)*sin(t + y) - cos(t + x)*cos(t + y);
  const double u1_y = cos(t + x)*cos(t + y) - sin(t + x)*sin(t + y);
  const double u2_x = sin(t + x)*sin(t + y) - cos(t + x)*cos(t + y);
  const double pval = 2*physical_properties.mu*cos(t + x)*sin(t + y) - 2*physical_properties.viscosity*(cos(t + x)*cos(t + y)
													- sin(t + x)*sin(t + y));

  double answer = 0;
  switch (component)
    {
    case 0:
      result[0]=2*physical_properties.viscosity*u1_x-pval;
      result[1]=2*physical_properties.viscosity*0.5*(u1_y+u2_x);
      break;
    case 1:
      result[0]=2*physical_properties.viscosity*0.5*(u1_y+u2_x);
      result[1]=2*physical_properties.viscosity*u2_y-pval;
      break;
    default:
      result=0;
    }
  answer = result[0]*0+result[1]*1;
  return answer;
}


template <int dim>
Tensor<1,dim> FluidStressValues<dim>::gradient (const Point<dim>  &p,
						const unsigned int component) const
{
  /*
   * u1=cos(t + x)*sin(t + y) + cos(t + y)*sin(t + x);
   * u2=- cos(t + x)*sin(t + y) - cos(t + y)*sin(t + x);
   * p=2*lambda*cos(t + x)*sin(t + y) - 2*viscosity*(cos(t + x)*cos(t + y) - sin(t + x)*sin(t + y));
   * 2*viscosity*(diff(u1,x)*n1+0.5*(diff(u1,y)+diff(u2,x))*n2)-p*n1*u1
   * 2*viscosity*(diff(u2,y)*n2+0.5*(diff(u1,y)+diff(u2,x))*n1)-p*n2*u2
   *
   */
  Tensor<1,dim> result;
  const double t = this->get_time();
  const double x = p[0];
  const double y = p[1];
  const double u1_x = cos(t + x)*cos(t + y) - sin(t + x)*sin(t + y);
  const double u2_y = sin(t + x)*sin(t + y) - cos(t + x)*cos(t + y);
  const double u1_y = cos(t + x)*cos(t + y) - sin(t + x)*sin(t + y);
  const double u2_x = sin(t + x)*sin(t + y) - cos(t + x)*cos(t + y);
  const double pval = 2*physical_properties.mu*cos(t + x)*sin(t + y) - 2*physical_properties.viscosity*(cos(t + x)*cos(t + y)
													- sin(t + x)*sin(t + y));

  switch (component)
    {
    case 0:
      result[0]=2*physical_properties.viscosity*u1_x-pval;
      result[1]=2*physical_properties.viscosity*0.5*(u1_y+u2_x);
      return result;
    case 1:
      result[0]=2*physical_properties.viscosity*0.5*(u1_y+u2_x);
      result[1]=2*physical_properties.viscosity*u2_y-pval;
      return result;
    default:
      result=0;
      return result;
    }
}

template <int dim>
class StructureStressValues : public Function<dim>
{
 public:
  Parameters::PhysicalProperties physical_properties;
  unsigned int side;
  StructureStressValues (const Parameters::PhysicalProperties & physical_properties_) : Function<dim>(dim+1), physical_properties(physical_properties_)  {}

  virtual double value (const Point<dim>   &p,
			const unsigned int  component = 0) const;
  virtual Tensor<1,dim> gradient (const Point<dim>   &p,
				  const unsigned int  component = 0) const;
};
template <int dim>
double StructureStressValues<dim>::value (const Point<dim>  &p,
					  const unsigned int component) const
{
  /*
    >> n1=sin(x + t)*sin(y + t);
    >> n2=cos(x + t)*cos(y + t);
  */
  Tensor<1,dim> result;
  const double t = this->get_time();
  const double x = p[0];
  const double y = p[1];
  const double n1_x =cos(t + x)*sin(t + y);
  const double n2_y =-cos(t + x)*sin(t + y);
  const double n1_y =cos(t + y)*sin(t + x);
  const double n2_x =-cos(t + y)*sin(t + x);
  switch (component)
    {
    case 0:
      result[0]=2*physical_properties.mu*n1_x+physical_properties.lambda*(n1_x+n2_y);
      result[1]=2*physical_properties.mu*0.5*(n1_y+n2_x);
      break;
    case 1:
      result[0]=2*physical_properties.mu*0.5*(n1_y+n2_x);
      result[1]=2*physical_properties.mu*n2_y+physical_properties.lambda*(n1_x+n2_y);
      break;
    default:
      result=0;
    }
  return result[0]*0+result[1]*(-1);
}
template <int dim>
Tensor<1,dim> StructureStressValues<dim>::gradient (const Point<dim>  &p,
						    const unsigned int component) const
{
  /*
    >> n1=sin(x + t)*sin(y + t);
    >> n2=cos(x + t)*cos(y + t);
  */
  Tensor<1,dim> result;
  const double t = this->get_time();
  const double x = p[0];
  const double y = p[1];
  const double n1_x =cos(t + x)*sin(t + y);
  const double n2_y =-cos(t + x)*sin(t + y);
  const double n1_y =cos(t + y)*sin(t + x);
  const double n2_x =-cos(t + y)*sin(t + x);
  switch (component)
    {
    case 0:
      result[0]=2*physical_properties.mu*n1_x+physical_properties.lambda*(n1_x+n2_y);
      result[1]=2*physical_properties.mu*0.5*(n1_y+n2_x);
      return result;
    case 1:
      result[0]=2*physical_properties.mu*0.5*(n1_y+n2_x);
      result[1]=2*physical_properties.mu*n2_y+physical_properties.lambda*(n1_x+n2_y);
      return result;
    default:
      result=0;
      return result;
    }
}

template <int dim>
class FluidRightHandSide : public Function<dim>
{
 public:
  Parameters::PhysicalProperties physical_properties;
  FluidRightHandSide (const Parameters::PhysicalProperties & physical_properties_) : Function<dim>(dim+1), physical_properties(physical_properties_)  {}

  virtual double value (const Point<dim>   &p,
			const unsigned int  component = 0) const;
};
template <int dim>
double FluidRightHandSide<dim>::value (const Point<dim>  &p,
				       const unsigned int component) const
{
  const double t = this->get_time();
  const double x = p[0];
  const double y = p[1];
  // >> u1=cos(t + x)*sin(t + y) + cos(t + y)*sin(t + x);
  // >> u2=- cos(t + x)*sin(t + y) - cos(t + y)*sin(t + x);
  // >> p=2*mu*cos(t + x)*sin(t + y) - 2*viscosity*(cos(t + x)*cos(t + y) - sin(t + x)*sin(t + y));
  // >> rho_f*diff(u1,t)-2*viscosity*(diff(diff(u1,x),x)+0.5*(diff(diff(u1,y),y)+diff(diff(u2,x),y)))+diff(p,x) + convection
  // >> rho_f*diff(u2,t)-2*viscosity*(diff(diff(u2,y),y)+0.5*(diff(diff(u2,x),x)+diff(diff(u1,y),x)))+diff(p,y) + convection
  switch (component)
    {
    case 0:
      return physical_properties.rho_f*(2*cos(t + x)*cos(t + y) - 2*sin(t + x)*sin(t + y)) + 4*physical_properties.viscosity 
	* (cos(t + x)*sin(t + y) + cos(t + y)*sin(t + x)) - 2*physical_properties.mu*sin(t + x)*sin(t + y);
      //+ physical_properties.rho_f*(2*(cos(t + x)*sin(t + y) + cos(t + y)*sin(t + x))*(cos(t + x)*cos(t + y) - sin(t + x)*sin(t + y)));
    case 1:
      return 2*physical_properties.mu*cos(t + x)*cos(t + y) - physical_properties.rho_f*(2*cos(t + x)*cos(t + y) - 2*sin(t + x)*sin(t + y));
      //+ physical_properties.rho_f*(2*(cos(t + x)*sin(t + y) + cos(t + y)*sin(t + x))*(cos(t + x)*cos(t + y) - sin(t + x)*sin(t + y)));
    case 2:
      return 0;
    default:
      return 0;
    }
}
template <int dim>
class StructureRightHandSide : public Function<dim>
{
 public:
  Parameters::PhysicalProperties physical_properties;
  StructureRightHandSide (const Parameters::PhysicalProperties & physical_properties_) : Function<dim>(2*dim), physical_properties(physical_properties_) {}

  virtual double value (const Point<dim>   &p,
			const unsigned int  component = 0) const;
};
template <int dim>
double StructureRightHandSide<dim>::value (const Point<dim>  &p,
					   const unsigned int component) const
{
  /*
    >> n1=sin(x + t)*sin(y + t);
    >> n2=cos(x + t)*cos(y + t);
    >> rho_s*diff(diff(n1,t),t)-2*mu*(diff(diff(n1,x),x)+0.5*(diff(diff(n1,y),y)+diff(diff(n2,x),y)))-lambda*(diff(diff(n1,x),x)+diff(diff(n2,y),x))
    >> rho_s*diff(diff(n2,t),t)-2*mu*(diff(diff(n2,y),y)+0.5*(diff(diff(n2,x),x)+diff(diff(n1,y),x)))-lambda*(diff(diff(n1,x),y)+diff(diff(n2,y),y))
  */
  const double t = this->get_time();
  const double x = p[0];
  const double y = p[1];
  switch (component)
    {
    case 0:
      return physical_properties.rho_s*(2*cos(t + x)*cos(t + y) - 2*sin(t + x)*sin(t + y)) + 2*physical_properties.mu*sin(t + x)*sin(t + y);
    case 1:
      return 2*physical_properties.mu*cos(t + x)*cos(t + y) - physical_properties.rho_s*(2*cos(t + x)*cos(t + y) - 2*sin(t + x)*sin(t + y));
    default:
      return 0;
    }
}


template <int dim>
class FluidBoundaryValues : public Function<dim>
{
 public:
  Parameters::PhysicalProperties physical_properties;
  FluidBoundaryValues (const Parameters::PhysicalProperties & physical_properties_) : Function<dim>(dim+1), physical_properties(physical_properties_) {}

  virtual double value (const Point<dim>   &p,
			const unsigned int  component = 0) const;
  virtual void vector_value (const Point<dim> &p,
			     Vector<double>   &value) const;
  virtual Tensor<1,dim> gradient (const Point<dim>   &p,
				  const unsigned int  component = 0) const;
};
template <int dim>
double FluidBoundaryValues<dim>::value (const Point<dim> &p,
					const unsigned int component) const
{
  Assert (component < 3, ExcInternalError());
  const double t = this->get_time();
  const double x = p[0];
  const double y = p[1];
  switch (component)
    {
    case 0:
      return cos(x + t)*sin(y + t) + sin(x + t)*cos(y + t);
    case 1:
      return -sin(x + t)*cos(y + t) - cos(x + t)*sin(y + t);
    case 2:
      return 2*physical_properties.viscosity*(sin(x + t)*sin(y + t) - cos(x + t)*cos(y + t)) + 2*physical_properties.mu*cos(x + t)*sin(y + t);
    default:
      return 0;
    }
}
template <int dim>
void
FluidBoundaryValues<dim>::vector_value (const Point<dim> &p,
					Vector<double>   &values) const
{
  for (unsigned int c=0; c<this->n_components; ++c)
    {
      values(c) = FluidBoundaryValues<dim>::value (p, c);
    }
}
template <int dim>
Tensor<1,dim> FluidBoundaryValues<dim>::gradient (const Point<dim>   &p,
						  const unsigned int component) const
{
  Tensor<1,dim> result;
  const double t = this->get_time();
  const double x = p[0];
  const double y = p[1];
  const double u1_x = cos(t + x)*cos(t + y) - sin(t + x)*sin(t + y);
  const double u2_y = sin(t + x)*sin(t + y) - cos(t + x)*cos(t + y);
  const double u1_y = cos(t + x)*cos(t + y) - sin(t + x)*sin(t + y);
  const double u2_x = sin(t + x)*sin(t + y) - cos(t + x)*cos(t + y);

  switch (component)
    {
    case 0:
      result[0]=u1_x;
      result[1]=u1_y;
      return result;
    case 1:
      result[0]=u2_x;
      result[1]=u2_y;
      return result;
    default:
      result=0;
      return result;
    }
}
template <int dim>
class StructureBoundaryValues : public Function<dim>
{
 public:
  Parameters::PhysicalProperties physical_properties;
  StructureBoundaryValues (const Parameters::PhysicalProperties & physical_properties_) : Function<dim>(2*dim), physical_properties(physical_properties_) {}

  virtual double value (const Point<dim>   &p,
			const unsigned int  component = 0) const;
  virtual void vector_value (const Point<dim> &p,
			     Vector<double>   &value) const;
  virtual Tensor<1,dim> gradient (const Point<dim>   &p,
				  const unsigned int  component = 0) const;
};
template <int dim>
double StructureBoundaryValues<dim>::value (const Point<dim> &p,
					    const unsigned int component) const
{
  /*
    >> n1=sin(x + t)*sin(y + t);
    >> n2=cos(x + t)*cos(y + t);
  */
  Assert (component < 4, ExcInternalError());
  const double t = this->get_time();
  const double x = p[0];
  const double y = p[1];
  switch (component)
    {
    case 0:
      return sin(x + t)*sin(y + t);
    case 1:
      return cos(x + t)*cos(y + t);
    case 2:
      return sin(x + t)*cos(y + t)+cos(x + t)*sin(y + t);
    case 3:
      return -sin(x + t)*cos(y + t)-cos(x + t)*sin(y + t);
    default:
      Assert(false,ExcDimensionMismatch(5,4));
      return 0;
    }
}
template <int dim>
void
StructureBoundaryValues<dim>::vector_value (const Point<dim> &p,
					    Vector<double>   &values) const
{
  for (unsigned int c=0; c<this->n_components; ++c)
    {
      values(c) = StructureBoundaryValues<dim>::value (p, c);
    }
}
template <int dim>
Tensor<1,dim> StructureBoundaryValues<dim>::gradient (const Point<dim>   &p,
						      const unsigned int component) const
{
  /*
    >> n1=sin(x + t)*sin(y + t);
    >> n2=cos(x + t)*cos(y + t);
  */
  Tensor<1,dim> result;
  const double t = this->get_time();
  const double x = p[0];
  const double y = p[1];
  const double n1_x =cos(t + x)*sin(t + y);
  const double n2_y =-cos(t + x)*sin(t + y);
  const double n1_y =cos(t + y)*sin(t + x);
  const double n2_x =-cos(t + y)*sin(t + x);

  switch (component)
    {
    case 0:
      result[0]=n1_x;
      result[1]=n1_y;
      return result;
    case 1:
      result[0]=n2_x;
      result[1]=n2_y;
      return result;
    default:
      result=0;
      return result;//
    }
}
template <int dim>
class AleBoundaryValues : public Function<dim>
{
 public:
  Parameters::PhysicalProperties physical_properties;
  AleBoundaryValues (const Parameters::PhysicalProperties & physical_properties_) : Function<dim>(dim), physical_properties(physical_properties_)  {}

  virtual double value (const Point<dim>   &p,
			const unsigned int  component = 0) const;
  virtual void vector_value (const Point<dim> &p,
			     Vector<double>   &value) const;
};
template <int dim>
double AleBoundaryValues<dim>::value (const Point<dim> &p,
                                      const unsigned int component) const
{
  Assert (component < dim, ExcInternalError());
  if (component==0)
    {
      return 3;
    }
  else
    {
      return 0;
    }
}
template <int dim>
void
AleBoundaryValues<dim>::vector_value (const Point<dim> &p,
				      Vector<double>   &values) const
{
  for (unsigned int c=0; c<this->n_components; ++c)
    {
      values(c) = AleBoundaryValues<dim>::value (p, c);
    }
}




#endif
