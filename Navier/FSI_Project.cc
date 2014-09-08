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
#include <fstream>
#include <iostream>
#include <boost/timer.hpp>

#include "parameters.h"


namespace FSI_Project
{
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
	const unsigned int   dofs_per_cell   = fluid_fe.dofs_per_cell;
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
	const unsigned int   dofs_per_cell   = fluid_fe.dofs_per_cell;
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

  template<int dim>
  class Info
  {
  public:
	unsigned int dof;
	Point<dim> coord;
	unsigned int component;
	Info(){};
	Info(const unsigned int dof_, Point<dim> & coord_, unsigned int component_):dof(dof_), coord(coord_), component(component_) {};
	static bool by_dof (const Info & first, const Info & second)
	{
		if (first.dof<second.dof)
		{
			return true;
		}
		else
		{
			return false;
		}
	}
	static bool by_point (const Info &first, const Info & second)
	{
		for (unsigned int i=0; i<dim; ++i)
		{
			if (first.coord[i]<second.coord[i])
			{
				return true;
			}
			else if (std::fabs(first.coord[i]-second.coord[i])>1e-13)
			{
				return false;
			}
		}
		if (first.component>second.component) return false;
		return true;
	}
	bool operator== (const Info &other) const
	{
		if (coord.distance(other.coord)>1e-10)
		{
			return false;
		}
		else
		{
			if (dof==other.dof) return true;
			else return false;
		}
	}
  };


  template <int dim>
  void FSIProblem<dim>::build_dof_mapping()
  {
	std::vector<Info<dim> > f_a;
	std::vector<Info<dim> > n_a;
	std::vector<Info<dim> > v_a;
	std::vector<Info<dim> > a_a;
	std::vector<Info<dim> > f_all;
	std::vector<Info<dim> > a_all;
	{
	typename DoFHandler<dim>::active_cell_iterator
	cell = fluid_dof_handler.begin_active(),
	endc = fluid_dof_handler.end();
	for (; cell!=endc; ++cell)
	{
		{
			std::vector<unsigned int> temp(fluid_fe.dofs_per_cell);
			cell->get_dof_indices(temp);
			Quadrature<dim> q(fluid_fe.get_unit_support_points());
			FEValues<dim> fe_values (fluid_fe, q,
									 update_quadrature_points);
			fe_values.reinit (cell);
			std::vector<Point<dim> > temp2(q.size());
			temp2=fe_values.get_quadrature_points();
			 for (unsigned int i=0;i<temp2.size();++i)
			 {
			         if (fluid_fe.system_to_component_index(i).first<dim) // <dim gives the velocities
				 {
					 f_all.push_back(Info<dim>(temp[i],temp2[i],fluid_fe.system_to_component_index(i).first));
				 }
			 }
		}
		for (unsigned int f=0; f<GeometryInfo<dim>::faces_per_cell; ++f)
		  if (cell->face(f)->boundary_indicator()==2)
		  {
			 std::vector<unsigned int> temp(2*fluid_dof_handler.get_fe()[0].dofs_per_vertex + fluid_dof_handler.get_fe()[0].dofs_per_line);
			 cell->face(f)->get_dof_indices(temp);
			 Quadrature<dim-1> q(fluid_fe.get_unit_face_support_points());
			 FEFaceValues<dim> fe_face_values (fluid_fe, q,
			                                      update_quadrature_points);
			 fe_face_values.reinit (cell, f);
			 std::vector<Point<dim> > temp2(q.size());
			 temp2=fe_face_values.get_quadrature_points();
			 for (unsigned int i=0;i<temp2.size();++i)
			 {
				 if (fluid_fe.system_to_component_index(i).first<dim)
				 {
					 f_a.push_back(Info<dim>(temp[i],temp2[i],fluid_fe.system_to_component_index(i).first));
				 }
			 }
		  }
	}
	 std::sort(f_a.begin(),f_a.end(),Info<dim>::by_dof);
	 f_a.erase( unique( f_a.begin(), f_a.end() ), f_a.end() );
	 std::sort(f_a.begin(),f_a.end(),Info<dim>::by_point);

	 std::sort(f_all.begin(),f_all.end(),Info<dim>::by_dof);
	 f_all.erase( unique( f_all.begin(), f_all.end() ), f_all.end() );
	 std::sort(f_all.begin(),f_all.end(),Info<dim>::by_point);
	}
	{
	typename DoFHandler<dim>::active_cell_iterator
	cell = structure_dof_handler.begin_active(),
	endc = structure_dof_handler.end();
	for (; cell!=endc; ++cell)
	{
		for (unsigned int f=0; f<GeometryInfo<dim>::faces_per_cell; ++f)
		  if (cell->face(f)->boundary_indicator()==0)
		  {
			 std::vector<unsigned int> temp(2*structure_dof_handler.get_fe()[0].dofs_per_vertex + structure_dof_handler.get_fe()[0].dofs_per_line);
			 cell->face(f)->get_dof_indices(temp);
			 Quadrature<dim-1> q(structure_fe.get_unit_face_support_points());
			 FEFaceValues<dim> fe_face_values (structure_fe, q,
			                                      update_quadrature_points);
			 fe_face_values.reinit (cell, f);
			 std::vector<Point<dim> > temp2(q.size());
			 temp2=fe_face_values.get_quadrature_points();
			 for (unsigned int i=0;i<temp2.size();++i)
			 {
			         if (structure_fe.system_to_component_index(i).first<dim) // this chooses displacement entries
				   {
				     n_a.push_back(Info<dim>(temp[i],temp2[i],structure_fe.system_to_component_index(i).first));
				   }
				 else
				   {
				     v_a.push_back(Info<dim>(temp[i],temp2[i],structure_fe.system_to_component_index(i).first));
				   }
			 }
		  }
	}
	 std::sort(n_a.begin(),n_a.end(),Info<dim>::by_dof);
	 n_a.erase( unique( n_a.begin(), n_a.end() ), n_a.end() );
	 std::sort(n_a.begin(),n_a.end(),Info<dim>::by_point);
	 std::sort(v_a.begin(),v_a.end(),Info<dim>::by_dof);
	 v_a.erase( unique( v_a.begin(), v_a.end() ), v_a.end() );
	 std::sort(v_a.begin(),v_a.end(),Info<dim>::by_point);
	}
	{
	typename DoFHandler<dim>::active_cell_iterator
	cell = ale_dof_handler.begin_active(),
	endc = ale_dof_handler.end();
	for (; cell!=endc; ++cell)
	{
		{
			std::vector<unsigned int> temp(ale_fe.dofs_per_cell);
			cell->get_dof_indices(temp);
			Quadrature<dim> q(ale_fe.get_unit_support_points());
			FEValues<dim> fe_values (ale_fe, q,
									 update_quadrature_points);
			fe_values.reinit (cell);
			std::vector<Point<dim> > temp2(q.size());
			temp2=fe_values.get_quadrature_points();
			 for (unsigned int i=0;i<temp2.size();++i)
			 {
				 if (ale_fe.system_to_component_index(i).first<dim)
				 {
					 a_all.push_back(Info<dim>(temp[i],temp2[i],ale_fe.system_to_component_index(i).first));
				 }
			 }
		}
		for (unsigned int f=0; f<GeometryInfo<dim>::faces_per_cell; ++f)
		  if (cell->face(f)->boundary_indicator()==2)
		  {
			 std::vector<unsigned int> temp(2*ale_dof_handler.get_fe()[0].dofs_per_vertex + ale_dof_handler.get_fe()[0].dofs_per_line);
			 cell->face(f)->get_dof_indices(temp);
			 Quadrature<dim-1> q(ale_fe.get_unit_face_support_points());
			 FEFaceValues<dim> fe_face_values (ale_fe, q,
			                                      update_quadrature_points);
			 fe_face_values.reinit (cell, f);
			 std::vector<Point<dim> > temp2(q.size());
			 temp2=fe_face_values.get_quadrature_points();
			 for (unsigned int i=0;i<temp2.size();++i)
			 {
				 if (ale_fe.system_to_component_index(i).first<dim)
				 {
					 a_a.push_back(Info<dim>(temp[i],temp2[i],ale_fe.system_to_component_index(i).first));
				 }
			 }
		  }
	}
	 std::sort(a_a.begin(),a_a.end(),Info<dim>::by_dof);
	 a_a.erase( unique( a_a.begin(), a_a.end() ), a_a.end() );
	 std::sort(a_a.begin(),a_a.end(),Info<dim>::by_point);

	 std::sort(a_all.begin(),a_all.end(),Info<dim>::by_dof);
	 a_all.erase( unique( a_all.begin(), a_all.end() ), a_all.end() );
	 std::sort(a_all.begin(),a_all.end(),Info<dim>::by_point);
	}
	for (unsigned int i=0; i<f_a.size(); ++i)
	{
		f2n.insert(std::pair<unsigned int,unsigned int>(f_a[i].dof,n_a[i].dof));
		n2f.insert(std::pair<unsigned int,unsigned int>(n_a[i].dof,f_a[i].dof));
		f2v.insert(std::pair<unsigned int,unsigned int>(f_a[i].dof,v_a[i].dof));
		v2f.insert(std::pair<unsigned int,unsigned int>(v_a[i].dof,f_a[i].dof));
		n2a.insert(std::pair<unsigned int,unsigned int>(n_a[i].dof,a_a[i].dof));
		a2n.insert(std::pair<unsigned int,unsigned int>(a_a[i].dof,n_a[i].dof));
		v2a.insert(std::pair<unsigned int,unsigned int>(v_a[i].dof,a_a[i].dof));
		a2v.insert(std::pair<unsigned int,unsigned int>(a_a[i].dof,v_a[i].dof));
		a2f.insert(std::pair<unsigned int,unsigned int>(a_a[i].dof,f_a[i].dof));
		f2a.insert(std::pair<unsigned int,unsigned int>(f_a[i].dof,a_a[i].dof));
		v2n.insert(std::pair<unsigned int,unsigned int>(v_a[i].dof,n_a[i].dof));
		n2v.insert(std::pair<unsigned int,unsigned int>(n_a[i].dof,v_a[i].dof));
	}
	for (unsigned int i=0; i<f_all.size(); ++i)
	{
		a2f_all.insert(std::pair<unsigned int,unsigned int>(a_all[i].dof,f_all[i].dof));
		f2a_all.insert(std::pair<unsigned int,unsigned int>(f_all[i].dof,a_all[i].dof));
	}
  }


  template <int dim>
  void FSIProblem<dim>::transfer_all_dofs(BlockVector<double> & solution_1, BlockVector<double> & solution_2, unsigned int from, unsigned int to)
   {
	  std::map<unsigned int, unsigned int> mapping;
	  if (from==2 && to==0)
	  {
		  mapping = a2f_all;
	  }
	  else if (from==0 && to==2)
	  {
		  mapping = f2a_all;
	  }
	  else
	  {
		  AssertThrow(false,ExcNotImplemented());
	  }
	  for  (std::map<unsigned int, unsigned int>::iterator it=mapping.begin(); it!=mapping.end(); ++it)
	  {
		  solution_2.block(to)[it->second]=solution_1.block(from)[it->first];
	  }
   }

  template <int dim>
  void FSIProblem<dim>::transfer_interface_dofs(BlockVector<double> & solution_1, BlockVector<double> & solution_2, unsigned int from, unsigned int to, StructureComponent structure_var_1, StructureComponent structure_var_2)
  {
    std::map<unsigned int, unsigned int> mapping;
    if (from==1) // structure origin
      {
	if (structure_var_1==Displacement || structure_var_1==NotSet)
	  {
	    if (to==0)
	      {
		mapping = n2f;
	      }
	    else if (to==1)
	      {
		if (structure_var_2==Displacement)
		  {
		    mapping = n2a; //  not the correct mapping, just a place holder 
		  }
		else if (structure_var_2==Velocity)
		  {
		    mapping = n2v;
		  }
		else
		  {
		    mapping = n2a; // this is a place holder, but makes the assumption that they want to transfer displacements
		    //AssertThrow(false,ExcNotImplemented());// 'transfer_interface_dofs needs to know which component of the structure you wish to transfer to.');
		  }
	      }
	    else // to==2
	      {
		mapping = n2a;
	      }
	  }
	else if (structure_var_1==Velocity)
	  {
	    if (to==0)
	      {
		mapping = v2f;
	      }
	    else if (to==1)
	      {
		if (structure_var_2==Displacement)
		  {
		    mapping = v2n;  
		  }
		else if (structure_var_2==Velocity)
		  {
		    mapping = v2a; //  not the correct mapping, just a place holder
		  }
		else
		  {
		    mapping = v2a; // placeholder and assume that they want velocity -> velocity
		    //AssertThrow(false,ExcNotImplemented()); // 'transfer_interface_dofs needs to know which component of the structure you wish to transfer to.');
		  }
	      }
	    else // to==2
	      {
		mapping = v2a;
	      }
	  }
	// NotSet behaves like choosing Displacement
	// else // structure_var_1==NotSet
	//   {
	//     AssertThrow(false,ExcNotImplemented()); // 'transfer_interface_dofs needs to know which component of the structure you wish to transfer from.');
	//   }
      }
    else if (from==2)
      {
	if (to==0)
	  {
	    mapping = a2f;
	  }
	else if (to==1)
	  {
	    if (!(structure_var_1==Displacement && structure_var_2==Velocity) && !(structure_var_1==Velocity && structure_var_2==Displacement))
	      { // either both are the same and are displacement or velocity, or one is NotSet
		// we must find which one is not the notset and use that
		if (structure_var_1==Displacement || structure_var_2==Displacement)
		  {
		    mapping = a2n;
		  }
		else if (structure_var_1==Velocity || structure_var_2==Velocity)
		  {
		    mapping = a2v;
		  }
		else // both are NotSet
		  {
		    mapping = a2n; // assume they want to send to displacement
		    //AssertThrow(false,ExcNotImplemented()); // 'transfer_interface_dofs needs to know which component of the structure you wish to transfer to.');
		  }
	      }
	    else
	      {
		AssertThrow(false,ExcNotImplemented()); //  'transfer_interface_dofs has been given conflicting information about which component of the structure you wish to transfer to.');
	      }
	  }
	else // to == 2
	  {
	    mapping = a2f; // placeholder since this will get mapped to itself
	  }
      }
    else // fluid origin
      {
	if (to==0)
	  {
	    mapping = f2n; // placeholder since this will get mapped to itself
	  }
	else if (to==1)
	  {
	    if (!(structure_var_1==Displacement && structure_var_2==Velocity) && !(structure_var_1==Velocity && structure_var_2==Displacement))
	      { // either both are the same and are displacement or velocity, or one is NotSet
		// we must find which one is not the notset and use that
		if (structure_var_1==Displacement || structure_var_2==Displacement)
		  {
		    mapping = f2n;
		  }
		else if (structure_var_1==Velocity || structure_var_2==Velocity)
		  {
		    mapping = f2v;
		  }
		else // both are NotSet
		  {
		    mapping = f2n; // Assume they want displacements
		    //AssertThrow(false,ExcNotImplemented()); // 'transfer_interface_dofs needs to know which component of the structure you wish to transfer to.');
		  }
	      }
	    else
	      {
		AssertThrow(false,ExcNotImplemented()); // 'transfer_interface_dofs has been given conflicting information about which component of the structure you wish to transfer to.');
	      }
	  }
	else // to==2
	  {
	    mapping = f2a;
	  }
      }
    if (from!=to)
      {
	for  (std::map<unsigned int, unsigned int>::iterator it=mapping.begin(); it!=mapping.end(); ++it)
	  {
	    solution_2.block(to)[it->second]=solution_1.block(from)[it->first];
	  }
      }
    else
      {
	// if (structure_var_1!=structure_var_2)
	//   {
	//     // ERROR COULD BE HERE! HAVE NOT CHECKED THAT THERE ARE NOT NOTSET's available
	//     solution_2.block(to)[it->second]=solution_1.block(from)[it->first];
	//   }
	// else if ( COULD BE both notset and from to are not equal 1      OR if is 1, at least one not set)
	if ((from==0 || from==2) || (from==1 && structure_var_1==structure_var_2))
	  { 
	    for  (std::map<unsigned int, unsigned int>::iterator it=mapping.begin(); it!=mapping.end(); ++it)
	      {
		solution_2.block(to)[it->first]=solution_1.block(from)[it->first];
	      }
	  }
	else
	  {
	    for  (std::map<unsigned int, unsigned int>::iterator it=mapping.begin(); it!=mapping.end(); ++it)
	      {
		solution_2.block(to)[it->second]=solution_1.block(from)[it->first];
	      }
	  }
      }
  }

  template <int dim>
  void FSIProblem<dim>::assemble_fluid (Mode enum_, bool assemble_matrix)
  {
    SparseMatrix<double> *fluid_matrix;
    Vector<double> *fluid_rhs;
    if (enum_==state)
      {
	fluid_matrix = &system_matrix.block(0,0);
	fluid_rhs = &system_rhs.block(0);
      }
    else if (enum_==adjoint)
      {
	fluid_matrix = &adjoint_matrix.block(0,0);
	fluid_rhs = &adjoint_rhs.block(0);
      }
    else
      {
	fluid_matrix = &linear_matrix.block(0,0);
	fluid_rhs = &linear_rhs.block(0);
      }

    const FEValuesExtractors::Vector velocities (0);
    const FEValuesExtractors::Scalar pressure (dim);

    Vector<double> tmp;
    Vector<double> forcing_terms;

    tmp.reinit (fluid_rhs->size());
    forcing_terms.reinit (fluid_rhs->size());

    if (assemble_matrix)
      {
	*fluid_matrix=0;
      }
    *fluid_rhs=0;

    QGauss<dim>   quadrature_formula(fem_properties.fluid_degree+2);
    QGauss<dim-1> face_quadrature_formula(fem_properties.fluid_degree+2);

    FEValues<dim> fe_values (fluid_fe, quadrature_formula,
			     update_values    |
			     update_quadrature_points  |
			     update_JxW_values |
			     update_gradients);

    FEFaceValues<dim> fe_face_values (fluid_fe, face_quadrature_formula,
				      update_values    | update_normal_vectors |
				      update_quadrature_points  | update_JxW_values);

    const unsigned int   dofs_per_cell   = fluid_fe.dofs_per_cell;
    const unsigned int   n_q_points      = quadrature_formula.size();
    const unsigned int   n_face_q_points = face_quadrature_formula.size();
    FullMatrix<double>   local_matrix (dofs_per_cell, dofs_per_cell);
    Vector<double>       local_rhs (dofs_per_cell);

    std::vector<Vector<double> > old_solution_values(n_q_points, Vector<double>(dim+1));
    std::vector<Vector<double> > old_old_solution_values(n_q_points, Vector<double>(dim+1));
    std::vector<Vector<double> > adjoint_rhs_values(n_face_q_points, Vector<double>(dim+1));
    std::vector<Vector<double> > linear_rhs_values(n_face_q_points, Vector<double>(dim+1));
    std::vector<Vector<double> > u_star_values(n_q_points, Vector<double>(dim+1));

    std::vector<Tensor<2,dim> > grad_u_old (n_q_points);
    std::vector<Tensor<2,dim> > grad_u_star (n_q_points);
    std::vector<Tensor<2,dim> > F (n_q_points);

    std::vector<types::global_dof_index> local_dof_indices (dofs_per_cell);

    if (enum_==state)
      {
	FluidRightHandSide<dim> rhs_function(physical_properties);
	rhs_function.set_time(time);
	VectorTools::create_right_hand_side(fluid_dof_handler,
					    QGauss<dim>(fluid_fe.degree+2),
					    rhs_function,
					    tmp);
	forcing_terms = tmp;
	forcing_terms *= fluid_theta;
	rhs_function.set_time(time - time_step);
	VectorTools::create_right_hand_side(fluid_dof_handler,
					    QGauss<dim>(fluid_fe.degree+2),
					    rhs_function,
					    tmp);
	forcing_terms.add((1 - fluid_theta), tmp);
	*fluid_rhs += forcing_terms;
      }

    FluidStressValues<dim> fluid_stress_values(physical_properties);
    std::vector<Tensor<1,dim> > stress_values (dim+1);
    std::vector<Vector<double> > g_stress_values(n_face_q_points, Vector<double>(dim+1));

    std::vector<Tensor<1,dim> > 		  phi_u (dofs_per_cell);
    std::vector<SymmetricTensor<2,dim> >      symgrad_phi_u (dofs_per_cell);
    std::vector<Tensor<2,dim> > 		  grad_phi_u (dofs_per_cell);
    std::vector<double>                       div_phi_u   (dofs_per_cell);
    std::vector<double>                       phi_p       (dofs_per_cell);

    double length = 0;
    double residual = 0;


    typename DoFHandler<dim>::active_cell_iterator
      cell = fluid_dof_handler.begin_active(),
      endc = fluid_dof_handler.end();
    //if (enum_==state)
    for (; cell!=endc; ++cell)
      {
	fe_values.reinit (cell);
	local_matrix = 0;
	local_rhs = 0;

	if (assemble_matrix)
	  {
	fe_values.get_function_values (old_solution.block(0), old_solution_values);
	fe_values.get_function_values (old_old_solution.block(0), old_old_solution_values);
	fe_values.get_function_values (solution_star.block(0),u_star_values);
	fe_values[velocities].get_function_gradients(old_solution.block(0),grad_u_old);
	fe_values[velocities].get_function_gradients(solution_star.block(0),grad_u_star);

	for (unsigned int q=0; q<n_q_points; ++q)
	  {
	    F[q]=0;
	    F[q][0][0]=1;
	    F[q][1][1]=1;
	    double determinantJ = determinant(F[q]);
	    //std::cout << determinantJ << std::endl;
	    Tensor<2,dim> detTimesFinv;
	    detTimesFinv[0][0]=F[q][1][1];
	    detTimesFinv[0][1]=-F[q][0][1];
	    detTimesFinv[1][0]=-F[q][1][0];
	    detTimesFinv[1][1]=F[q][0][0];
	    // This should be computed at this and the previous time step so that we can have a mesh for the center

	    Tensor<1,dim> u_star, u_old, u_old_old;
	    for (unsigned int d=0; d<dim; ++d)
	      {
		u_star[d] = u_star_values[q](d);
		u_old[d] = old_solution_values[q](d);
		u_old_old[d] = old_old_solution_values[q](d);
	      }

	    for (unsigned int k=0; k<dofs_per_cell; ++k)
	      {
		phi_u[k]	   = fe_values[velocities].value (k, q);
		symgrad_phi_u[k] = fe_values[velocities].symmetric_gradient (k, q);
		grad_phi_u[k]    = fe_values[velocities].gradient (k, q);
		div_phi_u[k]     = fe_values[velocities].divergence (k, q);
		phi_p[k]         = fe_values[pressure].value (k, q);
	      }
	    for (unsigned int i=0; i<dofs_per_cell; ++i)
	      {
		for (unsigned int j=0; j<dofs_per_cell; ++j)
		  {
		    double epsilon = 0*1e-10; // only when all Dirichlet b.c.s
		    if (physical_properties.navier_stokes)
		      {
			if (enum_==state)
			  {
			    if (fem_properties.newton)
			      {
				local_matrix(i,j) += pow(fluid_theta,2) * physical_properties.rho_f * 
				  ( 
				   phi_u[j]*(transpose(detTimesFinv)*transpose(grad_u_star[q]))*phi_u[i]
				   + u_star*(transpose(detTimesFinv)*transpose(grad_phi_u[j]))*phi_u[i]
				    ) * fe_values.JxW(q);
			      }
			    else if (fem_properties.richardson)
			      {
				local_matrix(i,j) += pow(fluid_theta,2) * physical_properties.rho_f * 
				  (
				   (4./3*u_old_old-1./3*u_old)*(transpose(detTimesFinv)*transpose(grad_phi_u[j]))*phi_u[i]
				   ) * fe_values.JxW(q);
			      }
			    else
			      {
				local_matrix(i,j) += pow(fluid_theta,2) * physical_properties.rho_f * 
				  (
				   phi_u[j]*(transpose(detTimesFinv)*transpose(grad_u_star[q]))*phi_u[i]
				   ) * fe_values.JxW(q);
			      }
			    local_matrix(i,j) += (1-fluid_theta)*fluid_theta * physical_properties.rho_f * 
			      (
			       phi_u[j]*(transpose(detTimesFinv)*transpose(grad_u_old[q]))*phi_u[i]
			       +u_old*(transpose(detTimesFinv)*transpose(grad_phi_u[j]))*phi_u[i]
			       ) * fe_values.JxW(q);
			  }
			else if (enum_==adjoint) 
			  {
			    local_matrix(i,j) += pow(fluid_theta,2) * (physical_properties.rho_f * (phi_u[i]*(transpose(detTimesFinv)*transpose(grad_u_star[q])))*phi_u[j]
						  + physical_properties.rho_f * (u_star*(transpose(detTimesFinv)*transpose(grad_phi_u[i])))*phi_u[j])* fe_values.JxW(q);
			  }
			else // enum_==linear
			  {
			    local_matrix(i,j) += pow(fluid_theta,2) * (physical_properties.rho_f * (phi_u[j]*(transpose(detTimesFinv)*transpose(grad_u_star[q])))*phi_u[i]
						  + physical_properties.rho_f * (u_star*(transpose(detTimesFinv)*transpose(grad_phi_u[j])))*phi_u[i])* fe_values.JxW(q);
			  }
		      }
		    local_matrix(i,j) += ( physical_properties.rho_f/time_step*phi_u[i]*phi_u[j]
					   + fluid_theta * ( 2*physical_properties.viscosity
							     *0.25*1./determinantJ
							     *scalar_product(grad_phi_u[i]*detTimesFinv+transpose(detTimesFinv)*transpose(grad_phi_u[i]),grad_phi_u[j]*detTimesFinv+transpose(detTimesFinv)*transpose(grad_phi_u[j]))
							     )		   
					   - scalar_product(grad_phi_u[i],transpose(detTimesFinv)) * phi_p[j] // (p,\div v)  momentum
					   - phi_p[i] * scalar_product(grad_phi_u[j],transpose(detTimesFinv)) // (\div u, q) mass
					   + epsilon * phi_p[i] * phi_p[j])
		      * fe_values.JxW(q);
		    //std::cout << physical_properties.rho_f * (phi_u[j]*(transpose(detTimesFinv)*transpose(grad_u_star[q])))*phi_u[i] << std::endl;
		  }
	      }
	    if (enum_==state)
	      for (unsigned int i=0; i<dofs_per_cell; ++i)
		{
		  const double old_p = old_solution_values[q](dim);
		  Tensor<1,dim> old_u;
		  for (unsigned int d=0; d<dim; ++d)
		    old_u[d] = old_solution_values[q](d);
		  const Tensor<1,dim> phi_i_s      = fe_values[velocities].value (i, q);
		  //const Tensor<2,dim> symgrad_phi_i_s = fe_values[velocities].symmetric_gradient (i, q);
		  //const double div_phi_i_s =  fe_values[velocities].divergence (i, q);
		  const Tensor<2,dim> grad_phi_i_s = fe_values[velocities].gradient (i, q);
		  const double div_phi_i_s =  fe_values[velocities].divergence (i, q);
		  if (physical_properties.navier_stokes)
		    {
		      if (fem_properties.newton) 
			{
			  local_rhs(i) += pow(1-fluid_theta,2) * physical_properties.rho_f * (u_star*(transpose(detTimesFinv)*transpose(grad_u_star[q])))*phi_i_s * fe_values.JxW(q);
			}
		      local_rhs(i) += pow(1-fluid_theta,2) * physical_properties.rho_f * (u_old*(transpose(detTimesFinv)*transpose(grad_u_old[q])))*phi_i_s * fe_values.JxW(q);
		    }

		    

		  local_rhs(i) += (physical_properties.rho_f/time_step *phi_i_s*old_u
				   + (1-fluid_theta)
				   * (-2*physical_properties.viscosity
				      *0.25/determinantJ*scalar_product(grad_u_old[q]*detTimesFinv+transpose(grad_u_old[q]*detTimesFinv),grad_phi_i_s*detTimesFinv+transpose(grad_phi_i_s*detTimesFinv))
				      //*(-2*physical_properties.viscosity
				      //*(grad_u_old[q][0][0]*symgrad_phi_i_s[0][0]
				      //+ 0.5*(grad_u_old[q][1][0]+grad_u_old[q][0][1])*(symgrad_phi_i_s[1][0]+symgrad_phi_i_s[0][1])
				      //+ grad_u_old[q][1][1]*symgrad_phi_i_s[1][1]
				      )
				   )
		    * fe_values.JxW(q);
			  
		}
	  }
	  }
	unsigned int total_loops;
	if (enum_==state)
	  {
	    total_loops = 2;
	  }
	else
	  {
	    total_loops = 1;
	  }
	for (unsigned int i=0; i<total_loops; ++i)
	  {
	    double multiplier;
	    Vector<double> *stress_vector;
	    if (i==0)
	      {
		fluid_stress_values.set_time(time);
		multiplier=fluid_theta;
		stress_vector = &stress.block(0);
	      }
	    else
	      {
		fluid_stress_values.set_time(time-time_step);
		multiplier=(1-fluid_theta);
		stress_vector = &old_stress.block(0);
	      }
	    for (unsigned int face_no=0;
		 face_no<GeometryInfo<dim>::faces_per_cell;
		 ++face_no)
	      {
		if (cell->at_boundary(face_no))
		  {
		    if (fluid_boundaries[cell->face(face_no)->boundary_indicator()]==Neumann)
		      {
			if (enum_==state)
			  {
			    fe_face_values.reinit (cell, face_no);

			    for (unsigned int q=0; q<n_face_q_points; ++q)
			      for (unsigned int i=0; i<dofs_per_cell; ++i)
				{
				  fluid_stress_values.vector_gradient(fe_face_values.quadrature_point(q),
								      stress_values);
				  Tensor<2,dim> new_stresses;
				  new_stresses[0][0]=stress_values[0][0];
				  new_stresses[1][0]=stress_values[1][0];
				  new_stresses[1][1]=stress_values[1][1];
				  new_stresses[0][1]=stress_values[0][1];
				  local_rhs(i) += multiplier*(fe_face_values[velocities].value (i, q)*
							      new_stresses*fe_face_values.normal_vector(q) *
							      fe_face_values.JxW(q));
				  if (physical_properties.stability_terms)
				    {
				      local_rhs(i) += multiplier*(fe_face_values[velocities].value (i, q)*
								  new_stresses*fe_face_values.normal_vector(q) *
								  fe_face_values.JxW(q));
				    }
				}
			  }
		      }
		    else if (fluid_boundaries[cell->face(face_no)->boundary_indicator()]==Interface)
		      {
			if (enum_==state)
			  {
			    fe_face_values.reinit (cell, face_no);
			    fe_face_values.get_function_values (*stress_vector, g_stress_values);

			    for (unsigned int q=0; q<n_face_q_points; ++q)
			      {
				Tensor<1,dim> g_stress;
				for (unsigned int d=0; d<dim; ++d)
				  g_stress[d] = g_stress_values[q](d);
				for (unsigned int i=0; i<dofs_per_cell; ++i)
				  {
				    local_rhs(i) += multiplier*(fe_face_values[velocities].value (i, q)*
								g_stress * fe_face_values.JxW(q));
				  }
			      }
			  }
			else if (enum_==adjoint)
			  {
			    fe_face_values.reinit (cell, face_no);
			    fe_face_values.get_function_values (rhs_for_adjoint.block(0), adjoint_rhs_values);

			    for (unsigned int q=0; q<n_face_q_points; ++q)
			      {
				Tensor<1,dim> r;
				for (unsigned int d=0; d<dim; ++d)
				  r[d] = adjoint_rhs_values[q](d);
				for (unsigned int i=0; i<dofs_per_cell; ++i)
				  {
				    local_rhs(i) += fluid_theta*(fe_face_values[velocities].value (i, q)*
								 r * fe_face_values.JxW(q));
				  }
				length += fe_face_values.JxW(q);
				residual += 0.5 * r * r * fe_face_values.JxW(q);
			      }
			  }
			else // enum_==linear
			  {
			    fe_face_values.reinit (cell, face_no);
			    fe_face_values.get_function_values (rhs_for_linear.block(0), linear_rhs_values);

			    for (unsigned int q=0; q<n_face_q_points; ++q)
			      {
				Tensor<1,dim> h;
				for (unsigned int d=0; d<dim; ++d)
				  h[d] = linear_rhs_values[q](d);
				for (unsigned int i=0; i<dofs_per_cell; ++i)
				  {
				    local_rhs(i) += fluid_theta*(fe_face_values[velocities].value (i, q)*
								 h * fe_face_values.JxW(q));
				  }
			      }
			  }
		      }
		  }
	      }
	  }
	cell->get_dof_indices (local_dof_indices);
	if (assemble_matrix)
	  {
	    fluid_constraints.distribute_local_to_global (local_matrix, local_rhs,
							  local_dof_indices,
							  *fluid_matrix, *fluid_rhs);
	  }
	else
	  {
	    fluid_constraints.distribute_local_to_global (local_rhs,
							  local_dof_indices,
							  *fluid_rhs);
	  }
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

    const unsigned int   dofs_per_cell = structure_fe.dofs_per_cell;
    const unsigned int   n_q_points    = quadrature_formula.size();
    const unsigned int   n_face_q_points = face_quadrature_formula.size();
    FullMatrix<double>   local_matrix (dofs_per_cell, dofs_per_cell);
    Vector<double>       local_rhs (dofs_per_cell);
    std::vector<types::global_dof_index> local_dof_indices (dofs_per_cell);

    std::vector<Vector<double> > old_solution_values(n_q_points, Vector<double>(2*dim));
    std::vector<Vector<double> > adjoint_rhs_values(n_face_q_points, Vector<double>(2*dim));
    std::vector<Vector<double> > linear_rhs_values(n_face_q_points, Vector<double>(2*dim));
    std::vector<Tensor<2,dim> > grad_n (n_q_points);

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

    StructureStressValues<dim> structure_stress_values(physical_properties);
    std::vector<Tensor<1,dim> > stress_values (3);
    std::vector<Vector<double> > g_stress_values(n_face_q_points, Vector<double>(2*dim));

    std::vector<Tensor<1,dim> > 		phi_n (dofs_per_cell);
    std::vector<SymmetricTensor<2,dim> > 	symgrad_phi_n (dofs_per_cell);
    std::vector<double>                  	div_phi_n   (dofs_per_cell);
    std::vector<Tensor<1,dim> >           	phi_v       (dofs_per_cell);

    const FEValuesExtractors::Vector displacements (0);
    const FEValuesExtractors::Vector velocities (dim);
    typename DoFHandler<dim>::active_cell_iterator cell = structure_dof_handler.begin_active(),
      endc = structure_dof_handler.end();
    for (; cell!=endc; ++cell)
      {
	fe_values.reinit (cell);
	local_matrix = 0;
	local_rhs = 0;
	if (assemble_matrix)
	  {
	fe_values.get_function_values (old_solution.block(1), old_solution_values);
	fe_values[displacements].get_function_gradients(old_solution.block(1),grad_n);
	for (unsigned int q_point=0; q_point<n_q_points;
	     ++q_point)
	  {
	    for (unsigned int k=0; k<dofs_per_cell; ++k)
	      {
		phi_n[k]		   = fe_values[displacements].value (k, q_point);
		symgrad_phi_n[k] = fe_values[displacements].symmetric_gradient (k, q_point);
		div_phi_n[k]     = fe_values[displacements].divergence (k, q_point);
		phi_v[k]         = fe_values[velocities].value (k, q_point);
	      }
	    for (unsigned int i=0; i<dofs_per_cell; ++i)
	      {
		const unsigned int
		  component_i = structure_fe.system_to_component_index(i).first;
		for (unsigned int j=0; j<dofs_per_cell; ++j)
		  {
		    const unsigned int
		      component_j = structure_fe.system_to_component_index(j).first;

		    if (enum_==state || enum_==linear)
		      {

			if (component_i<dim)
			  {
			    if (component_j<dim)
			      {
				local_matrix(i,j)+=(.5*	( 2*physical_properties.mu*symgrad_phi_n[i] * symgrad_phi_n[j]
							  + physical_properties.lambda*div_phi_n[i] * div_phi_n[j]))
				  *fe_values.JxW(q_point);
			      }
			    else
			      {
				local_matrix(i,j)+=physical_properties.rho_s/time_step*phi_n[i]*phi_v[j]*fe_values.JxW(q_point);
			      }
			  }
			else
			  {
			    if (component_j<dim)
			      {
				local_matrix(i,j)+=(-1./time_step*phi_v[i]*phi_n[j])
				  *fe_values.JxW(q_point);
			      }
			    else
			      {
				local_matrix(i,j)+=(0.5*phi_v[i]*phi_v[j])
				  *fe_values.JxW(q_point);
			      }
			  }
		      }
		    else // enum_==adjoint
		      {
			if (component_i<dim)
			  {
			    if (component_j<dim)
			      {
				local_matrix(i,j)+=(.5*	( 2*physical_properties.mu*symgrad_phi_n[i] * symgrad_phi_n[j]
							  + physical_properties.lambda*div_phi_n[i] * div_phi_n[j]))
				  *fe_values.JxW(q_point);
			      }
			    else
			      {
				local_matrix(i,j)+=-1./time_step*phi_n[i]*phi_v[j]*fe_values.JxW(q_point);
			      }
			  }
			else
			  {
			    if (component_j<dim)
			      {
				local_matrix(i,j)+=physical_properties.rho_s/time_step*phi_v[i]*phi_n[j]*fe_values.JxW(q_point);
			      }
			    else
			      {
				local_matrix(i,j)+=(0.5*phi_v[i]*phi_v[j])
				  *fe_values.JxW(q_point);
			      }
			  }
		      }
		  }
	      }
	    if (enum_==state)
	      {
		for (unsigned int i=0; i<dofs_per_cell; ++i)
		  {
		    const unsigned int component_i = structure_fe.system_to_component_index(i).first;
		    Tensor<1,dim> old_n;
		    Tensor<1,dim> old_v;
		    for (unsigned int d=0; d<dim; ++d)
		      old_n[d] = old_solution_values[q_point](d);
		    for (unsigned int d=0; d<dim; ++d)
		      old_v[d] = old_solution_values[q_point](d+dim);
		    const Tensor<1,dim> phi_i_eta      	= fe_values[displacements].value (i, q_point);
		    const Tensor<2,dim> symgrad_phi_i_eta 	= fe_values[displacements].symmetric_gradient (i, q_point);
		    const double div_phi_i_eta 			= fe_values[displacements].divergence (i, q_point);
		    const Tensor<1,dim> phi_i_eta_dot  	= fe_values[velocities].value (i, q_point);
		    if (component_i<dim)
		      {
			local_rhs(i) += (physical_properties.rho_s/time_step *phi_i_eta*old_v
					 +0.5*(-2*physical_properties.mu*(scalar_product(grad_n[q_point],symgrad_phi_i_eta))
					       -physical_properties.lambda*((grad_n[q_point][0][0]+grad_n[q_point][1][1])*div_phi_i_eta))
					 )
			  * fe_values.JxW(q_point);
		      }
		    else
		      {
			local_rhs(i) += (-0.5*phi_i_eta_dot*old_v
					 -1./time_step*phi_i_eta_dot*old_n
					 )
			  * fe_values.JxW(q_point);
		      }
		  }
	      }
	  }
	  }
	unsigned int total_loops;
	if (enum_==state)
	  {
	    total_loops = 2;
	  }
	else
	  {
	    total_loops = 1;
	  }
	for (unsigned int i=0; i<total_loops; ++i)
	  {
	    double multiplier;
	    Vector<double> *stress_vector;
	    if (i==0)
	      {
		structure_stress_values.set_time(time);
		multiplier=structure_theta;
		stress_vector=&stress.block(1);
	      }
	    else
	      {
		structure_stress_values.set_time(time-time_step);
		multiplier=(1-structure_theta);
		stress_vector=&old_stress.block(1);
	      }

	    for (unsigned int face_no=0;
		 face_no<GeometryInfo<dim>::faces_per_cell;
		 ++face_no)
	      {
		if (cell->at_boundary(face_no))
		  {
		    if (structure_boundaries[cell->face(face_no)->boundary_indicator()]==Neumann)
		      {
			if (enum_==state)
			  {
			    fe_face_values.reinit (cell, face_no);
			    // GET SIDE ID!

			    for (unsigned int q=0; q<n_face_q_points; ++q)
			      for (unsigned int i=0; i<dofs_per_cell; ++i)
				{
				  structure_stress_values.vector_gradient(fe_face_values.quadrature_point(q),
									  stress_values);
				  Tensor<2,dim> new_stresses;
				  new_stresses[0][0]=stress_values[0][0];
				  new_stresses[1][0]=stress_values[1][0];
				  new_stresses[1][1]=stress_values[1][1];
				  new_stresses[0][1]=stress_values[0][1];
				  local_rhs(i) += multiplier*(fe_face_values[displacements].value (i, q)*
							      new_stresses*fe_face_values.normal_vector(q) *
							      fe_face_values.JxW(q));
				}
			  }
		      }
		    else if (structure_boundaries[cell->face(face_no)->boundary_indicator()]==Interface)
		      {
			if (enum_==state)
			  {
			    fe_face_values.reinit (cell, face_no);
			    fe_face_values.get_function_values (*stress_vector, g_stress_values);

			    for (unsigned int q=0; q<n_face_q_points; ++q)
			      {
				Tensor<1,dim> g_stress;
				for (unsigned int d=0; d<dim; ++d)
				  g_stress[d] = g_stress_values[q](d);
				for (unsigned int i=0; i<dofs_per_cell; ++i)
				  {
				    local_rhs(i) += multiplier*(fe_face_values[displacements].value (i, q)*
								(-g_stress) * fe_face_values.JxW(q));
				  }
			      }
			  }
			else if (enum_==adjoint)
			  {
			    fe_face_values.reinit (cell, face_no);
			    fe_face_values.get_function_values (rhs_for_adjoint.block(1), adjoint_rhs_values);
			    for (unsigned int q=0; q<n_face_q_points; ++q)
			      {
				Tensor<1,dim> r;
				if (fem_properties.adjoint_type==1)
				  {
				    for (unsigned int d=0; d<dim; ++d)
				      r[d] = adjoint_rhs_values[q](d);
				    for (unsigned int i=0; i<dofs_per_cell; ++i)
				      {
					local_rhs(i) += structure_theta*(fe_face_values[displacements].value (i, q)*
									 r * fe_face_values.JxW(q));
				      }
				  }
				else
				  {
				    if (fem_properties.optimization_method.compare("Gradient")==0)
				      {
					for (unsigned int d=0; d<dim; ++d)
					  r[d] = adjoint_rhs_values[q](d+dim);
					for (unsigned int i=0; i<dofs_per_cell; ++i)
					  {
					    local_rhs(i) += structure_theta*(fe_face_values[velocities].value (i, q)*
									     r * fe_face_values.JxW(q));
					  }
				      }
				    else
				      {
					for (unsigned int d=0; d<dim; ++d)
					  r[d] = adjoint_rhs_values[q](d+dim);
					for (unsigned int i=0; i<dofs_per_cell; ++i)
					  {
					    local_rhs(i) += structure_theta*(fe_face_values[velocities].value (i, q)*
									     r * fe_face_values.JxW(q));
					  }
				      }
				  }

			      }
			  }
			else // enum_==linear
			  {
			    fe_face_values.reinit (cell, face_no);
			    fe_face_values.get_function_values (rhs_for_linear.block(1), linear_rhs_values);
			    for (unsigned int q=0; q<n_face_q_points; ++q)
			      {
				Tensor<1,dim> h;
				for (unsigned int d=0; d<dim; ++d)
				  h[d] = linear_rhs_values[q](d);
				for (unsigned int i=0; i<dofs_per_cell; ++i)
				  {
				    local_rhs(i) += structure_theta*(fe_face_values[displacements].value (i, q)*
								     h * fe_face_values.JxW(q));
				  }
			      }
			  }
		      }
		  }
	      }
	  }
	cell->get_dof_indices (local_dof_indices);
	if (assemble_matrix)
	  {
	    structure_constraints.distribute_local_to_global (local_matrix, local_rhs,
							      local_dof_indices,
							      *structure_matrix, *structure_rhs);
	  }
	else
	  {
	    structure_constraints.distribute_local_to_global (local_rhs,
							      local_dof_indices,
							      *structure_rhs);
	  }
      }
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
		    VectorTools::interpolate_boundary_values (fluid_dof_handler,
							      i,
							      fluid_boundary_values_function,
							      fluid_boundary_values,
							      fluid_fe.component_mask(velocities));
		  }
	      }
	    MatrixTools::apply_boundary_values (fluid_boundary_values,
						system_matrix.block(0,0),
						solution.block(0),
						system_rhs.block(0));
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
							      structure_boundary_values,
							      structure_fe.component_mask(displacements));
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
	    for (unsigned int i=0; i<dofs_per_big_block[2]; ++i)
	      {
		if (a2n.count(i))
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
		if (i==1||i==3) fluid_boundaries.insert(std::pair<unsigned int, BoundaryCondition>(i,Neumann));
		else if (i==2) fluid_boundaries.insert(std::pair<unsigned int, BoundaryCondition>(i,Interface));
		else fluid_boundaries.insert(std::pair<unsigned int, BoundaryCondition>(i,Dirichlet));
	}

	for (unsigned int i=0; i<4; ++i)
	{
		if (i==0) structure_boundaries.insert(std::pair<unsigned int, BoundaryCondition>(i,Interface));
		else if (i==1||i==3) structure_boundaries.insert(std::pair<unsigned int, BoundaryCondition>(i,Neumann));
		else structure_boundaries.insert(std::pair<unsigned int, BoundaryCondition>(i,Dirichlet));
	}
	for (unsigned int i=0; i<4; ++i)
	{
		if (i==2) ale_boundaries.insert(std::pair<unsigned int, BoundaryCondition>(i,Interface));
		else if (i==0) ale_boundaries.insert(std::pair<unsigned int, BoundaryCondition>(i,Dirichlet));
		else ale_boundaries.insert(std::pair<unsigned int, BoundaryCondition>(i,Neumann));
	}

	// we need to track cells, faces, and temporarily the centers for the faces
	// also, we will initially have a temp_* vectors that we will rearrange to match the order of the fluid
	std::vector<Point<dim> > fluid_face_centers, temp_structure_face_centers(structure_triangulation.n_active_cells());
	std::vector<bool> temp_structure_interface_cells(structure_triangulation.n_active_cells());
	std::vector<unsigned int> temp_structure_interface_faces(structure_triangulation.n_active_cells());
	std::vector<bool> quadrature_orientation; // 1 means q increases on fluid means q increases on the structure, -1 if the opposite

	unsigned int ind=0;
    for (typename Triangulation<dim>::active_cell_iterator
         cell = fluid_triangulation.begin_active();
         cell != fluid_triangulation.end(); ++cell)
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
            	fluid_interface_cells.push_back(ind);
            	fluid_interface_faces.push_back(f);
            	fluid_face_centers.push_back(cell->face(f)->center());
              }
          }
        ++ind;
    }

	structure_interface_cells.resize(fluid_interface_cells.size());
	structure_interface_faces.resize(fluid_interface_cells.size());
	std::vector<Point<dim> > structure_face_centers(fluid_interface_faces.size());
    ind=0;
    for (typename Triangulation<dim>::active_cell_iterator
         cell = structure_triangulation.begin_active();
         cell != structure_triangulation.end(); ++cell)
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
              	temp_structure_interface_cells[ind]=true;
              	temp_structure_interface_faces[ind]=f;
              	temp_structure_face_centers[ind]=cell->face(f)->center();
              }
          }
        ++ind;
    }

    // find the matching cells and edges between the two subproblems
    for (unsigned int i=0; i < fluid_interface_cells.size(); ++i)
    {
    	unsigned int j=0;
        for (typename Triangulation<dim>::active_cell_iterator
             cell = structure_triangulation.begin_active();
             cell != structure_triangulation.end(); ++cell)
        {
        	if (temp_structure_interface_cells[j] && fluid_face_centers[i].distance(temp_structure_face_centers[j])<1e-13)
        	{
        		structure_interface_cells[i]=j;
        		structure_interface_faces[i]=temp_structure_interface_faces[j];
        		structure_face_centers[i]=temp_structure_face_centers[j];
        	}
        	++j;
        }
    }

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

    BlockCompressedSimpleSparsityPattern csp (n_big_blocks,n_big_blocks);
    dofs_per_big_block.push_back(dofs_per_block[0]+dofs_per_block[1]);
    dofs_per_big_block.push_back(dofs_per_block[2]+dofs_per_block[3]);
    dofs_per_big_block.push_back(dofs_per_block[4]);


    for (unsigned int i=0; i<n_big_blocks; ++i)
    	for (unsigned int j=0; j<n_big_blocks; ++j)
    		csp.block(i,j).reinit (dofs_per_big_block[i], dofs_per_big_block[j]);

    csp.collect_sizes();

	DoFTools::make_sparsity_pattern (fluid_dof_handler, csp.block(0,0), fluid_constraints, false);
	DoFTools::make_sparsity_pattern (structure_dof_handler, csp.block(1,1), structure_constraints, false);
	DoFTools::make_sparsity_pattern (ale_dof_handler, csp.block(2,2), ale_constraints, false);

    sparsity_pattern.copy_from (csp);

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
    for (unsigned int i=0; i<n_big_blocks; ++i)
    {
    	solution.block(i).reinit (dofs_per_big_block[i]);
    	solution_star.block(i).reinit (dofs_per_big_block[i]);
    	rhs_for_adjoint.block(i).reinit(dofs_per_big_block[i]);
    	rhs_for_adjoint_s.block(i).reinit(dofs_per_big_block[i]);
    	rhs_for_linear.block(i).reinit(dofs_per_big_block[i]);
    	rhs_for_linear_h.block(i).reinit(dofs_per_big_block[i]);
    	rhs_for_linear_p.block(i).reinit(dofs_per_big_block[i]);
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
    }
	solution.collect_sizes ();
	solution_star.collect_sizes ();
	rhs_for_adjoint.collect_sizes ();
	rhs_for_adjoint_s.collect_sizes ();
	rhs_for_linear.collect_sizes ();
	rhs_for_linear_h.collect_sizes ();
	rhs_for_linear_p.collect_sizes ();
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

    fluid_constraints.close ();
    structure_constraints.close ();
    ale_constraints.close ();
  }



  template <int dim>
  void FSIProblem<dim>::solve (const SparseDirectUMFPACK& direct_solver, const int block_num, Mode enum_)
  {
    BlockVector<double> *solution_vector;
    BlockVector<double> *rhs_vector;

    if (enum_==state)
      {
	solution_vector=&solution;
	rhs_vector=&system_rhs;
	direct_solver.vmult (solution_vector->block(block_num), rhs_vector->block(block_num));
      }
    else if (enum_==adjoint)
      {
	solution_vector=&adjoint_solution;
	rhs_vector=&adjoint_rhs;
	direct_solver.solve(rhs_vector->block(block_num));
	solution_vector->block(block_num) = rhs_vector->block(block_num);
      }
    else // enum_==linear
      {
	solution_vector=&linear_solution;
	rhs_vector=&linear_rhs;
	direct_solver.solve(rhs_vector->block(block_num));
	solution_vector->block(block_num) = rhs_vector->block(block_num);
      }

    switch (block_num)
      {
      case 0:
	fluid_constraints.distribute (solution_vector->block(block_num));
	break;
      case 1:
	structure_constraints.distribute (solution_vector->block(block_num));
	break;
      case 2:
	ale_constraints.distribute (solution_vector->block(block_num));
	break;
      default:
	AssertThrow(false,ExcNotImplemented());
      }
  }


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
	                                     Utilities::int_to_string (timestep_number, 3) +
	                                     ".vtk";
	    const std::string structure_filename = "structure-" +
	                                     Utilities::int_to_string (timestep_number, 3) +
	                                     ".vtk";
	    std::ofstream fluid_output (fluid_filename.c_str());
	    std::ofstream structure_output (structure_filename.c_str());
	    fluid_data_out.write_vtk (fluid_output);
	    structure_data_out.write_vtk (structure_output);
  }

  // compute H1 error at all times and L2 error at end time T
  template <int dim>
  void FSIProblem<dim>::compute_error ()
  {
    if (std::fabs(time-fem_properties.T)<1e-13)
      {
	Vector<double> fluid_cellwise_errors (fluid_triangulation.n_active_cells());
	Vector<double> structure_cellwise_errors (structure_triangulation.n_active_cells());
	QTrapez<1>     q_trapez;
	QIterated<dim> quadrature (q_trapez, 3);
	FluidBoundaryValues<dim> fluid_exact_solution(physical_properties);
	StructureBoundaryValues<dim> structure_exact_solution(physical_properties);
	fluid_exact_solution.set_time(time);
	structure_exact_solution.set_time(time);


	std::pair<unsigned int,unsigned int> fluid_indices(0,dim);
	ComponentSelectFunction<dim> fluid_velocity_mask(fluid_indices,dim+1);
	ComponentSelectFunction<dim> fluid_pressure_mask(dim,dim+1);


	VectorTools::integrate_difference (fluid_dof_handler, solution.block(0), fluid_exact_solution,
					   fluid_cellwise_errors, quadrature,
					   VectorTools::L2_norm,&fluid_velocity_mask);
	errors.fluid_velocity_L2_Error=std::max(errors.fluid_velocity_L2_Error,fluid_cellwise_errors.l2_norm());
	fluid_cellwise_errors=0;
	VectorTools::integrate_difference (fluid_dof_handler, solution.block(0), fluid_exact_solution,
					   fluid_cellwise_errors, quadrature, VectorTools::H1_norm,&fluid_velocity_mask);

    
	fluid_exact_solution.set_time(time-(1-fem_properties.fluid_theta)*time_step);
	errors.fluid_velocity_H1_Error += fluid_cellwise_errors.l2_norm();
	VectorTools::integrate_difference (fluid_dof_handler, solution.block(0), fluid_exact_solution,
					   fluid_cellwise_errors, quadrature,
					   VectorTools::L2_norm,&fluid_pressure_mask);
	errors.fluid_pressure_L2_Error=std::max(errors.fluid_pressure_L2_Error,fluid_cellwise_errors.l2_norm());
	fluid_exact_solution.set_time(time);

	std::pair<unsigned int,unsigned int> structure_displacement_indices(0,dim);
	std::pair<unsigned int,unsigned int> structure_velocity_indices(dim,2*dim);
	ComponentSelectFunction<dim> structure_displacement_mask(structure_displacement_indices,2*dim);
	ComponentSelectFunction<dim> structure_velocity_mask(structure_velocity_indices,2*dim);
	VectorTools::integrate_difference (structure_dof_handler, solution.block(1), structure_exact_solution,
					   structure_cellwise_errors, quadrature,
					   VectorTools::L2_norm,&structure_displacement_mask);
	errors.structure_displacement_L2_Error=std::max(errors.structure_displacement_L2_Error,structure_cellwise_errors.l2_norm());
	VectorTools::integrate_difference (structure_dof_handler, solution.block(1), structure_exact_solution,
					   structure_cellwise_errors, quadrature, VectorTools::H1_norm,&structure_displacement_mask);

	errors.structure_displacement_H1_Error += structure_cellwise_errors.l2_norm();

	VectorTools::integrate_difference (structure_dof_handler, solution.block(1), structure_exact_solution,
					   structure_cellwise_errors, quadrature,
					   VectorTools::L2_norm,&structure_velocity_mask);
	errors.structure_velocity_L2_Error += structure_cellwise_errors.l2_norm();
      }
    if (std::fabs(time-fem_properties.T)<1e-13)
      {
	AssertThrow(errors.fluid_velocity_L2_Error>0 && errors.fluid_velocity_H1_Error>0 && errors.fluid_pressure_L2_Error>0
		    && errors.structure_displacement_L2_Error>0 && errors.structure_displacement_H1_Error>0 && errors.structure_velocity_L2_Error>0,ExcIO());
	// errors.fluid_velocity_H1_Error *= time_step;
	// errors.structure_displacement_H1_Error *= time_step;
	// errors.structure_velocity_L2_Error *= time_step;

	std::cout << "dt = " << time_step
		  << " h_f = " << fluid_triangulation.begin_active()->diameter() << " h_s = " << structure_triangulation.begin_active()->diameter()
		  << " L2(T) error [fluid] = " << errors.fluid_velocity_L2_Error << ", "<< " L2(T) error [structure] = " << errors.structure_displacement_L2_Error << std::endl
		  << " L2(0,T;H1(t)) error [fluid] = " << errors.fluid_velocity_H1_Error << ", "
		  << " Pressure error [fluid] = " << errors.fluid_pressure_L2_Error << ", "
		  << " L2(0,T;H1(t)) errors [structure] = " << errors.structure_displacement_H1_Error << std::endl;
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

  template <int dim>
  void FSIProblem<dim>::run ()
  {
    setup_system();
    build_dof_mapping();

    StructureBoundaryValues<dim> structure_boundary_values(physical_properties);
    FluidBoundaryValues<dim> fluid_boundary_values(physical_properties);

    StructureStressValues<dim> structure_boundary_stress(physical_properties);
    FluidStressValues<dim> fluid_boundary_stress(physical_properties);

    structure_boundary_values.set_time(fem_properties.t0-time_step);
    fluid_boundary_values.set_time(fem_properties.t0-time_step);

    VectorTools::project (fluid_dof_handler, fluid_constraints, QGauss<dim>(fem_properties.fluid_degree+2),
                          fluid_boundary_values,
                          old_old_solution.block(0));

    structure_boundary_values.set_time(fem_properties.t0);
    fluid_boundary_values.set_time(fem_properties.t0);

    structure_boundary_stress.set_time(fem_properties.t0);
    fluid_boundary_stress.set_time(fem_properties.t0);

    VectorTools::project (fluid_dof_handler, fluid_constraints, QGauss<dim>(fem_properties.fluid_degree+2),
                          fluid_boundary_values,
                          old_solution.block(0));
    VectorTools::project (structure_dof_handler, structure_constraints, QGauss<dim>(fem_properties.structure_degree+2),
                          structure_boundary_values,
                          old_solution.block(1));

    VectorTools::project(fluid_dof_handler, fluid_constraints, QGauss<dim>(fem_properties.fluid_degree+2),
                          fluid_boundary_stress,
                          old_stress.block(0));

    transfer_interface_dofs(old_stress,old_stress,0,1);
    stress=old_stress;
    double total_time = 0;


	   
    // direct_solver.initialize (system_matrix.block(block_num,block_num));

    for (timestep_number=1, time=fem_properties.t0+time_step;
         time<fem_properties.T;++timestep_number)
      {
	boost::timer t;
    	time = fem_properties.t0 + timestep_number*time_step;
	std::cout << std::endl << "----------------------------------------" << std::endl;
        std::cout << "Time step " << timestep_number
                  << " at t=" << time
                  << std::endl;

        double velocity_jump = 1;
        double velocity_jump_old = 2;
        //unsigned int imprecord=0;
        //unsigned int relrecord=0;
        //unsigned int total_relrecord=0;

        unsigned int count = 0;
        rhs_for_adjoint=1;

	double alpha = fem_properties.steepest_descent_alpha;
	unsigned int imprecord = 0;
	unsigned int relrecord = 0;
	unsigned int consecutiverelrecord = 0;

	//stress=old_stress;

	unsigned int total_solves = 0;

        while (true)
        {
	  ++count;
	  if (count == 1 && fem_properties.true_control)
	    {
	      fluid_boundary_stress.set_time(time);
	      VectorTools::project(fluid_dof_handler, fluid_constraints, QGauss<dim>(fem_properties.fluid_degree+2),
				   fluid_boundary_stress,
				   stress.block(0));
	      transfer_interface_dofs(stress,stress,0,1);
	    }

	  // RHS and Neumann conditions are inside these functions
	  // Solve for the state variables
	  assemble_structure(state, true);
	  assemble_ale(state, true);
	  // This solving order will need changed later since the Dirichlet bcs for the ALE depend on the solution to the structure problem
	  

	  for (unsigned int i=1; i<3; ++i)
	    {
	      dirichlet_boundaries((System)i,state);
	      if (timestep_number==1)
	      	{
	      	  state_solver[i].initialize(system_matrix.block(i,i));
	      	}
	      // solver uses vmult which doesn't require factorization
	      solve(state_solver[i],i,state);
	    }

	  solution_star=1;
	  // FUTURE!!:: CHECK that the dofs on the edge make sense (is some points accidently being stored as interface)

	  while (solution_star.l2_norm()>1e-8)
	    {
	      solution_star=solution;
	      assemble_fluid(state, true);
	      dirichlet_boundaries((System)0,state);
	      if (timestep_number==1)
	      	{
	      	  state_solver[0].initialize(system_matrix.block(0,0));
	      	}
	      state_solver[0].factorize(system_matrix.block(0,0));
	      solve(state_solver[0],0,state);
	      solution_star-=solution;
	      ++total_solves;
	      if ((fem_properties.richardson && !fem_properties.newton) || !physical_properties.navier_stokes)
	      	{
	      	  break;
	      	}
	      else
	      	{
	      	  std::cout << solution_star.l2_norm() << std::endl;
	      	}
	    }
	  solution_star = solution; 
	  build_adjoint_rhs();

	  velocity_jump_old = velocity_jump;
	  velocity_jump=interface_error();

	  if (count%1==0) std::cout << "Jump Error: " << velocity_jump << std::endl;
	  if (count >= fem_properties.max_optimization_iterations || velocity_jump < pow(time_step,4)) break;

	  if (fem_properties.optimization_method.compare("Gradient")==0)
	    {
	      assemble_structure(adjoint, true);
	      assemble_fluid(adjoint, true);
	      assemble_ale(adjoint, true);
	      // Solve for the adjoint
	      for (unsigned int i=0; i<2; ++i)
		{
		  dirichlet_boundaries((System)i,adjoint);
		  adjoint_solver[i].factorize(adjoint_matrix.block(i,i));
		  solve(adjoint_solver[i], i, adjoint);
		}
	      ++total_solves;


	      if (velocity_jump>velocity_jump_old)
		{
		  ++imprecord;
		  //std::cout << "Bad Move." << std::endl;
		  consecutiverelrecord = 0;
		}
	      else if ((velocity_jump/velocity_jump_old)>=0.995) 
		{
		  ++relrecord;
		  ++consecutiverelrecord;
		  //std::cout << "Rel. Bad Move." << std::endl;
		  //std::cout << consecutiverelrecord << std::endl;
		}
	      else
		{
		  imprecord = 0;
		  relrecord = 0;
		  alpha *= 1.01;
		  //std::cout << "Good Move." << std::endl;
		  consecutiverelrecord = 0;
		}

	      if (relrecord > 1) 
		{
		  alpha *= 1.01;
		  relrecord = 0;
		}
	      else if (imprecord > 0)
		{
		  alpha *= 0.95;
		  imprecord = 0;
		}
	    
	      if (consecutiverelrecord>50)
		{
		  std::cout << "Break!" << std::endl;
		  //break;
		}

	      // Update the stress using the adjoint variables
	      stress.block(0)*=(1-alpha);
	      tmp=0;

	      if (fem_properties.adjoint_type==1)
		{
		  transfer_interface_dofs(adjoint_solution,tmp,1,0,Displacement);
		  tmp.block(0)*=1./time_step;
		}
	      else
		{
		  transfer_interface_dofs(adjoint_solution,tmp,1,0,Velocity);
		}

	      tmp.block(0).add(-fem_properties.fluid_theta,adjoint_solution.block(0));

	      //std::cout << "L2 " << tmp.block(0).l2_norm() << std::endl;

	      // not negated since tmp has reverse of proper negation
	      double multiplier = float(alpha)/fem_properties.penalty_epsilon;
            
	      stress.block(0).add(multiplier,tmp.block(0));

	      //std::cout << "STRESS: " << stress.block(0).l2_norm() << std::endl;

	      tmp=0;
	      transfer_interface_dofs(stress,tmp,0,0);
	      stress=0;
	      transfer_interface_dofs(tmp,stress,0,0);

	      transfer_interface_dofs(stress,stress,0,1,Displacement);
	      //if (count%50==0) std::cout << "alpha: " << alpha << std::endl;
	    }
	  else // fem_properties.optimization_method==CG
	    {
	      // if (fem_properties.adjoint_type==1)
	      // 	{
		  // tmp=fem_properties.cg_tolerance;
		  // //tmp=rhs_for_adjoint;
		  // //tmp*=-1;
		  // // x^0 = guess
		  // // get adjoint variables 
		  // // assemble_structure(adjoint);
		  // // assemble_fluid(adjoint);
		  // // for (unsigned int i=0; i<2; ++i)
		  // // 	{
		  // // 	  dirichlet_boundaries((System)i,adjoint);
		  // // 	  solve(i,adjoint);
		  // // 	}
		  // // ++total_solves;
		  // // tmp=0; tmp2=0;
		  // // rhs_for_linear_p=0;
		  // // transfer_interface_dofs(adjoint_solution,tmp,1,0);
		  // // tmp.block(0)*=-1/time_step;
		  // // transfer_interface_dofs(adjoint_solution,tmp2,0,0);
		  // // tmp.block(0)+=tmp2.block(0);
		  // // tmp.block(0).add(sqrt(fem_properties.penalty_epsilon),rhs_for_adjoint_s.block(0));
		  // // transfer_interface_dofs(tmp,rhs_for_linear_p,0,0);
		  // // transfer_interface_dofs(rhs_for_linear_p,rhs_for_linear_p,0,1);
		  // // rhs_for_linear_p.block(1)*=-1;   // copy, negate
		  // // rhs_for_linear_p*=-1;
		  // // Generate a random vector
		  // //for  (Vector<double>::iterator it=tmp.block(0).begin(); it!=tmp.block(0).end(); ++it)
		  // // *it = ((double)std::rand() / (double)(RAND_MAX)) * fem_properties.cg_tolerance; //std::rand(0,10);
		  // //std::cout << *it << std::endl;

		  // rhs_for_linear_h=0;
		  // transfer_interface_dofs(tmp,rhs_for_linear_h,0,0);
		  // transfer_interface_dofs(rhs_for_linear_h,rhs_for_linear_h,0,1);
		  // rhs_for_linear_h.block(1) *= -1;   // copy, negate

		  // // b = -u + v^n	       
		  // rhs_for_adjoint=0;
		  // transfer_interface_dofs(solution,rhs_for_adjoint,1,0);
		  // tmp=0;
		  // transfer_interface_dofs(solution,tmp,0,0);
		  // rhs_for_adjoint.block(0)-=tmp.block(0);
	     
		  // // MADE IT TO HERE
		  // // Some Notes:
		  // // (g,\xi) exists on displacement vector
		  // // (h,\xi) exists on displacement vector
		  // // (r,\xi) exists on velocity vector for Gradient method (not sure for CG method, prob need to write it out and it is probably on velocity)
		  // // need to make function to choose which component is copied

		  // // get linearized variables
		  // rhs_for_linear = rhs_for_linear_h;
		  // assemble_structure(linear, true);
		  // assemble_fluid(linear, true);
		  // for (unsigned int i=0; i<2; ++i)
		  //   {
		  //     dirichlet_boundaries((System)i,linear);
		  //     linear_solver[i].factorize(linear_matrix.block(i,i));
		  //     solve(linear_solver[i], i, linear);
		  //   }
		  // ++total_solves;
	      
		  // // -Ax = -w^n + phi^n
		  // tmp=0;tmp2=0;
		  // transfer_interface_dofs(linear_solution,tmp,1,0);
		  // transfer_interface_dofs(linear_solution,tmp2,0,0);
		  // tmp.block(0)-=tmp2.block(0);
	      
		  // // fixed to here

		  // // r^0 = b - Ax
		  // rhs_for_adjoint.block(0)+=tmp.block(0);
		  // transfer_interface_dofs(rhs_for_adjoint,rhs_for_adjoint,0,1);
		  // rhs_for_adjoint.block(1)*=-1;   // copy, negate
		  // // r_s^0 = - sqrt(delta)g^n - sqrt(delta)h^n
		  // rhs_for_adjoint_s=0;
		  // transfer_interface_dofs(rhs_for_linear_h,rhs_for_adjoint_s,0,0);
		  // rhs_for_adjoint_s.block(0)+=stress.block(0);
		  // rhs_for_adjoint_s.block(0)*=-sqrt(fem_properties.penalty_epsilon);

		  // // get adjoint variables 
		  // assemble_structure(adjoint, true);
		  // assemble_fluid(adjoint, true);
		  // for (unsigned int i=0; i<2; ++i)
		  //   {
		  //     dirichlet_boundaries((System)i,adjoint);
		  //     adjoint_solver[i].factorize(adjoint_matrix.block(i,i));
		  //     solve(adjoint_solver[i], i, adjoint);
		  //   }
		  // ++total_solves;
	      
		  // //fluid_constraints.distribute(
		  // // apply preconditioner
		  // //std::cout << solution.block(0).size() << " " << system_matrix.block(0,0).m() << std::endl; 
		  // for (unsigned int i=0; i<solution.block(0).size(); ++i)
		  //   adjoint_solution.block(0)[i] *= system_matrix.block(0,0).diag_element(i);
		  // for (unsigned int i=0; i<solution.block(1).size(); ++i)
		  //   adjoint_solution.block(1)[i] *= time_step*system_matrix.block(1,1).diag_element(i);
		  // // tmp=adjoint_solution;
		  // // PreconditionJacobi<SparseMatrix<double> > preconditioner;
		  // // preconditioner.initialize(system_matrix.block(0,0), 0.6);
		  // // preconditioner.step(adjoint_solution.block(0),tmp.block(0));
		  // // preconditioner.initialize(system_matrix.block(1,1), 0.6);
		  // // preconditioner.step(adjoint_solution.block(1),tmp.block(1));
	      
		  // //adjoint_solution*=float(time_step)/(time_step-1);

		  // // p^0 = beta^n - psi^n/dt + sqrt(delta)(-sqrt(delta) g^n -sqrt(delta) h^n)
		  // tmp=0; tmp2=0;
		  // rhs_for_linear_p=0;
		  // transfer_interface_dofs(adjoint_solution,tmp,1,0);
		  // tmp.block(0)*=-1/time_step;
		  // transfer_interface_dofs(adjoint_solution,tmp2,0,0);
		  // tmp.block(0)+=tmp2.block(0);
		  // tmp.block(0).add(sqrt(fem_properties.penalty_epsilon),rhs_for_adjoint_s.block(0));
		  // transfer_interface_dofs(tmp,rhs_for_linear_p,0,0);
		  // transfer_interface_dofs(rhs_for_linear_p,rhs_for_linear_p,0,1);
		  // rhs_for_linear_p.block(1)*=-1;   // copy, negate

		  // //rhs_for_linear_p = rhs_for_adjoint; // erase!! not symmetric
		  // premultiplier.block(0)=rhs_for_adjoint.block(0); // premult
		  // double p_n_norm_square = interface_norm(rhs_for_linear_p.block(0));
		  // //double p_n_norm_square = rhs_for_linear_p.block(0).l2_norm();
		  // //std::cout <<  p_n_norm_square << std::endl;
		  // rhs_for_linear_Ap_s=0;


		  // while (std::abs(p_n_norm_square) > fem_properties.cg_tolerance)
		  //   {
		  //     //std::cout << "more text" << std::endl;
		  //     // get linearized variables
		  //     rhs_for_linear = rhs_for_linear_p;
		  //     assemble_structure(linear, false);
		  //     assemble_fluid(linear, false);
		  //     for (unsigned int i=0; i<2; ++i)
		  // 	{
		  // 	  dirichlet_boundaries((System)i,linear);
		  // 	  solve(linear_solver[i], i, linear);
		  // 	}
		  //     ++total_solves;

		  //     // ||Ap||^2 = (w-phi/dt)^2+delta*h^2
		  //     tmp=0;tmp2=0;
		  //     transfer_interface_dofs(linear_solution,tmp,1,0);
		  //     tmp.block(0)*=-1./time_step;
		  //     transfer_interface_dofs(linear_solution,tmp2,0,0);
		  //     tmp.block(0)+=tmp2.block(0);
		  //     rhs_for_linear_Ap_s.block(0) = rhs_for_linear_p.block(0);
		  //     rhs_for_linear_Ap_s *= sqrt(fem_properties.penalty_epsilon);
		  //     premultiplier.block(0)=rhs_for_linear_p.block(0);
		  //     double ap_norm_square = interface_norm(tmp.block(0));
		  //     //double ap_norm_square = tmp.block(0).l2_norm();
		  //     ap_norm_square += interface_norm(rhs_for_linear_p.block(0));
		  //     //ap_norm_square += rhs_for_linear_p.block(0).l2_norm();
		  //     double sigma = p_n_norm_square/ap_norm_square;

		  //     // h^{n+1} = h^n + sigma * p^n
		  //     rhs_for_linear_h.block(0).add(sigma,rhs_for_linear_p.block(0));
		  //     transfer_interface_dofs(rhs_for_linear_h,rhs_for_linear_h,0,1);
		  //     rhs_for_linear_h.block(1)*=-1;   // copy, negate

		  //     // r^{n+1} = r^n - sigma * Ap
		  //     // Ap still stored in tmp, could make new vector rhs_for_linear_Ap
		  //     rhs_for_adjoint.block(0).add(-sigma, tmp.block(0));
		  //     transfer_interface_dofs(rhs_for_adjoint,rhs_for_adjoint,0,1);
		  //     rhs_for_adjoint.block(1)*=-1;   // copy, negate
		  //     rhs_for_adjoint_s.block(0).add(-sigma, rhs_for_linear_Ap_s.block(0));
		  
		  //     // get adjoint variables (b^{n+1},....)
		  //     assemble_structure(adjoint, false);
		  //     assemble_fluid(adjoint, false);
		  //     for (unsigned int i=0; i<2; ++i)
		  // 	{
		  // 	  dirichlet_boundaries((System)i,adjoint);
		  // 	  solve(adjoint_solver[i], i, adjoint);
		  // 	}
		  //     ++total_solves;

		  //     // apply preconditioner
		  //     // adjoint_solution*=float(time_step)/(time_step-1);
		  //     for (unsigned int i=0; i<solution.block(0).size(); ++i)
		  // 	adjoint_solution.block(0)[i] *= system_matrix.block(0,0).diag_element(i);
		  //     for (unsigned int i=0; i<solution.block(1).size(); ++i)
		  // 	adjoint_solution.block(1)[i] *= time_step*system_matrix.block(1,1).diag_element(i);
		 

		  //     // tmp=adjoint_solution;
		  //     // PreconditionJacobi<SparseMatrix<double> > preconditioner;
		  //     // preconditioner.initialize(system_matrix.block(0,0), 0.6);
		  //     // preconditioner.step(adjoint_solution.block(0),tmp.block(0));
		  //     // preconditioner.initialize(system_matrix.block(1,1), 0.6);
		  //     // preconditioner.step(adjoint_solution.block(1),tmp.block(1));

		  //     // A*r^{n+1} = beta^{n+1} - psi^{n+1}/dt + sqrt(delta)(second part of r)
		  //     tmp=0; tmp2=0;
		  //     transfer_interface_dofs(adjoint_solution,tmp,1,0);
		  //     tmp.block(0)*=-1/time_step;
		  //     transfer_interface_dofs(adjoint_solution,tmp2,0,0);
		  //     tmp.block(0)+=tmp2.block(0);
		  //     tmp.block(0).add(sqrt(fem_properties.penalty_epsilon),rhs_for_adjoint_s.block(0)); // not sure about this one

		  //     //rhs_for_linear_p = rhs_for_adjoint; // erase!! not symmetric
		  //     premultiplier.block(0)=rhs_for_adjoint.block(0);
		  //     double Astar_r_np1_norm_square = interface_norm(tmp.block(0));
		  //     //double Astar_r_np1_norm_square = tmp.block(0).l2_norm();
		  //     double tau = Astar_r_np1_norm_square / p_n_norm_square;

		  //     // p^{n+1} = A*r^{n+1} + tau * p^{n}
		  //     rhs_for_linear_p.block(0) *= tau;
		  //     rhs_for_linear_p.block(0)+=tmp.block(0);
		  //     transfer_interface_dofs(rhs_for_linear_p,rhs_for_linear_p,0,1);
		  //     rhs_for_linear_p.block(1)*=-1;   // copy, negate
		  //     p_n_norm_square = interface_norm(rhs_for_linear_p.block(0));
		  //     //p_n_norm_square = rhs_for_linear_p.block(0).l2_norm();
		  //     //std::cout << p_n_norm_square << std::endl;	 
		  //   }
	    // 	}
	    //   else
	    // 	{
	    	  tmp=fem_properties.cg_tolerance;
	    	  //tmp=rhs_for_adjoint;
	    	  //tmp*=-1;
	    	  // x^0 = guess
	    	  // get adjoint variables 
	    	  // assemble_structure(adjoint);
	    	  // assemble_fluid(adjoint);
	    	  // for (unsigned int i=0; i<2; ++i)
	    	  // 	{
	    	  // 	  dirichlet_boundaries((System)i,adjoint);
	    	  // 	  solve(i,adjoint);
	    	  // 	}
	    	  // ++total_solves;
	    	  // tmp=0; tmp2=0;
	    	  // rhs_for_linear_p=0;
	    	  // transfer_interface_dofs(adjoint_solution,tmp,1,0);
	    	  // tmp.block(0)*=-1/time_step;
	    	  // transfer_interface_dofs(adjoint_solution,tmp2,0,0);
	    	  // tmp.block(0)+=tmp2.block(0);
	    	  // tmp.block(0).add(sqrt(fem_properties.penalty_epsilon),rhs_for_adjoint_s.block(0));
	    	  // transfer_interface_dofs(tmp,rhs_for_linear_p,0,0);
	    	  // transfer_interface_dofs(rhs_for_linear_p,rhs_for_linear_p,0,1);
	    	  // rhs_for_linear_p.block(1)*=-1;   // copy, negate
	    	  // rhs_for_linear_p*=-1;
	    	  // Generate a random vector
	    	  //for  (Vector<double>::iterator it=tmp.block(0).begin(); it!=tmp.block(0).end(); ++it)
	    	  // *it = ((double)std::rand() / (double)(RAND_MAX)) * fem_properties.cg_tolerance; //std::rand(0,10);
	    	  //std::cout << *it << std::endl;

	    	  rhs_for_linear_h=0;
	    	  transfer_interface_dofs(tmp,rhs_for_linear_h,0,0);
		  transfer_interface_dofs(rhs_for_linear_h,rhs_for_linear_h,0,1,Displacement);
		  rhs_for_linear_h.block(1) *= -1;   // copy, negate

		  if (fem_properties.adjoint_type==1)
		    {
		      // b = -u + [n^n-n^n-1]/dt	       
		      tmp=0;
		      rhs_for_adjoint=0;
		      transfer_interface_dofs(solution,rhs_for_adjoint,1,0,Displacement);
		      rhs_for_adjoint.block(0)*=1./time_step;
		      transfer_interface_dofs(old_solution,tmp,1,0,Displacement);
		      rhs_for_adjoint.block(0).add(-1./time_step,tmp.block(0));
		      tmp=0;
		      transfer_interface_dofs(solution,tmp,0,0);
		      rhs_for_adjoint.block(0)-=tmp.block(0);
		    }
		  else
		    {
		      // b = -u + v^	       
		      tmp=0;
		      rhs_for_adjoint=0;
		      transfer_interface_dofs(solution,rhs_for_adjoint,1,0,Velocity);
		      tmp=0;
		      transfer_interface_dofs(solution,tmp,0,0);
		      rhs_for_adjoint.block(0)-=tmp.block(0);
		    }	     

	    	  // get linearized variables
	    	  rhs_for_linear = rhs_for_linear_h;
	    	  assemble_structure(linear, true);
	    	  assemble_fluid(linear, true);
	    	  for (unsigned int i=0; i<2; ++i)
	    	    {
	    	      dirichlet_boundaries((System)i,linear);
	    	      linear_solver[i].factorize(linear_matrix.block(i,i));
	    	      solve(linear_solver[i], i, linear);
	    	    }
	    	  ++total_solves;
	      

		  if (fem_properties.adjoint_type==1)
		    {
		      // -Ax = -w^n + phi^n/dt
		      tmp=0;tmp2=0;
		      transfer_interface_dofs(linear_solution,tmp,1,0,Displacement);
		      tmp.block(0)*=1./time_step;
		      transfer_interface_dofs(linear_solution,tmp2,0,0);
		      tmp.block(0)-=tmp2.block(0);
		    }
		  else
		    {
		      // -Ax = -w^n + phi_dot^n
		      tmp=0;tmp2=0;
		      transfer_interface_dofs(linear_solution,tmp,1,0,Velocity);
		      transfer_interface_dofs(linear_solution,tmp2,0,0);
		      tmp.block(0)-=tmp2.block(0);
		    }
	      
	    	  // r^0 = b - Ax
	    	  rhs_for_adjoint.block(0)+=tmp.block(0);

		  if (fem_properties.adjoint_type==1)
		    {
		      transfer_interface_dofs(rhs_for_adjoint,rhs_for_adjoint,0,1,Displacement);
		    }
		  else
		    {
		      transfer_interface_dofs(rhs_for_adjoint,rhs_for_adjoint,0,1,Velocity);
		    }

	    	  rhs_for_adjoint.block(1)*=-1;   // copy, negate
	    	  // r_s^0 = - sqrt(delta)g^n - sqrt(delta)h^n
	    	  rhs_for_adjoint_s=0;
	    	  transfer_interface_dofs(rhs_for_linear_h,rhs_for_adjoint_s,0,0);
	    	  rhs_for_adjoint_s.block(0)+=stress.block(0);
	    	  rhs_for_adjoint_s.block(0)*=-sqrt(fem_properties.penalty_epsilon);

	    	  // get adjoint variables 
	    	  assemble_structure(adjoint, true);
	    	  assemble_fluid(adjoint, true);
	    	  for (unsigned int i=0; i<2; ++i)
	    	    {
	    	      dirichlet_boundaries((System)i,adjoint);
	    	      adjoint_solver[i].factorize(adjoint_matrix.block(i,i));
	    	      solve(adjoint_solver[i], i, adjoint);
	    	    }
	    	  ++total_solves;
	      
	    	  //fluid_constraints.distribute(
	    	  // apply preconditioner
	    	  //std::cout << solution.block(0).size() << " " << system_matrix.block(0,0).m() << std::endl; 
	    	  // for (unsigned int i=0; i<solution.block(0).size(); ++i)
	    	  //   adjoint_solution.block(0)[i] *= system_matrix.block(0,0).diag_element(i);
	    	  // for (unsigned int i=0; i<solution.block(1).size(); ++i)
	    	  //   adjoint_solution.block(1)[i] *= time_step*system_matrix.block(1,1).diag_element(i);
	    	  // tmp=adjoint_solution;
	    	  // PreconditionJacobi<SparseMatrix<double> > preconditioner;
	    	  // preconditioner.initialize(system_matrix.block(0,0), 0.6);
	    	  // preconditioner.step(adjoint_solution.block(0),tmp.block(0));
	    	  // preconditioner.initialize(system_matrix.block(1,1), 0.6);
	    	  // preconditioner.step(adjoint_solution.block(1),tmp.block(1));
	      
	    	  //adjoint_solution*=float(time_step)/(time_step-1);

	    	  // p^0 = beta^n - psi^n/dt + sqrt(delta)(-sqrt(delta) g^n -sqrt(delta) h^n)
	    	  tmp=0; tmp2=0;
	    	  rhs_for_linear_p=0;

		  if (fem_properties.adjoint_type==1)
		    {
		      transfer_interface_dofs(adjoint_solution,tmp,1,0,Displacement);
		      tmp.block(0)*=-1/time_step;
		    }
		  else
		    {
		      transfer_interface_dofs(adjoint_solution,tmp,1,0,Velocity);
		      tmp.block(0)*=-1;
		    }

	    	  transfer_interface_dofs(adjoint_solution,tmp2,0,0);
	    	  tmp.block(0)+=tmp2.block(0);
	    	  tmp.block(0).add(sqrt(fem_properties.penalty_epsilon),rhs_for_adjoint_s.block(0));
	    	  transfer_interface_dofs(tmp,rhs_for_linear_p,0,0);
	    	  transfer_interface_dofs(rhs_for_linear_p,rhs_for_linear_p,0,1,Displacement);
	    	  rhs_for_linear_p.block(1)*=-1;   // copy, negate

	    	  //rhs_for_linear_p = rhs_for_adjoint; // erase!! not symmetric
	    	  premultiplier.block(0)=rhs_for_adjoint.block(0); // premult
	    	  double p_n_norm_square = interface_norm(rhs_for_linear_p.block(0));
	    	  //double p_n_norm_square = rhs_for_linear_p.block(0).l2_norm();
	    	  //std::cout <<  p_n_norm_square << std::endl;
	    	  rhs_for_linear_Ap_s=0;


	    	  while (std::abs(p_n_norm_square) > fem_properties.cg_tolerance)
	    	    {
	    	      //std::cout << "more text" << std::endl;
	    	      // get linearized variables
	    	      rhs_for_linear = rhs_for_linear_p;
	    	      assemble_structure(linear, false);
	    	      assemble_fluid(linear, false);
	    	      for (unsigned int i=0; i<2; ++i)
	    		{
	    		  dirichlet_boundaries((System)i,linear);
	    		  solve(linear_solver[i], i, linear);
	    		}
	    	      ++total_solves;

	    	      // ||Ap||^2 = (w-phi/dt)^2+delta*h^2
	    	      tmp=0;tmp2=0;
		      if (fem_properties.adjoint_type==1)
			{
			  transfer_interface_dofs(linear_solution,tmp,1,0,Displacement);
			  tmp.block(0)*=-1./time_step;
			}
		      else
			{
			  transfer_interface_dofs(linear_solution,tmp,1,0,Velocity);
			  tmp.block(0)*=-1;
			}
	    	      transfer_interface_dofs(linear_solution,tmp2,0,0);
	    	      tmp.block(0)+=tmp2.block(0);
	    	      rhs_for_linear_Ap_s.block(0) = rhs_for_linear_p.block(0);
	    	      rhs_for_linear_Ap_s *= sqrt(fem_properties.penalty_epsilon);
	    	      premultiplier.block(0)=rhs_for_linear_p.block(0);
	    	      double ap_norm_square = interface_norm(tmp.block(0));
	    	      //double ap_norm_square = tmp.block(0).l2_norm();
	    	      ap_norm_square += interface_norm(rhs_for_linear_p.block(0));
	    	      //ap_norm_square += rhs_for_linear_p.block(0).l2_norm();
	    	      double sigma = p_n_norm_square/ap_norm_square;

	    	      // h^{n+1} = h^n + sigma * p^n
	    	      rhs_for_linear_h.block(0).add(sigma,rhs_for_linear_p.block(0));
	    	      transfer_interface_dofs(rhs_for_linear_h,rhs_for_linear_h,0,1,Displacement);
	    	      rhs_for_linear_h.block(1)*=-1;   // copy, negate

	    	      // r^{n+1} = r^n - sigma * Ap
	    	      // Ap still stored in tmp, could make new vector rhs_for_linear_Ap
	    	      rhs_for_adjoint.block(0).add(-sigma, tmp.block(0));
		      if (fem_properties.adjoint_type==1)
			{
			  transfer_interface_dofs(rhs_for_adjoint,rhs_for_adjoint,0,1,Displacement);
			}
		      else
			{
			  transfer_interface_dofs(rhs_for_adjoint,rhs_for_adjoint,0,1,Velocity);
			}
	    	      rhs_for_adjoint.block(1)*=-1;   // copy, negate
	    	      rhs_for_adjoint_s.block(0).add(-sigma, rhs_for_linear_Ap_s.block(0));
		  
	    	      // get adjoint variables (b^{n+1},....)
	    	      assemble_structure(adjoint, false);
	    	      assemble_fluid(adjoint, false);
	    	      for (unsigned int i=0; i<2; ++i)
	    		{
	    		  dirichlet_boundaries((System)i,adjoint);
	    		  solve(adjoint_solver[i], i, adjoint);
	    		}
	    	      ++total_solves;

	    	      // apply preconditioner
	    	      // adjoint_solution*=float(time_step)/(time_step-1);
	    	      // for (unsigned int i=0; i<solution.block(0).size(); ++i)
	    	      // 	adjoint_solution.block(0)[i] *= system_matrix.block(0,0).diag_element(i);
	    	      // for (unsigned int i=0; i<solution.block(1).size(); ++i)
	    	      // 	adjoint_solution.block(1)[i] *= time_step*system_matrix.block(1,1).diag_element(i);
		 

	    	      // tmp=adjoint_solution;
	    	      // PreconditionJacobi<SparseMatrix<double> > preconditioner;
	    	      // preconditioner.initialize(system_matrix.block(0,0), 0.6);
	    	      // preconditioner.step(adjoint_solution.block(0),tmp.block(0));
	    	      // preconditioner.initialize(system_matrix.block(1,1), 0.6);
	    	      // preconditioner.step(adjoint_solution.block(1),tmp.block(1));

	    	      // A*r^{n+1} = beta^{n+1} - psi^{n+1}/dt + sqrt(delta)(second part of r)
	    	      tmp=0; tmp2=0;
		      if (fem_properties.adjoint_type==1)
			{
			  transfer_interface_dofs(adjoint_solution,tmp,1,0,Displacement);
			  tmp.block(0)*=-1/time_step;
			}
		      else
			{
			  transfer_interface_dofs(adjoint_solution,tmp,1,0,Displacement);
			  tmp.block(0)*=-1;
			}
	    	      transfer_interface_dofs(adjoint_solution,tmp2,0,0);
	    	      tmp.block(0)+=tmp2.block(0);
	    	      tmp.block(0).add(sqrt(fem_properties.penalty_epsilon),rhs_for_adjoint_s.block(0)); // not sure about this one

	    	      //rhs_for_linear_p = rhs_for_adjoint; // erase!! not symmetric
	    	      premultiplier.block(0)=rhs_for_adjoint.block(0);
	    	      double Astar_r_np1_norm_square = interface_norm(tmp.block(0));
	    	      //double Astar_r_np1_norm_square = tmp.block(0).l2_norm();
	    	      double tau = Astar_r_np1_norm_square / p_n_norm_square;

	    	      // p^{n+1} = A*r^{n+1} + tau * p^{n}
	    	      rhs_for_linear_p.block(0) *= tau;
	    	      rhs_for_linear_p.block(0)+=tmp.block(0);
	    	      transfer_interface_dofs(rhs_for_linear_p,rhs_for_linear_p,0,1,Displacement);
	    	      rhs_for_linear_p.block(1)*=-1;   // copy, negate
	    	      p_n_norm_square = interface_norm(rhs_for_linear_p.block(0));
	    	      //p_n_norm_square = rhs_for_linear_p.block(0).l2_norm();
	    	      //std::cout << p_n_norm_square << std::endl;
	    	    }
		  //}
		  
	      // update stress
	      stress.block(0) += rhs_for_linear_h.block(0);
	      tmp=0;
	      transfer_interface_dofs(stress,tmp,0,0);
	      transfer_interface_dofs(stress,tmp,1,1,Displacement);
	      stress=0;
	      transfer_interface_dofs(tmp,stress,0,0);
	      transfer_interface_dofs(tmp,stress,1,1,Displacement);
	      transfer_interface_dofs(stress,stress,0,1,Displacement);
	    }

	}
	std::cout << "Total Solves: " << total_solves << std::endl;
        if (fem_properties.make_plots) output_results ();
	if (fem_properties.richardson) 
	  {
	    old_old_solution = old_solution;
	  }
        old_solution = solution;
        old_stress = stress;
        if (fem_properties.print_error) compute_error();
	std::cout << "Comp. Time: " << t.elapsed() << std::endl;
	total_time += t.elapsed();
	std::cout << "Est. Rem.: " << (fem_properties.T-time)/time_step*total_time/timestep_number << std::endl;
	t.restart();
      }
  }
}




int main (int argc, char *argv[])
{
  const unsigned int dim = 2;

  if (argc != 2)
    {
      std::cerr << "  usage: ./FSIProblem <parameter-file.prm>" << std::endl;
      return -1;
    }
  try
    {
      using namespace dealii;
      using namespace FSI_Project;

      deallog.depth_console (0);

      ParameterHandler prm;
      
      Parameters::declare_parameters<dim>(prm);

      bool success=prm.read_input(argv[1]);
      if (!success)
      {
    	  std::cerr << "Couldn't read filename: " << argv[1] << std::endl;
      }
      FSIProblem<2> fsi_solver(prm);
      fsi_solver.run();
    }
  catch (std::exception &exc)
    {
      std::cerr << std::endl << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Exception on processing: " << std::endl
                << exc.what() << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;

      return 1;
    }
  catch (...)
    {
      std::cerr << std::endl << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Unknown exception!" << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;
      return 1;
    }

  return 0;
}
