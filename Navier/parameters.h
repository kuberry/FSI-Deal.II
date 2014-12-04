#pragma once
#include <deal.II/base/parameter_handler.h>

namespace Parameters
{
  using namespace dealii;

  struct ComputationData
  {
    unsigned int fluid_active_cells;
    unsigned int fluid_velocity_dofs;
    unsigned int fluid_pressure_dofs;
    double fluid_velocity_L2_Error;
    double fluid_velocity_H1_Error;
    double fluid_pressure_L2_Error;

    unsigned int structure_active_cells;
    unsigned int structure_displacement_dofs;
    unsigned int structure_velocity_dofs;
    double structure_displacement_L2_Error;
    double structure_displacement_H1_Error;
    double structure_velocity_L2_Error;
  };
  struct SimulationProperties
  {
    // FE_Q Degrees
    unsigned int fluid_degree;
    unsigned int pressure_degree;
    unsigned int structure_degree;
    unsigned int ale_degree;

    // Time Parameters
    double      t0;
    double	T;
    unsigned int	n_time_steps;
    double 	fluid_theta;
    double        structure_theta;

    // Domain Parameters
    double		fluid_width;
    double		fluid_height;
    double 		structure_width;
    double		structure_height;
    unsigned int	nx_f,ny_f,nx_s,ny_s;

    // Output Parameters
    bool			make_plots;
    bool			print_error;
    std::string 	convergence_mode;

    // Optimization Parameters
    double		jump_tolerance;
    double		cg_tolerance;
    double		steepest_descent_alpha;
    double		penalty_epsilon;
    unsigned int max_optimization_iterations;
    bool                true_control;
    std::string 	optimization_method;
    unsigned int        adjoint_type;

    // Solver Parameters
    bool                  richardson;
    bool                  newton; 
  };
  struct PhysicalProperties
  {
    // Problem Parameters
    int                 simulation_type;
    double		viscosity;
    double		lambda;
    double		mu;
    double		nu;
    double		rho_f;
    double 		rho_s;
    bool		moving_domain;
    bool                move_domain;
    int			n_fourier_coeffs;
    bool                navier_stokes;
    bool                stability_terms;
    bool                nonlinear_elasticity;
  };

  template <int dim>
  void declare_parameters (ParameterHandler & prm)
  {
	  // FE_Q Degrees
	  prm.declare_entry("fluid velocity degree", "2", Patterns::Integer(1),
			  "order of the finite element to use for the fluid velocity.");
	  prm.declare_entry("fluid pressure degree", "1", Patterns::Integer(1),
			  "order of the finite element to use for the fluid pressure.");
	  prm.declare_entry("structure degree", "2", Patterns::Integer(1),
			  "order of the finite element to use for the structure displacement and velocity.");
	  prm.declare_entry("ale degree", "2", Patterns::Integer(1),
			  "order of the finite element to use for the ALE mesh update.");

	  // Time Parameters
	  prm.declare_entry("t0", "0.0", Patterns::Double(0),
	  			  "time to run the simulation from.");
	  prm.declare_entry("T", "1.0", Patterns::Double(0),
	  			  "time to run the simulation until.");
	  prm.declare_entry("number of time steps", "16", Patterns::Integer(1),
	  			  "number of time steps to divide T by.");
	  prm.declare_entry("fluid theta", "0.5", Patterns::Double(0,1),
	  			  "theta value for the fluid, 0.5 is Crank-Nicolson and 1.0 is Implicit Euler.");
	  prm.declare_entry("structure theta", "0.5", Patterns::Double(0,1),
	  			  "theta value for the structure, 0.5 is midpoint and anything else isn't implemented.");

	  // Domain Parameters
	  prm.declare_entry("fluid width", "1.0", Patterns::Double(0),
	  			  "width of the fluid domain.");
	  prm.declare_entry("fluid height", "1.0", Patterns::Double(0),
	  			  "height of the fluid domain.");
	  prm.declare_entry("structure width", "1.0", Patterns::Double(0),
	  			  "width of the structure domain.");
	  prm.declare_entry("structure height", "0.25", Patterns::Double(0),
	  			  "height of the structure domain.");
	  prm.declare_entry("nx fluid", "1", Patterns::Integer(1),
	  			  "# of horizontal edges of the fluid.");
	  prm.declare_entry("ny fluid", "1", Patterns::Integer(1),
	  			  "# of vertical edges of the fluid.");
	  prm.declare_entry("nx structure", "1", Patterns::Integer(1),
	  			  "# of horizontal edges of the structure.");
	  prm.declare_entry("ny structure", "1", Patterns::Integer(1),
	  			  "# of vertical edges of the structure.");

	  // Problem Parameters
	  prm.declare_entry("simulation type", "0", Patterns::Integer(0),
	  			  "0 for analytic solution for gradient paper. 1 for bloodflow problem from SINUM paper.");
	  prm.declare_entry("viscosity", "1.0", Patterns::Double(0),
	  			  "viscosity of the fluid.");
	  prm.declare_entry("lambda", "1.0", Patterns::Double(0),
	  			  "lambda (Lame's first parameter) of the structure.");
	  prm.declare_entry("mu", "1.0", Patterns::Double(0),
	  			  "mu (shear modulus) of the structure.");
	  prm.declare_entry("nu", "0.0", Patterns::Double(0),
	  			  "nu (Poisson ratio)) of the structure.");
	  prm.declare_entry("fluid rho", "1.0", Patterns::Double(0),
	  			  "density of the fluid.");
	  prm.declare_entry("structure rho", "1.0", Patterns::Double(0),
	  			  "density of the structure.");
	  prm.declare_entry("number fourier coefficients", "20", Patterns::Integer(1),
			  	  "# of fourier coefficients to use.");
	  prm.declare_entry("navier stokes", "true", Patterns::Bool(),
			  	  "should the convective term be added.");
	  prm.declare_entry("stability terms", "true", Patterns::Bool(),
			  	  "should the stability terms used in the papers be added.");
	  prm.declare_entry("nonlinear elasticity", "false", Patterns::Bool(),
			  	  "should St. Venant-Kirchhoff tensor be used.");

	  // Output Parameters
	  prm.declare_entry("make plots", "true", Patterns::Bool(),
	  			  "create plots of the solution at each time step.");
	  prm.declare_entry("output error", "true", Patterns::Bool(),
	  			  "give error output info at each time step.");
	  prm.declare_entry("convergence method", "time",
			    Patterns::Selection("time|space"),
			    "convergence method. choice between 'time' and 'space'.");

	  // Optimization Parameters
	  prm.declare_entry("jump tolerance","1.0", Patterns::Double(0),
			    "tolerance to which the velocities must match on the interface.");
	  prm.declare_entry("cg tolerance","1.0", Patterns::Double(0),
			    "tolerance to which the inner CG optimization must converge.");
	  prm.declare_entry("steepest descent alpha","0.0001", Patterns::Double(0),
			    "tuning parameter for the steepest descent algorithm.");
	  prm.declare_entry("penalty epsilon","0.01", Patterns::Double(0),
			    "second tuning parameter for the steepest descent algorithm.");
	  prm.declare_entry("max optimization iterations","100", Patterns::Integer(1),
			    "maximum number of optimization iterations per time step.");
	  prm.declare_entry("true control","false", Patterns::Bool(),
			    "Use the true stress as the initial control at each time step.");
	  prm.declare_entry("optimization method","CG", Patterns::Selection("CG|Gradient"),
			    "optimization method choices {CG,Gradient}.");
	  prm.declare_entry("adjoint type","1", Patterns::Integer(1),
			    "adjoint displacement (1) or velocity (2) used in objective function.");

	  // Operations Parameters
	  prm.declare_entry("richardson", "true", Patterns::Bool(),
			    "use Richardson extrapolation for the convective term.");
	  prm.declare_entry("newton", "true", Patterns::Bool(),
			    "use Newton's method for convergence of nonlinearity in NS solve.");
	  prm.declare_entry("moving domain", "true", Patterns::Bool(),
	  			  "should the ALE be used.");
	  prm.declare_entry("move domain", "false", Patterns::Bool(),
	  			  "should the points be physically moved (vs using determinants).");

  }
}
