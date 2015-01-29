#ifndef SOLVER_CONTROLS_H
#define SOLVER_CONTROLS_H
#include "FSI_Project.h"

template <int dim>
class DN_solver {
public:
DN_solver(FSIProblem<dim> *sim): problem_space(sim), d_n(problem_space->old_old_solution.block(1)),
				 d_np1(problem_space->old_solution.block(1)),
				 SF_eta_nm1(problem_space->old_solution.block(1).size()), 
				 SF_eta_n(problem_space->old_solution.block(1).size()),
				 w_n(problem_space->old_solution.block(1).size()), 
				 w_nm1(problem_space->old_solution.block(1).size()),
                                 error(problem_space->old_solution.block(1).size()),
				 omega_k(problem_space->fem_properties.steepest_descent_alpha),
				 omega_ref(problem_space->fem_properties.steepest_descent_alpha),
    current_time(problem_space->time)  
     { 
       multiplier = .0001; 
       AG_line_search = false;
       line_search_multiplier = 1.0;
     }
 

  ~DN_solver(){};

  void set_line_search(bool val) {
    AG_line_search = val;
  };

  bool get_line_search() {
    return AG_line_search;
  };

  void first_step() {
    omega_k = multiplier * omega_ref; 
    /* w_nm1 = w_n; */
    /* w_n *= 0; */
    /* problem_space->vector_vector_transfer_interface_dofs(problem_space->mesh_velocity.block(0),w_n,0,1,problem_space->Displacement); */

    /* d_np1 = problem_space->old_solution.block(1); */
    /* d_np1.add(1.5*problem_space->time_step,w_n); */
    /* d_np1.add(-.5*problem_space->time_step,w_nm1); */
    /* d_n = d_np1; // this is n vs np1 in iteration, not time step */
    d_np1 = d_n;
  } 

  void update(double time_, const unsigned int initialized_timestep_number) {
    // FIGURE OUT WAY TO IMPLEMENT LINE SEARCH
    if (AG_line_search) {
      // we have d_n, rebuild d_np1 with updated omega_k
      d_np1 = d_n;
      d_np1 *= (1. - line_search_multiplier*omega_k); // What about this omega_k?
      d_np1.add(line_search_multiplier*omega_k, SF_eta_n);

      problem_space->solution.block(1) = d_np1;
      problem_space->fluid_state_solve(initialized_timestep_number);
      // Take the stress from fluid and give it to the structure
      problem_space->stress.block(1)=0;
      problem_space->tmp.block(0)=0;
      problem_space->ale_transform_fluid();
      problem_space->get_fluid_stress();
      problem_space->ref_transform_fluid();
      problem_space->transfer_interface_dofs(problem_space->tmp,problem_space->stress,0,1,problem_space->Displacement);
      problem_space->structure_state_solve(initialized_timestep_number);
      
    } else {
      if (time_ != current_time) first_step();
      current_time = time_;
  

      // These commands change values outside this class
      problem_space->solution.block(1) = d_np1;
      problem_space->fluid_state_solve(initialized_timestep_number);
      // Take the stress from fluid and give it to the structure
      problem_space->stress.block(1)=0;
      problem_space->tmp.block(0)=0;
      problem_space->ale_transform_fluid();
      problem_space->get_fluid_stress();
      problem_space->ref_transform_fluid();
      problem_space->transfer_interface_dofs(problem_space->tmp,problem_space->stress,0,1,problem_space->Displacement);
      problem_space->structure_state_solve(initialized_timestep_number);
      //BlockVector<double> dtilde_np1 = solution;
      // d_np1 = omega_k * dtilde_np1 + (1. - omega_k) * d_np1
	    
  
      SF_eta_nm1 = SF_eta_n;
      SF_eta_n = problem_space->solution.block(1);

      d_n = d_np1;
      d_np1 *= (1. - omega_k);
      d_np1.add(omega_k, SF_eta_n);

      Vector<double> diff1 = d_np1;
      diff1.add(-1.0,d_n);;

      Vector<double> diff2 = diff1;
      diff2.add(1.0, SF_eta_nm1);
      diff2.add(-1.0, SF_eta_n);
	    
      Vector<double> diff1_fluid(problem_space->old_solution.block(0).size());
      Vector<double> diff2_fluid(problem_space->old_solution.block(0).size());
      problem_space->vector_vector_transfer_interface_dofs(diff1, diff1_fluid, 1, 0, problem_space->Displacement);
      problem_space->vector_vector_transfer_interface_dofs(diff2, diff2_fluid, 1, 0, problem_space->Displacement);
	    
      omega_k = multiplier * (diff1*diff2) / (diff2*diff2);
      //omega_k = problem_space->interface_inner_product(diff1,diff2) / problem_space->interface_inner_product(diff2,diff2);
      //std::cout << "omega_k: " << omega_k << std::endl;

      // This command changes values outside this class
      problem_space->solution.block(1) = d_np1;
    } 
  } 
  bool check_line_search() {
    if (get_line_search()) std::cout << "Line search in progress." << std::endl;
    if (converged()) {
      omega_k = line_search_multiplier*omega_k;
      line_search_multiplier = 1.0;
      set_line_search(false);
      return true;
    } else {
      line_search_multiplier *= 0.5;
      std::cout << "Line search multiplier: " << line_search_multiplier << std::endl;
      return false;
    }
  }

  bool converged() {
    // Check difference between current result of using d_np1 and and d_n
    error = d_n;
    error.add(-1.0, d_np1);
    problem_space->rhs_for_adjoint.block(1) *= 0;
    problem_space->vector_vector_transfer_interface_dofs(error,problem_space->rhs_for_adjoint.block(0),1,0,problem_space->Displacement);
    double error_val = problem_space->interface_error();
    std::cout << "Error: " << error_val << std::endl;
    if (error_val < problem_space->fem_properties.jump_tolerance)
      return true;
    else 
      return false;
  }


private:
  FSIProblem<dim> *problem_space;
  Vector<double> d_n;
  Vector<double> d_np1;
  Vector<double> SF_eta_nm1;
  Vector<double> SF_eta_n;
  Vector<double> w_n; 
  Vector<double> w_nm1;
  Vector<double> error;
  double omega_k; 
  double omega_ref;
  double current_time;
  double multiplier;
  bool AG_line_search;
  double line_search_multiplier;
};

#endif
