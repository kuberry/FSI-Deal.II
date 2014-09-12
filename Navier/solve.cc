#include "FSI_Project.h"

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

template void FSIProblem<2>::solve (const SparseDirectUMFPACK& direct_solver, const int block_num, Mode enum_);
