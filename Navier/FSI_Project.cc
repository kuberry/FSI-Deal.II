#include "FSI_Project.h"
#include <stdlib.h>     /* atoi */

int main (int argc, char *argv[])
{
  const unsigned int dim = 3;
  //Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
  if (argc < 2)
    {
      std::cerr << "  usage: ./FSIProblem <parameter-file.prm>" << std::endl;
      return -1;
    }
  try
    {
      using namespace dealii;

      deallog.depth_console (0);

      ParameterHandler prm;
      
      Parameters::declare_parameters<dim>(prm);

      bool success=prm.read_input(argv[1]);
      if (!success)
      {
    	  std::cerr << "Couldn't read filename: " << argv[1] << std::endl;
      }
      if (argc == 2) {
	FSIProblem<dim> fsi_solver(prm);
	fsi_solver.run();
      } else if (argc == 3) {
	FSIProblem<dim> fsi_solver(prm, std::atoi(argv[2]));
	fsi_solver.run();	
      }
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


template class FSIProblem<2>;
template class FSIProblem<3>;
