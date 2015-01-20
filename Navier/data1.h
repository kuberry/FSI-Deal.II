#ifndef DATA1_H
#define DATA1_H
#include <deal.II/base/function.h>
#include <deal.II/lac/block_vector.h>
#include <deal.II/base/tensor_base.h>
#include "parameters.h"

using namespace dealii;

template <int dim>
Tensor<2,dim> get_Jacobian(double x, double y, double t, bool move_domain);

template <int dim>
Tensor<2,dim> get_DetTimesJacobianInv(Tensor<2,dim> Jacobian);

template <int dim>
class FluidStressValues : public dealii::Function<dim>
{
 public:
  Parameters::PhysicalProperties physical_properties;
  FluidStressValues (const Parameters::PhysicalProperties & physical_properties_) : dealii::Function<dim>(dim+1), physical_properties(physical_properties_)  {}

  virtual double value (const dealii::Point<dim>   &p,
			const unsigned int  component = 0) const;
  virtual dealii::Tensor<1,dim> gradient (const dealii::Point<dim>   &p,
					  const unsigned int  component = 0) const;
};



template <int dim>
class StructureStressValues : public dealii::Function<dim>
{
 public:
  Parameters::PhysicalProperties physical_properties;
  unsigned int side;
  StructureStressValues (const Parameters::PhysicalProperties & physical_properties_) : dealii::Function<dim>(2*dim), physical_properties(physical_properties_)  {}

  virtual double value (const dealii::Point<dim>   &p,
			const unsigned int  component = 0) const;
  virtual dealii::Tensor<1,dim> gradient (const dealii::Point<dim>   &p,
					  const unsigned int  component = 0) const;
};



template <int dim>
class FluidRightHandSide : public dealii::Function<dim>
{
 public:
  Parameters::PhysicalProperties physical_properties;
  FluidRightHandSide (const Parameters::PhysicalProperties & physical_properties_) : dealii::Function<dim>(dim+1), physical_properties(physical_properties_)  {}

  virtual double value (const dealii::Point<dim>   &p,
			const unsigned int  component = 0) const;
};



template <int dim>
class StructureRightHandSide : public dealii::Function<dim>
{
 public:
  Parameters::PhysicalProperties physical_properties;
  StructureRightHandSide (const Parameters::PhysicalProperties & physical_properties_) : dealii::Function<dim>(2*dim), physical_properties(physical_properties_) {}

  virtual double value (const dealii::Point<dim>   &p,
  			const unsigned int  component) const;
};



template <int dim>
class FluidBoundaryValues : public dealii::Function<dim>
{
 public:
  Parameters::PhysicalProperties physical_properties;
  Parameters::SimulationProperties fem_properties;
  FluidBoundaryValues (const Parameters::PhysicalProperties & physical_properties_,
		       const Parameters::SimulationProperties & fem_properties_) 
    :dealii::Function<dim>(dim+1), physical_properties(physical_properties_), fem_properties(fem_properties_) {}

  virtual double value (const dealii::Point<dim>   &p,
			const unsigned int  component = 0) const;
  virtual void vector_value (const dealii::Point<dim> &p,
			     dealii::Vector<double>   &value) const;
  virtual dealii::Tensor<1,dim> gradient (const dealii::Point<dim>   &p,
					  const unsigned int  component = 0) const;
};



template <int dim>
class StructureBoundaryValues : public dealii::Function<dim>
{
 public:
  Parameters::PhysicalProperties physical_properties;
  StructureBoundaryValues (const Parameters::PhysicalProperties & physical_properties_) : dealii::Function<dim>(2*dim), physical_properties(physical_properties_) {}

  virtual double value (const dealii::Point<dim>   &p,
			const unsigned int  component = 0) const;
  virtual void vector_value (const dealii::Point<dim> &p,
			     dealii::Vector<double>   &value) const;
  virtual dealii::Tensor<1,dim> gradient (const dealii::Point<dim>   &p,
					  const unsigned int  component = 0) const;
};



template <int dim>
class AleBoundaryValues : public dealii::Function<dim>
{
 public:
  Parameters::PhysicalProperties physical_properties;
  AleBoundaryValues (const Parameters::PhysicalProperties & physical_properties_) : dealii::Function<dim>(dim), physical_properties(physical_properties_)  {}

  virtual double value (const dealii::Point<dim>   &p,
			const unsigned int  component = 0) const;
  virtual void vector_value (const dealii::Point<dim> &p,
			     dealii::Vector<double>   &value) const;
};
#endif
