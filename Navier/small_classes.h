#ifndef SMALL_CLASSES_H
#define SMALL_CLASSES_H
#include "data1.h"
#include "parameters.h"

using namespace dealii;

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
struct PerTaskData {
  FullMatrix<double> cell_matrix;
  Vector<double> cell_rhs;
  std::vector<types::global_dof_index> dof_indices;
  SparseMatrix<double>* global_matrix;
  Vector<double>* global_rhs;
  bool assemble_matrix;

  PerTaskData (const FiniteElement<dim> &fe, SparseMatrix<double>* matrix_, Vector<double>* rhs_, const bool assemble_matrix_)
    :
  cell_matrix (fe.dofs_per_cell, fe.dofs_per_cell),
    cell_rhs (fe.dofs_per_cell),
    dof_indices (fe.dofs_per_cell),
    global_matrix(matrix_),
    global_rhs(rhs_),
    assemble_matrix(assemble_matrix_)
  {}
};

/* template <int dim> */
/* struct Structure_PerTaskData : public PerTaskData<dim> { */
/*   Structure_PerTaskData (const FiniteElement<dim> &fe, SparseMatrix<double>* matrix_, Vector<double>* rhs_) */
/*     : PerTaskData<dim>(fe,matrix_,rhs_)  */
/*     {} */
/* }; */

template <int dim>
struct BaseScratchData {
  unsigned int mode_type;
  FEValues<dim> fe_values;
  unsigned int n_q_points;

  BaseScratchData (const FiniteElement<dim> &fe,
	       const Quadrature<dim> &quadrature,
	       const UpdateFlags update_flags,
	       const unsigned int mode_type_
	       )
  :
  mode_type(mode_type_),
    fe_values (fe, quadrature, update_flags),
    n_q_points (quadrature.size())
  {}

  BaseScratchData (const BaseScratchData &scratch)
  :
  mode_type(scratch.mode_type),
    fe_values (scratch.fe_values.get_fe(),
	       scratch.fe_values.get_quadrature(),
	       scratch.fe_values.get_update_flags()
	       ),
    n_q_points(scratch.n_q_points)
  {}
};



template <int dim>
struct FullScratchData : public BaseScratchData<dim> {
  FEFaceValues<dim> fe_face_values;
  unsigned int n_face_q_points;
  FullScratchData ( const FiniteElement<dim> &fe,
			  const Quadrature<dim> &quadrature,
			  const UpdateFlags update_flags,
			  const Quadrature<dim-1> &face_quadrature,
			  const UpdateFlags face_update_flags,
			  const unsigned int mode_type_
			  )
    : BaseScratchData<dim>(fe, quadrature, update_flags, mode_type_),
    fe_face_values (fe, face_quadrature, face_update_flags),
    n_face_q_points(face_quadrature.size())
      {}
      
  FullScratchData (const FullScratchData &scratch)
    : BaseScratchData<dim>(scratch),
    fe_face_values(scratch.fe_face_values.get_fe(),
  		   scratch.fe_face_values.get_quadrature(),
  		   scratch.fe_face_values.get_update_flags()
  		   ),
    n_face_q_points(scratch.n_face_q_points)
      {}
};


#endif
