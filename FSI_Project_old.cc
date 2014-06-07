/* ---------------------------------------------------------------------
 *  Time Dependent FSI Problem with ALE on Fluid Domain
 * ---------------------------------------------------------------------
 *
 * Originally authored by Wolfgang Bangerth, Texas A&M University, 2011
 * and ammended significantly by Paul Kuberry, Clemson University, 2014
 */


#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/function.h>
#include <deal.II/base/utilities.h>

#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/sparse_direct.h>
#include <deal.II/lac/constraint_matrix.h>

#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/grid_out.h>

#include <deal.II/dofs/dof_tools.h>
#include <deal.II/dofs/dof_accessor.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_nothing.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>

#include <deal.II/hp/dof_handler.h>
#include <deal.II/hp/fe_collection.h>
#include <deal.II/hp/fe_values.h>

#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/error_estimator.h>

#include <fstream>
#include <sstream>


namespace FSIProblem
{
  using namespace dealii;


  template <int dim>
  class FluidStructureProblem
  {
  public:
    FluidStructureProblem (const unsigned int stokes_degree,
                           const unsigned int elasticity_degree);
    void run ();

  private:
    enum
    {
      fluid_domain_id,
      solid_domain_id
    };

    static bool
    cell_is_in_fluid_domain (const typename hp::DoFHandler<dim>::cell_iterator &cell);

    static bool
    cell_is_in_solid_domain (const typename hp::DoFHandler<dim>::cell_iterator &cell);


    void make_grid ();
    void set_active_fe_indices ();
    void setup_dofs ();
    void assemble_system ();
    void assemble_interface_term (const FEFaceValuesBase<dim>          &elasticity_fe_face_values,
                                  const FEFaceValuesBase<dim>          &stokes_fe_face_values,
                                  std::vector<Tensor<1,dim> >          &elasticity_phi,
                                  std::vector<SymmetricTensor<2,dim> > &stokes_symgrad_phi_u,
                                  std::vector<double>                  &stokes_phi_p,
                                  FullMatrix<double>                   &local_interface_matrix) const;
    void solve ();
    void output_results () const;
    void refine_mesh ();

    const unsigned int    stokes_degree;
    const unsigned int    elasticity_degree;

    Triangulation<dim>    	triangulation;
    FESystem<dim>         	stokes_fe;
    FESystem<dim>         	elasticity_fe;
    hp::FECollection<dim> 	fe_collection;
    hp::DoFHandler<dim>   	dof_handler;

    ConstraintMatrix       	constraints;

    SparsityPattern        	sparsity_pattern;
    SparseMatrix<double>  	system_matrix;

    Vector<double>        	solution;
    Vector<double>        	old_solution;
    Vector<double>        	system_rhs;

    const double          	viscosity;
    const double          	lambda;
    const double          	mu;

    double                	time;
    double                	time_step;
    unsigned int         	timestep_number;

    const double         	theta;
  };



  template <int dim>
  class StokesBoundaryValues : public Function<dim>
  {
  public:
    StokesBoundaryValues (const double viscosity_) : Function<dim>(dim+1+dim), viscosity(viscosity_) {}

    virtual double value (const Point<dim>   &p,
                          const unsigned int  component = 0) const;

    virtual void vector_value (const Point<dim> &p,
                               Vector<double>   &value) const;
  private:
    const double viscosity;
  };


//  template <int dim>
//  double
//  StokesBoundaryValues<dim>::value (const Point<dim>  &p,
//                                    const unsigned int component) const
//  {
//    Assert (component < this->n_components,
//            ExcIndexRange (component, 0, this->n_components));
//
//    if (component == dim-1)
//      switch (dim)
//        {
//        case 2:
//          return std::sin(numbers::PI*p[0]);
//        case 3:
//          return std::sin(numbers::PI*p[0]) * std::sin(numbers::PI*p[1]);
//        default:
//          Assert (false, ExcNotImplemented());
//        }
//
//    return 0;
//  }

  template <int dim>
  double
  StokesBoundaryValues<dim>::value (const Point<dim>  &p,
                                    const unsigned int component) const
  {
    Assert (component < this->n_components,
            ExcIndexRange (component, 0, this->n_components));

    switch(component)
    {
    	case 0:
    		return 0;
    	case 1:
    		return 100;
    	default:
    	    return 0;
    }


  }


  template <int dim>
  void
  StokesBoundaryValues<dim>::vector_value (const Point<dim> &p,
                                           Vector<double>   &values) const
  {
    for (unsigned int c=0; c<this->n_components; ++c)
      values(c) = StokesBoundaryValues<dim>::value (p, c);
  }

  template <int dim>
    class ElasticBoundaryValues : public Function<dim>
    {
    public:
      ElasticBoundaryValues () : Function<dim>(dim+1+dim) {}

      virtual double value (const Point<dim>   &p,
                            const unsigned int  component = 0) const;

      virtual void vector_value (const Point<dim> &p,
                                 Vector<double>   &value) const;
    };

  template <int dim>
  double
  ElasticBoundaryValues<dim>::value (const Point<dim>  &p,
                                    const unsigned int component) const
  {
    Assert (component < this->n_components,
            ExcIndexRange (component, 0, this->n_components));
    switch(component)
    {
    	case dim+1:
    		return 0;
    	case dim+2:
    		return 100;
    	default:
    	    return 0;
    }
  }


  template <int dim>
  void
  ElasticBoundaryValues<dim>::vector_value (const Point<dim> &p,
                                           Vector<double>   &values) const
  {
    for (unsigned int c=0; c<this->n_components; ++c)
      values(c) = ElasticBoundaryValues<dim>::value (p, c);
  }
<<<<<<< HEAD:FSI_Project_old.cc



  template <int dim>
  class ExactValues : public Function<dim>
  {
  public:
    ExactValues () : Function<dim>(dim+1+dim), fluid(0) {}
    StokesBoundaryValues<dim> fluid;
    ElasticBoundaryValues<dim> structure;
    void set_time(const double time)
    {
    	fluid.set_time(time);
    	structure.set_time(time);
    }
    virtual double value (const Point<dim>   &p,
                          const unsigned int  component = 0) const;

    virtual void vector_value (const Point<dim> &p,
                               Vector<double>   &value) const;
  };

  template <int dim>
  double
  ExactValues<dim>::value (const Point<dim>  &p,
                                    const unsigned int component) const
  {
    Assert (component < this->n_components,
            ExcIndexRange (component, 0, this->n_components));
    if (component<dim+1)
    {
    	return fluid.value(p,component);
    }
    else
    {
    	return structure.value(p,component);
    }
  }


  template <int dim>
  void
  ExactValues<dim>::vector_value (const Point<dim> &p,
                                           Vector<double>   &values) const
  {
    for (unsigned int c=0; c<dim+1; ++c)
          values(c) = fluid.value(p, c);
    for (unsigned int c=dim+1; c<2*dim+1; ++c)
          values(c) = structure.value(p, c);
  }
=======
>>>>>>> 53e72c5830d38bd9fa69e0760bee7ef8a9e40c3e:FSI_Project.cc

  template <int dim>
  class RightHandSide : public Function<dim>
  {
  public:
    RightHandSide () : Function<dim>(dim+1) {}

    virtual double value (const Point<dim>   &p,
                          const unsigned int  component = 0) const;

    virtual void vector_value (const Point<dim> &p,
                               Vector<double>   &value) const;

  };




  template <int dim>
  double
  RightHandSide<dim>::value (const Point<dim>  &/*p*/,
                             const unsigned int /*component*/) const
  {
    return 0;
  }


  template <int dim>
  void
  RightHandSide<dim>::vector_value (const Point<dim> &p,
                                    Vector<double>   &values) const
  {
    for (unsigned int c=0; c<this->n_components; ++c)
      values(c) = RightHandSide<dim>::value (p, c);
  }





  template <int dim>


  FluidStructureProblem<dim>::
  FluidStructureProblem (const unsigned int stokes_degree,
                         const unsigned int elasticity_degree)
    :
    stokes_degree (stokes_degree),
    elasticity_degree (elasticity_degree),
    triangulation (Triangulation<dim>::maximum_smoothing),
    stokes_fe (FE_Q<dim>(stokes_degree+1), dim,
               FE_Q<dim>(stokes_degree), 1,
               FE_Nothing<dim>(), dim),
    elasticity_fe (FE_Nothing<dim>(), dim,
                   FE_Nothing<dim>(), 1,
                   FE_Q<dim>(elasticity_degree), dim),
    dof_handler (triangulation),
    viscosity (2),
    lambda (1),
    mu (1),
    time_step(1./32.),
    theta(1.0)
  {
    fe_collection.push_back (stokes_fe);
    fe_collection.push_back (elasticity_fe);
    Assert (dim==2, ExcNotImplemented());
    Assert (stokes_degree==elasticity_degree,ExcInvalidState());
  }




  template <int dim>
  bool
  FluidStructureProblem<dim>::
  cell_is_in_fluid_domain (const typename hp::DoFHandler<dim>::cell_iterator &cell)
  {
    return (cell->material_id() == fluid_domain_id);
  }


  template <int dim>
  bool
  FluidStructureProblem<dim>::
  cell_is_in_solid_domain (const typename hp::DoFHandler<dim>::cell_iterator &cell)
  {
    return (cell->material_id() == solid_domain_id);
  }



  template <int dim>
  void
  FluidStructureProblem<dim>::make_grid ()
  {
	// This is DOMAIN SPECIFIC!
	const unsigned int nx_f = 10; // Number of horizontal edges for the fluid
	const unsigned int ny_f = 10; // Number of vertical edges for the fluid
	const unsigned int nx_s = nx_f; // Number of horizontal edges for the structure
	const unsigned int ny_s = 2; // Number of vertical edges for the structure

	const double fluid_length=1;
	const double fluid_height=1;
	const double structure_length=fluid_length;
	const double structure_height=0.25;
    const double INTERFACE_HEIGHT=fluid_height;
	// Structure sits on top of fluid
	Assert(nx_f==nx_s,ExcNotImplemented()); // Check that the interface edges are equally refined
	Assert(std::fabs(fluid_length-structure_length)<1e-5,ExcNotImplemented());

	std::vector<double> x_scales(nx_f,fluid_length/((double)nx_f));
	std::vector<double> y_scales(ny_f+ny_s);
	for(unsigned int i=0;i<ny_f+ny_s;++i){
		if(i<ny_f)
		{
			y_scales[i]=fluid_height/((double)ny_f);
		}
		else
		{
			y_scales[i]=structure_height/((double)ny_s);
		}
	}

	std::vector<std::vector<double>> step_sizes(2);
	step_sizes[0]=x_scales;
	step_sizes[1]=y_scales;

	Point<dim> bottom_left(0,0);
	Point<dim> top_right(fluid_length,fluid_height+structure_height);

	GridGenerator::subdivided_hyper_rectangle (triangulation, step_sizes, bottom_left, top_right,false);

    // This is DOMAIN SPECIFIC!
    for (typename Triangulation<dim>::active_cell_iterator
         cell = dof_handler.begin_active();
         cell != dof_handler.end(); ++cell)
      if (std::fabs(cell->center()[1]) < INTERFACE_HEIGHT)
        cell->set_material_id (fluid_domain_id);
      else
        cell->set_material_id (solid_domain_id);



    for (typename Triangulation<dim>::active_cell_iterator
         cell = triangulation.begin_active();
         cell != triangulation.end(); ++cell)

      for (unsigned int f=0; f<GeometryInfo<dim>::faces_per_cell; ++f)
        if (cell->face(f)->at_boundary())
        {
            if (std::fabs(cell->face(f)->center()[1])<1e-5*(1./ny_f))
            { // BOTTOM OF FLUID BOUNDARY
            	cell->face(f)->set_all_boundary_indicators(0);
            }
            else if (std::fabs(cell->face(f)->center()[1]-(INTERFACE_HEIGHT+structure_height))<1e-5*(1./ny_s))
            { // TOP OF STRUCTURE BOUNDARY
            	cell->face(f)->set_all_boundary_indicators(5);
            }
            else if (std::fabs(cell->face(f)->center()[0])<1e-5*(1./nx_f) && cell->face(f)->center()[1]-INTERFACE_HEIGHT<=0)
            { // LEFT SIDE OF FLUID BOUNDARY
            	cell->face(f)->set_all_boundary_indicators(3);
            }
            else if (std::fabs(cell->face(f)->center()[0])<1e-5*(1./nx_f) && cell->face(f)->center()[1]-INTERFACE_HEIGHT>0)
            { // LEFT SIDE OF STRUCTURE BOUNDARY
            	cell->face(f)->set_all_boundary_indicators(6);
            }
            else if (std::fabs(cell->face(f)->center()[0]-fluid_length)<1e-5*(1./nx_f) && cell->face(f)->center()[1]-INTERFACE_HEIGHT<=0)
            { // RIGHT SIDE OF FLUID BOUNDARY
            	cell->face(f)->set_all_boundary_indicators(1);
            }
            else if (std::fabs(cell->face(f)->center()[0]-structure_length)<1e-5*(1./nx_f) && cell->face(f)->center()[1]-INTERFACE_HEIGHT>0)
            { // RIGHT SIDE OF STRUCTURE BOUNDARY
            	cell->face(f)->set_all_boundary_indicators(4);
            }
        }
        else if (std::fabs(cell->face(f)->center()[1]-INTERFACE_HEIGHT)<1e-5*std::min(1./ny_f,1./ny_s))
        { // INTERFACE BOUNDARY
            	cell->face(f)->set_all_boundary_indicators(2);
        }

  }


  template <int dim>
  void
  FluidStructureProblem<dim>::set_active_fe_indices ()
  {
    for (typename hp::DoFHandler<dim>::active_cell_iterator
         cell = dof_handler.begin_active();
         cell != dof_handler.end(); ++cell)
      {
        if (cell_is_in_fluid_domain(cell))
          cell->set_active_fe_index (0);
        else if (cell_is_in_solid_domain(cell))
          cell->set_active_fe_index (1);
        else
          Assert (false, ExcNotImplemented());
      }
  }



  template <int dim>
  void
  FluidStructureProblem<dim>::setup_dofs ()
  {
    set_active_fe_indices ();
    dof_handler.distribute_dofs (fe_collection);
    {
<<<<<<< HEAD:FSI_Project_old.cc
	      constraints.clear ();
	      DoFTools::make_hanging_node_constraints (dof_handler,
	                                               constraints);
=======
      constraints.clear ();
      DoFTools::make_hanging_node_constraints (dof_handler,
                                               constraints);

      const FEValuesExtractors::Vector velocities(0);
	  const FEValuesExtractors::Vector displacements(dim+1);

      unsigned int j=0;


      for (unsigned int k=0; k<7; ++k)
      {
    	  if (k!=j && k!=1 && k!=3 && k!=5 && k!=2)
    	  {
			  VectorTools::interpolate_boundary_values (dof_handler,
														k,
														ZeroFunction<dim>(dim+1+dim),
														constraints,
														fe_collection.component_mask(velocities));
			  VectorTools::interpolate_boundary_values (dof_handler,
														k,
														ZeroFunction<dim>(dim+1+dim),
														constraints,
														fe_collection.component_mask(displacements));
    	  }
      }
      VectorTools::interpolate_boundary_values (dof_handler,
                                                j,
                                                StokesBoundaryValues<dim>(),
                                                constraints,
                                                fe_collection.component_mask(velocities));
//      VectorTools::interpolate_boundary_values (dof_handler,
//                                                j,
//                                                ElasticBoundaryValues<dim>(),
//                                                constraints,
//                                                fe_collection.component_mask(displacements));
    }
>>>>>>> 53e72c5830d38bd9fa69e0760bee7ef8a9e40c3e:FSI_Project.cc

    }
    {
      std::vector<types::global_dof_index> local_face_dof_indices (stokes_fe.dofs_per_face);
      for (typename hp::DoFHandler<dim>::active_cell_iterator
           cell = dof_handler.begin_active();
           cell != dof_handler.end(); ++cell)
        if (cell_is_in_fluid_domain (cell))
          for (unsigned int f=0; f<GeometryInfo<dim>::faces_per_cell; ++f)
            if (!cell->at_boundary(f))
              {
                bool face_is_on_interface = false;

                if ((cell->neighbor(f)->has_children() == false)
                    &&
                    (cell_is_in_solid_domain (cell->neighbor(f))))
                  face_is_on_interface = true;
                else if (cell->neighbor(f)->has_children() == true)
                  {
                    for (unsigned int sf=0; sf<cell->face(f)->n_children(); ++sf)
                      if (cell_is_in_solid_domain (cell->neighbor_child_on_subface
                                                   (f, sf)))
                        {
                          face_is_on_interface = true;
                          break;
                        }
                  }

                if (face_is_on_interface)
                  {
                    cell->face(f)->get_dof_indices (local_face_dof_indices, 0);
                    for (unsigned int i=0; i<local_face_dof_indices.size(); ++i)
                      if (stokes_fe.face_system_to_component_index(i).first < dim)
                        constraints.add_line (local_face_dof_indices[i]);
                  }
              }

    }
    constraints.close ();


    std::cout << "   Number of active cells: "
              << triangulation.n_active_cells()
              << std::endl
              << "   Number of degrees of freedom: "
              << dof_handler.n_dofs()
              << std::endl;

    {
      CompressedSimpleSparsityPattern csp (dof_handler.n_dofs(),
                                           dof_handler.n_dofs());

      Table<2,DoFTools::Coupling> cell_coupling (fe_collection.n_components(),
                                                 fe_collection.n_components());
      Table<2,DoFTools::Coupling> face_coupling (fe_collection.n_components(),
                                                 fe_collection.n_components());

      for (unsigned int c=0; c<fe_collection.n_components(); ++c)
        for (unsigned int d=0; d<fe_collection.n_components(); ++d)
          {
            if (((c<dim+1) && (d<dim+1)
                 && !((c==dim) && (d==dim)))
                ||
                ((c>=dim+1) && (d>=dim+1)))
              cell_coupling[c][d] = DoFTools::always;

            if ((c>=dim+1) && (d<dim+1))
              face_coupling[c][d] = DoFTools::always;
          }

      DoFTools::make_flux_sparsity_pattern (dof_handler, csp,
                                            cell_coupling, face_coupling);
      constraints.condense (csp);
      sparsity_pattern.copy_from (csp);
    }

    system_matrix.reinit (sparsity_pattern);

    solution.reinit (dof_handler.n_dofs());
    old_solution.reinit (dof_handler.n_dofs());
    system_rhs.reinit (dof_handler.n_dofs());
  }




  template <int dim>
  void FluidStructureProblem<dim>::assemble_system ()
  {
    system_matrix=0;
    system_rhs=0;

    const QGauss<dim> stokes_quadrature(stokes_degree+2);
    const QGauss<dim> elasticity_quadrature(elasticity_degree+2);

    hp::QCollection<dim>  q_collection;
    q_collection.push_back (stokes_quadrature);
    q_collection.push_back (elasticity_quadrature);

    hp::FEValues<dim> hp_fe_values (fe_collection, q_collection,
                                    update_values    |
                                    update_quadrature_points  |
                                    update_JxW_values |
                                    update_gradients);

    const QGauss<dim-1> common_face_quadrature(std::max (stokes_degree+2,
                                                         elasticity_degree+2));

    FEFaceValues<dim>    stokes_fe_face_values (stokes_fe,
                                                common_face_quadrature,
                                                update_JxW_values |
                                                update_normal_vectors |
                                                update_gradients);
    FEFaceValues<dim>    elasticity_fe_face_values (elasticity_fe,
                                                    common_face_quadrature,
                                                    update_values);
    FESubfaceValues<dim> stokes_fe_subface_values (stokes_fe,
                                                   common_face_quadrature,
                                                   update_JxW_values |
                                                   update_normal_vectors |
                                                   update_gradients);
    FESubfaceValues<dim> elasticity_fe_subface_values (elasticity_fe,
                                                       common_face_quadrature,
                                                       update_values);

    const unsigned int        stokes_dofs_per_cell     = stokes_fe.dofs_per_cell;
    const unsigned int        elasticity_dofs_per_cell = elasticity_fe.dofs_per_cell;

    FullMatrix<double>        local_matrix;
    FullMatrix<double>        local_interface_matrix (elasticity_dofs_per_cell,
                                                      stokes_dofs_per_cell);
    Vector<double>            local_rhs;

    std::vector<types::global_dof_index> local_dof_indices;
    std::vector<types::global_dof_index> neighbor_dof_indices (stokes_dofs_per_cell);

    const RightHandSide<dim>  right_hand_side;

    const FEValuesExtractors::Vector     velocities (0);
    const FEValuesExtractors::Scalar     pressure (dim);
    const FEValuesExtractors::Vector     displacements (dim+1);

    std::vector<SymmetricTensor<2,dim> > stokes_symgrad_phi_u (stokes_dofs_per_cell);
    std::vector<double>                  stokes_div_phi_u     (stokes_dofs_per_cell);
    std::vector<double>                  stokes_phi_p         (stokes_dofs_per_cell);

    std::vector<Tensor<2,dim> >          elasticity_grad_phi (elasticity_dofs_per_cell);
    std::vector<double>                  elasticity_div_phi  (elasticity_dofs_per_cell);
    std::vector<Tensor<1,dim> >          elasticity_phi      (elasticity_dofs_per_cell);

    typename hp::DoFHandler<dim>::active_cell_iterator
    cell = dof_handler.begin_active(),
    endc = dof_handler.end();
    for (; cell!=endc; ++cell)
      {
        hp_fe_values.reinit (cell);

        const FEValues<dim> &fe_values = hp_fe_values.get_present_fe_values();

        local_matrix.reinit (cell->get_fe().dofs_per_cell,
                             cell->get_fe().dofs_per_cell);
        local_rhs.reinit (cell->get_fe().dofs_per_cell);

        if (cell_is_in_fluid_domain (cell))
          {
            const unsigned int dofs_per_cell = cell->get_fe().dofs_per_cell;
            Assert (dofs_per_cell == stokes_dofs_per_cell,
                    ExcInternalError());

            for (unsigned int q=0; q<fe_values.n_quadrature_points; ++q)
              {
                for (unsigned int k=0; k<dofs_per_cell; ++k)
                  {
                    stokes_symgrad_phi_u[k] = fe_values[velocities].symmetric_gradient (k, q);
                    stokes_div_phi_u[k]     = fe_values[velocities].divergence (k, q);
                    stokes_phi_p[k]         = fe_values[pressure].value (k, q);
                  }

                for (unsigned int i=0; i<dofs_per_cell; ++i)
                  for (unsigned int j=0; j<dofs_per_cell; ++j)
                    local_matrix(i,j) += (2 * viscosity * stokes_symgrad_phi_u[i] * stokes_symgrad_phi_u[j]
                                          - stokes_div_phi_u[i] * stokes_phi_p[j]
                                          - stokes_phi_p[i] * stokes_div_phi_u[j])
                                         * fe_values.JxW(q);
              }
          }
        else
          {
            const unsigned int dofs_per_cell = cell->get_fe().dofs_per_cell;
            Assert (dofs_per_cell == elasticity_dofs_per_cell,
                    ExcInternalError());

            for (unsigned int q=0; q<fe_values.n_quadrature_points; ++q)
              {
                for (unsigned int k=0; k<dofs_per_cell; ++k)
                  {
                    elasticity_grad_phi[k] = fe_values[displacements].gradient (k, q);
                    elasticity_div_phi[k]  = fe_values[displacements].divergence (k, q);
                  }

                for (unsigned int i=0; i<dofs_per_cell; ++i)
                  for (unsigned int j=0; j<dofs_per_cell; ++j)
                    {
                      local_matrix(i,j)
                      +=  (lambda *
                           elasticity_div_phi[i] * elasticity_div_phi[j]
                           +
                           mu *
                           scalar_product(elasticity_grad_phi[i], elasticity_grad_phi[j])
                           +
                           mu *
                           scalar_product(elasticity_grad_phi[i], transpose(elasticity_grad_phi[j]))
                          )
                          *
                          fe_values.JxW(q);
                    }
              }
          }

        local_dof_indices.resize (cell->get_fe().dofs_per_cell);
        cell->get_dof_indices (local_dof_indices);
        constraints.distribute_local_to_global (local_matrix, local_rhs,
                                                local_dof_indices,
                                                system_matrix, system_rhs);

        if (cell_is_in_solid_domain (cell))
          for (unsigned int f=0; f<GeometryInfo<dim>::faces_per_cell; ++f)
            if (cell->at_boundary(f) == false)
              {
                if ((cell->neighbor(f)->level() == cell->level())
                    &&
                    (cell->neighbor(f)->has_children() == false)
                    &&
                    cell_is_in_fluid_domain (cell->neighbor(f)))
                  {
                    elasticity_fe_face_values.reinit (cell, f);
                    stokes_fe_face_values.reinit (cell->neighbor(f),
                                                  cell->neighbor_of_neighbor(f));

                    assemble_interface_term (elasticity_fe_face_values, stokes_fe_face_values,
                                             elasticity_phi, stokes_symgrad_phi_u, stokes_phi_p,
                                             local_interface_matrix);

                    cell->neighbor(f)->get_dof_indices (neighbor_dof_indices);
                    constraints.distribute_local_to_global(local_interface_matrix,
                                                           local_dof_indices,
                                                           neighbor_dof_indices,
                                                           system_matrix);
                  }

                else if ((cell->neighbor(f)->level() == cell->level())
                         &&
                         (cell->neighbor(f)->has_children() == true))
                  {
                    for (unsigned int subface=0;
                         subface<cell->face(f)->n_children();
                         ++subface)
                      if (cell_is_in_fluid_domain (cell->neighbor_child_on_subface
                                                   (f, subface)))
                        {
                          elasticity_fe_subface_values.reinit (cell,
                                                               f,
                                                               subface);
                          stokes_fe_face_values.reinit (cell->neighbor_child_on_subface (f, subface),
                                                        cell->neighbor_of_neighbor(f));

                          assemble_interface_term (elasticity_fe_subface_values,
                                                   stokes_fe_face_values,
                                                   elasticity_phi,
                                                   stokes_symgrad_phi_u, stokes_phi_p,
                                                   local_interface_matrix);

                          cell->neighbor_child_on_subface (f, subface)
                          ->get_dof_indices (neighbor_dof_indices);
                          constraints.distribute_local_to_global(local_interface_matrix,
                                                                 local_dof_indices,
                                                                 neighbor_dof_indices,
                                                                 system_matrix);
                        }
                  }

                else if (cell->neighbor_is_coarser(f)
                         &&
                         cell_is_in_fluid_domain(cell->neighbor(f)))
                  {
                    elasticity_fe_face_values.reinit (cell, f);
                    stokes_fe_subface_values.reinit (cell->neighbor(f),
                                                     cell->neighbor_of_coarser_neighbor(f).first,
                                                     cell->neighbor_of_coarser_neighbor(f).second);

                    assemble_interface_term (elasticity_fe_face_values,
                                             stokes_fe_subface_values,
                                             elasticity_phi,
                                             stokes_symgrad_phi_u, stokes_phi_p,
                                             local_interface_matrix);

                    cell->neighbor(f)->get_dof_indices (neighbor_dof_indices);
                    constraints.distribute_local_to_global(local_interface_matrix,
                                                           local_dof_indices,
                                                           neighbor_dof_indices,
                                                           system_matrix);

                  }
              }
      }
  }



  template <int dim>
  void
  FluidStructureProblem<dim>::
  assemble_interface_term (const FEFaceValuesBase<dim>          &elasticity_fe_face_values,
                           const FEFaceValuesBase<dim>          &stokes_fe_face_values,
                           std::vector<Tensor<1,dim> >          &elasticity_phi,
                           std::vector<SymmetricTensor<2,dim> > &stokes_symgrad_phi_u,
                           std::vector<double>                  &stokes_phi_p,
                           FullMatrix<double>                   &local_interface_matrix) const
  {
    Assert (stokes_fe_face_values.n_quadrature_points ==
            elasticity_fe_face_values.n_quadrature_points,
            ExcInternalError());
    const unsigned int n_face_quadrature_points
      = elasticity_fe_face_values.n_quadrature_points;

    const FEValuesExtractors::Vector velocities (0);
    const FEValuesExtractors::Scalar pressure (dim);
    const FEValuesExtractors::Vector displacements (dim+1);

    local_interface_matrix = 0;
    for (unsigned int q=0; q<n_face_quadrature_points; ++q)
      {
        const Tensor<1,dim> normal_vector = stokes_fe_face_values.normal_vector(q);

        for (unsigned int k=0; k<stokes_fe_face_values.dofs_per_cell; ++k)
          stokes_symgrad_phi_u[k] = stokes_fe_face_values[velocities].symmetric_gradient (k, q);
        for (unsigned int k=0; k<elasticity_fe_face_values.dofs_per_cell; ++k)
          elasticity_phi[k] = elasticity_fe_face_values[displacements].value (k,q);

        for (unsigned int i=0; i<elasticity_fe_face_values.dofs_per_cell; ++i)
          for (unsigned int j=0; j<stokes_fe_face_values.dofs_per_cell; ++j)
            local_interface_matrix(i,j) += -((2 * viscosity *
                                              (stokes_symgrad_phi_u[j] *
                                               normal_vector)
                                              +
                                              stokes_phi_p[j] *
                                              normal_vector) *
                                             elasticity_phi[i] *
                                             stokes_fe_face_values.JxW(q));
      }
  }



  template <int dim>
  void
  FluidStructureProblem<dim>::solve ()
  {
    SparseDirectUMFPACK direct_solver;
    direct_solver.initialize (system_matrix);
    direct_solver.vmult (solution, system_rhs);

    constraints.distribute (solution);
  }




  template <int dim>
  void
  FluidStructureProblem<dim>::
  output_results ()  const
  {
	std::cout << "Time step: " << time_step << std::endl;
    std::vector<std::string> solution_names (dim, "velocity");
    solution_names.push_back ("pressure");
    for (unsigned int d=0; d<dim; ++d)
      solution_names.push_back ("displacement");

    std::vector<DataComponentInterpretation::DataComponentInterpretation>
    data_component_interpretation
    (dim, DataComponentInterpretation::component_is_part_of_vector);
    data_component_interpretation
    .push_back (DataComponentInterpretation::component_is_scalar);
    for (unsigned int d=0; d<dim; ++d)
      data_component_interpretation
      .push_back (DataComponentInterpretation::component_is_part_of_vector);

    DataOut<dim,hp::DoFHandler<dim> > data_out;
    data_out.attach_dof_handler (dof_handler);

    data_out.add_data_vector (solution, solution_names,
                              DataOut<dim,hp::DoFHandler<dim> >::type_dof_data,
                              data_component_interpretation);
    data_out.build_patches ();

//    std::ostringstream filename;
//    filename << "solution-"
//             << Utilities::int_to_string (time_step, 2)
//             << ".vtk";
    const std::string filename = "solution-"
                                     + Utilities::int_to_string(timestep_number, 3) +
                                     ".vtk";

    //std::ofstream output (filename.str().c_str());
    std::ofstream output(filename.c_str());
    data_out.write_vtk (output);
<<<<<<< HEAD:FSI_Project_old.cc
=======

    std::ofstream out ("grid.gl");
    GridOut grid_out;
    grid_out.write_mathgl (triangulation, out);
>>>>>>> 53e72c5830d38bd9fa69e0760bee7ef8a9e40c3e:FSI_Project.cc
  }



  template <int dim>
  void
  FluidStructureProblem<dim>::refine_mesh ()
  {
//    triangulation.refine_global(3);
//
//    SolutionTransfer<dim> solution_trans(dof_handler);
//
//    Vector<double> previous_solution;
//    previous_solution = solution;
//    triangulation.prepare_coarsening_and_refinement();
//    solution_trans.prepare_for_coarsening_and_refinement(previous_solution);
//
//    // Now everything is ready, so do the refinement and recreate the dof
//    // structure on the new grid, and initialize the matrix structures and the
//    // new vectors in the <code>setup_system</code> function. Next, we actually
//    // perform the interpolation of the solution from old to new grid.
//    triangulation.execute_coarsening_and_refinement ();
//    setup_system ();
//
//    solution_trans.interpolate(previous_solution, solution);
  }




  template <int dim>
  void FluidStructureProblem<dim>::run ()
  {

    make_grid ();
    //refine_mesh ();
    setup_dofs ();

    const FEValuesExtractors::Vector velocities(0);
    const FEValuesExtractors::Vector displacements(dim+1);
//    VectorTools::interpolate(dof_handler, /* dof_handler */
//                             ZeroFunction<dim>(2*dim+1),
//                             old_solution);
//    ,
//                             fe_collection.component_mask(velocities));
//    VectorTools::interpolate_based_on_material_id ( fe_collection,
//    							 dof_handler,
//    							 fluid_domain_id,
//                                 ZeroFunction<dim>(2*dim+1),
//                                 old_solution,
//                                 fe_collection.component_mask(velocities));
    solution = old_solution; // Needed to give Newton iterations a good start (later)
    solution = 0;


    timestep_number = 0;
    time            = 0;
    output_results ();

    double T = 0.5;
    while (time < T)
    {
    		double time_diff=T-time;
    		if (time_diff < time_step)
    		{
    			time += time_diff;
    		}
    		else
    		{
    			time += time_step;
    		}
            ++timestep_number;

            std::cout << "Time step " << timestep_number << " at t=" << time
                      << std::endl;
			assemble_system ();
			// build RHS
			{


			      const FEValuesExtractors::Vector velocities(0);
				  const FEValuesExtractors::Vector displacements(dim+1);
				  std::map<types::global_dof_index, double> boundary_values;

			      unsigned int j=0;


//			      for (unsigned int k=0; k<7; ++k)
//			      {
//			    	  if (k!=j && k!=1 && k!=3 && k!=5 && k!=2)
//			    	  {
//						  VectorTools::interpolate_boundary_values (dof_handler,
//																	k,
//																	ZeroFunction<dim>(dim+1+dim),
//																	constraints,
//																	fe_collection.component_mask(velocities));
//						  VectorTools::interpolate_boundary_values (dof_handler,
//																	k,
//																	ZeroFunction<dim>(dim+1+dim),
//																	constraints,
//																	fe_collection.component_mask(displacements));
//			    	  }
//			      }
//			      VectorTools::interpolate_boundary_values (dof_handler,
//			                                                j,
//			                                                StokesBoundaryValues<dim>(),
//			                                                constraints,
//			                                                fe_collection.component_mask(velocities));
		          VectorTools::interpolate_boundary_values(dof_handler,
		                                                   j,
		                                                   StokesBoundaryValues<dim>(viscosity),
		                                                   boundary_values,
		                                                   fe_collection.component_mask(velocities));
		          MatrixTools::apply_boundary_values(boundary_values,
		                                             system_matrix,
		                                             solution,
		                                             system_rhs);

			//      VectorTools::interpolate_boundary_values (dof_handler,
			//                                                j,
			//                                                ElasticBoundaryValues<dim>(),
			//                                                constraints,
			//                                                fe_collection.component_mask(displacements));
			}

			solve ();
			output_results ();
	        old_solution = solution;
    }
    {
      // compute error at final time T
      //solution=0;
      Vector<double> cellwise_errors (triangulation.n_active_cells());
      QGauss<dim> fe_U(1);
      QGauss<dim> fe_N(1);
      hp::QCollection<dim>  Q_collection;
      Q_collection.push_back (fe_U);
      Q_collection.push_back (fe_N);
      ExactValues<dim> exact_solution;
      exact_solution.set_time(T);

      // something wrong here with this solution
      exact_solution.set_time(0);
      // using solution = 0 before this I get a much smaller norm than comparing
      // the computed solution (which doesn't make sense)
      VectorTools::integrate_difference (dof_handler, solution, exact_solution,
					 cellwise_errors, Q_collection,
					 VectorTools::L2_norm);
      const double l2_error = cellwise_errors.l2_norm();
      std::cout << "dt = " << time_step
		<< " h = " << triangulation.begin_active()->diameter()
		<< " L2(T) error = " << l2_error << std::endl;

    }
    std::cout << "Computation completed." << std::endl;
  }
}




int main ()
{
  try
    {
      using namespace dealii;
      using namespace FSIProblem;

      deallog.depth_console (0);



      FluidStructureProblem<2> flow_problem(1, 1);
      flow_problem.run ();
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
