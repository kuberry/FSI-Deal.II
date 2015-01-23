#include "FSI_Project.h"
#include "small_classes.h"
// this function is fast, no need to parallelize


// currently it checks whether face boundary id is zero to determine if it is an interface

template <int dim>
void FSIProblem<dim>::build_dof_mapping()
{
  std::set<Info<dim> > f_a;
  std::set<Info<dim> > n_a;
  std::set<Info<dim> > v_a;
  std::set<Info<dim> > a_a;
  std::set<Info<dim> > f_all;
  std::set<Info<dim> > a_all;
  {
    typename DoFHandler<dim>::active_cell_iterator
      cell = fluid_dof_handler.begin_active(),
      endc = fluid_dof_handler.end();
    for (; cell!=endc; ++cell)
      {
	{
	  std::vector<unsigned int> temp(cell->get_fe().dofs_per_cell);
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
		  f_all.insert(Info<dim>(temp[i],temp2[i],fluid_fe.system_to_component_index(i).first));
		}
	    }
	}
	for (unsigned int f=0; f<GeometryInfo<dim>::faces_per_cell; ++f)
	  if (fluid_interface_boundaries.count(cell->face(f)->boundary_indicator())!=0)
	    {
	      
	      std::vector<unsigned int> temp(4*cell->get_fe()[0].dofs_per_vertex + 4*cell->get_fe()[0].dofs_per_line + cell->get_fe()[0].dofs_per_quad);
	      cell->face(f)->get_dof_indices(temp);

	      Quadrature<dim-1> q(fluid_fe.get_unit_face_support_points());
	      FEFaceValues<dim> fe_face_values (fluid_fe, q,
						update_quadrature_points);
	      fe_face_values.reinit (cell, f);
	      std::vector<Point<dim> > temp2(q.size());
	      temp2=fe_face_values.get_quadrature_points();
	      for (unsigned int i=0;i<temp2.size();++i)
		{
		  if (fluid_fe.face_system_to_component_index(i).first<dim)
		    {
		      f_a.insert(Info<dim>(temp[i],temp2[i],fluid_fe.face_system_to_component_index(i).first));
		    }
		}
	    }
      }
  }
  {
    typename DoFHandler<dim>::active_cell_iterator
      cell = structure_dof_handler.begin_active(),
      endc = structure_dof_handler.end();
    for (; cell!=endc; ++cell)
      {
	for (unsigned int f=0; f<GeometryInfo<dim>::faces_per_cell; ++f)
	  if (structure_interface_boundaries.count(cell->face(f)->boundary_indicator())!=0)
	    {
	      std::vector<unsigned int> temp(4*cell->get_fe()[0].dofs_per_vertex + 4*cell->get_fe()[0].dofs_per_line + cell->get_fe()[0].dofs_per_quad);
	      cell->face(f)->get_dof_indices(temp);
	      Quadrature<dim-1> q(structure_fe.get_unit_face_support_points());
	      FEFaceValues<dim> fe_face_values (structure_fe, q,
						update_quadrature_points);
	      fe_face_values.reinit (cell, f);
	      std::vector<Point<dim> > temp2(q.size());
	      temp2=fe_face_values.get_quadrature_points();
	      for (unsigned int i=0;i<temp2.size();++i)
		{
		  if (structure_fe.face_system_to_component_index(i).first<dim) // this chooses displacement entries
		    {
		      n_a.insert(Info<dim>(temp[i],temp2[i],structure_fe.face_system_to_component_index(i).first));
		    }
		  else
		    {
		      v_a.insert(Info<dim>(temp[i],temp2[i],structure_fe.face_system_to_component_index(i).first));
		    }
		}
	    }
      }
  }
  {
    typename DoFHandler<dim>::active_cell_iterator
      cell = ale_dof_handler.begin_active(),
      endc = ale_dof_handler.end();
    for (; cell!=endc; ++cell)
      {
	{
	  std::vector<unsigned int> temp(cell->get_fe()[0].dofs_per_cell);
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
		  a_all.insert(Info<dim>(temp[i],temp2[i],ale_fe.system_to_component_index(i).first));
		}
	    }
	}
	for (unsigned int f=0; f<GeometryInfo<dim>::faces_per_cell; ++f)
	  if (fluid_interface_boundaries.count(cell->face(f)->boundary_indicator())!=0)
	    {
	      std::vector<unsigned int> temp(4*cell->get_fe()[0].dofs_per_vertex + 4*cell->get_fe()[0].dofs_per_line + cell->get_fe()[0].dofs_per_quad);
	      cell->face(f)->get_dof_indices(temp);
	      Quadrature<dim-1> q(ale_fe.get_unit_face_support_points());
	      FEFaceValues<dim> fe_face_values (ale_fe, q,
						update_quadrature_points);
	      fe_face_values.reinit (cell, f);
	      std::vector<Point<dim> > temp2(q.size());
	      temp2=fe_face_values.get_quadrature_points();
	      for (unsigned int i=0;i<temp2.size();++i)
		{
		  if (ale_fe.face_system_to_component_index(i).first<dim)
		    {
		      a_a.insert(Info<dim>(temp[i],temp2[i],ale_fe.face_system_to_component_index(i).first));
		    }
		}
	    }
      }
  }

  typename std::set<Info<dim> >::iterator  n = n_a.begin();
  typename std::set<Info<dim> >::iterator  v = v_a.begin();
  typename std::set<Info<dim> >::iterator  f = f_a.begin();
  typename std::set<Info<dim> >::iterator  a = a_a.begin();

  AssertThrow(f_a.size() == n_a.size(), ExcDimensionMismatch(f_a.size(), n_a.size()));
  AssertThrow(v_a.size() == n_a.size(), ExcDimensionMismatch(f_a.size(), n_a.size()));
  AssertThrow(f_a.size() == a_a.size(), ExcDimensionMismatch(f_a.size(), n_a.size()));
  for (unsigned int i=0; i<f_a.size(); ++i)
    {
      f2n.insert(std::pair<unsigned int,unsigned int>(f->dof,n->dof));
      n2f.insert(std::pair<unsigned int,unsigned int>(n->dof,f->dof));
      f2v.insert(std::pair<unsigned int,unsigned int>(f->dof,v->dof));
      v2f.insert(std::pair<unsigned int,unsigned int>(v->dof,f->dof));
      n2a.insert(std::pair<unsigned int,unsigned int>(n->dof,a->dof));
      a2n.insert(std::pair<unsigned int,unsigned int>(a->dof,n->dof));
      v2a.insert(std::pair<unsigned int,unsigned int>(v->dof,a->dof));
      a2v.insert(std::pair<unsigned int,unsigned int>(a->dof,v->dof));
      a2f.insert(std::pair<unsigned int,unsigned int>(a->dof,f->dof));
      f2a.insert(std::pair<unsigned int,unsigned int>(f->dof,a->dof));
      v2n.insert(std::pair<unsigned int,unsigned int>(v->dof,n->dof));
      n2v.insert(std::pair<unsigned int,unsigned int>(n->dof,v->dof));
      n++;
      v++;
      f++;
      a++;
    }

  AssertThrow(f_all.size() == a_all.size(), ExcDimensionMismatch(f_a.size(), n_a.size()));
  f = f_all.begin();
  a = a_all.begin();
  for (unsigned int i=0; i<f_all.size(); ++i)
    {
      a2f_all.insert(std::pair<unsigned int,unsigned int>(a->dof,f->dof));
      f2a_all.insert(std::pair<unsigned int,unsigned int>(f->dof,a->dof));
      f++;
      a++;
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
void FSIProblem<dim>::transfer_interface_dofs(const BlockVector<double> & solution_1, BlockVector<double> & solution_2, unsigned int from, unsigned int to, StructureComponent structure_var_1, StructureComponent structure_var_2)
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
      if ((from==0 || from==2) || (from==1 && (structure_var_1==structure_var_2 || structure_var_2== NotSet)))
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

// THIS FUNCTION SHOULD BE COMBINED WITH THE PREVIOUS FUNCTION AT SOME POINT (in a more elegant way)
template <int dim>
void FSIProblem<dim>::vector_vector_transfer_interface_dofs(const Vector<double> & solution_1, Vector<double> & solution_2, unsigned int from, unsigned int to, StructureComponent structure_var_1, StructureComponent structure_var_2)
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
	  solution_2[it->second]=solution_1[it->first];
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
      if ((from==0 || from==2) || (from==1 && (structure_var_1==structure_var_2 || structure_var_2== NotSet)))
	{ 
	  for  (std::map<unsigned int, unsigned int>::iterator it=mapping.begin(); it!=mapping.end(); ++it)
	    {
	      solution_2[it->first]=solution_1[it->first];
	    }
	}
      else
	{
	  for  (std::map<unsigned int, unsigned int>::iterator it=mapping.begin(); it!=mapping.end(); ++it)
	    {
	      solution_2[it->second]=solution_1[it->first];
	    }
	}
    }
}

template void FSIProblem<2>::build_dof_mapping();
template void FSIProblem<2>::transfer_all_dofs(BlockVector<double> & solution_1, BlockVector<double> & solution_2, unsigned int from, unsigned int to);
template void FSIProblem<2>::transfer_interface_dofs(const BlockVector<double> & solution_1, BlockVector<double> & solution_2, unsigned int from, unsigned int to, StructureComponent structure_var_1, StructureComponent structure_var_2);
template void FSIProblem<2>::vector_vector_transfer_interface_dofs(const Vector<double> & solution_1, Vector<double> & solution_2, unsigned int from, unsigned int to, StructureComponent structure_var_1, StructureComponent structure_var_2);

template void FSIProblem<3>::build_dof_mapping();
template void FSIProblem<3>::transfer_all_dofs(BlockVector<double> & solution_1, BlockVector<double> & solution_2, unsigned int from, unsigned int to);
template void FSIProblem<3>::transfer_interface_dofs(const BlockVector<double> & solution_1, BlockVector<double> & solution_2, unsigned int from, unsigned int to, StructureComponent structure_var_1, StructureComponent structure_var_2);
template void FSIProblem<3>::vector_vector_transfer_interface_dofs(const Vector<double> & solution_1, Vector<double> & solution_2, unsigned int from, unsigned int to, StructureComponent structure_var_1, StructureComponent structure_var_2);
