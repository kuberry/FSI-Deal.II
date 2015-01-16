#include "FSI_Project.h"
#include "small_classes.h"
// this function is fast, no need to parallelize

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
	  if (fluid_interface_boundaries.count(cell->face(f)->boundary_indicator())!=0)
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
	  if (structure_interface_boundaries.count(cell->face(f)->boundary_indicator())!=0)
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
	  if (fluid_interface_boundaries.count(cell->face(f)->boundary_indicator())!=0)
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
