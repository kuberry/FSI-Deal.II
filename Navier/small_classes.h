#ifndef SMALL_CLASSES_H
#define SMALL_CLASSES_H

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


#endif
