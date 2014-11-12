#include "data1.h"

// It is okay if not divergence free so long as it is adjust for as a source
// term in the right hand side. This should similarly permit arbitrary non-
// divergence free solutions to be used to test moving domain problems.
// 
// Also note that the divergence source must be negated, since it is in the 
// operator (to provide a symmetric operator for Stokes)

using namespace dealii;

Tensor<2,2> get_Jacobian(double x, double y, double t, bool move_domain) {
  Tensor<2,2> F;
  // x_new = x_old + x*sin(y-t)*.1;
  // y_new = y_old - x*2./3*sin(x-t)*.1;
  F[0][0] = 1 + sin(y-0*t)*1.5;
  F[1][1] = 1;
  if (move_domain)
    {
      F[0][1] = x*cos(y-0*t)*1.5;
      F[1][0] = -2./3*sin(x-0*t)*2.5 -x*2./3*cos(x-0*t)*2.5;
      // F[0][1] = 3*x*t;
      // F[1][0] = -2./3*cos(x-t)*.1;
    }
  return F;
}

Point<2> reference_coord(double x, double y, double t, bool move_domain) {
  Point<2> inverse_coord(x,y);
  // if (move_domain) {
    
  // }
  return inverse_coord;
}

Tensor<2,2> get_DetTimesJacobianInv(Tensor<2,2> Jacobian) {
  Tensor<2,2> Finv;
  Finv[0][0] = Jacobian[1][1];
  Finv[1][1] = Jacobian[0][0];
  Finv[0][1] = -Jacobian[0][1];
  Finv[1][0] = -Jacobian[1][0];
  return Finv;
}

template <int dim>
double FluidStressValues<dim>::value (const Point<dim>  &p,
				      const unsigned int component) const
{
  double answer = 0;
  if (physical_properties.simulation_type==0){
    // This function can be deleted later. It is just for interpolating g on the boundary
    /*
     * u1=cos(t + x)*sin(t + y) + cos(t + y)*sin(t + x);
     * u2=- cos(t + x)*sin(t + y) - cos(t + y)*sin(t + x);
     * p=2*lambda*cos(t + x)*sin(t + y) - 2*viscosity*(cos(t + x)*cos(t + y) - sin(t + x)*sin(t + y));
     * 2*viscosity*(diff(u1,x)*n1+0.5*(diff(u1,y)+diff(u2,x))*n2)-p*n1*u1
     * 2*viscosity*(diff(u2,y)*n2+0.5*(diff(u1,y)+diff(u2,x))*n1)-p*n2*u2
     *
     */
    Tensor<1,dim> result;
    const double t = this->get_time();
    const double x = p[0];
    const double y = p[1];
    const double u1_x = cos(t + x)*cos(t + y) - sin(t + x)*sin(t + y);
    const double u2_y = sin(t + x)*sin(t + y) - cos(t + x)*cos(t + y);
    const double u1_y = cos(t + x)*cos(t + y) - sin(t + x)*sin(t + y);
    const double u2_x = sin(t + x)*sin(t + y) - cos(t + x)*cos(t + y);
    const double pval = 2*physical_properties.mu*cos(t + x)*sin(t + y) - 2*physical_properties.viscosity*(cos(t + x)*cos(t + y)
													  - sin(t + x)*sin(t + y));

    switch (component)
      {
      case 0:
	result[0]=2*physical_properties.viscosity*u1_x-pval;
	result[1]=2*physical_properties.viscosity*0.5*(u1_y+u2_x);
	break;
      case 1:
	result[0]=2*physical_properties.viscosity*0.5*(u1_y+u2_x);
	result[1]=2*physical_properties.viscosity*u2_y-pval;
	break;
      default:
	result=0;
      }
    answer = result[0]*0+result[1]*1;
  } else if (physical_properties.simulation_type==2){
    // This function can be deleted later. It is just for interpolating g on the boundary
    /*
     * u1=2*sin(y-t)+3*x*t;
     * u2=3*sin(x-t)-3*y*t;
     * p=100*x;
     * 2*viscosity*(diff(u1,x)*n1+0.5*(diff(u1,y)+diff(u2,x))*n2)-p*n1*u1
     * 2*viscosity*(diff(u2,y)*n2+0.5*(diff(u1,y)+diff(u2,x))*n1)-p*n2*u2
     *
     */
    AssertThrow( false, ExcInternalError()); // The unit normal would need to be computed
    // Tensor<2,dim> grad_u;
    // Tensor<1,dim> result;
    // const double t = this->get_time();
    // const double x = p[0];
    // const double y = p[1];
    // grad_u[0][0] = 3*t;
    // grad_u[1][1] = -3*t;
    // grad_u[1][0] = 2*cos(y-t);
    // grad_u[0][1] = 3*cos(x-t);
    // Tensor<2,dim> F = get_Jacobian(x, y, t, physical_properties.move_domain);
    // Tensor<2,dim> detTimesFInv = get_DetTimesJacobianInv(F);
    // Tensor<2,dim> FInv = 1./determinant(F)*detTimesFInv;
    // grad_u = .5* (transpose(FInv)*grad_u + transpose(grad_u)*FInv);
    // grad_u *= 2*physical_properties.viscosity;
    // const double pval = 100*x;
    // grad_u[0][0]-=pval;
    // grad_u[1][1]-=pval;

    // switch (component)
    //   {
    //   case 0:
    // 	result[0]=grad_u[0][0];
    // 	result[1]=grad_u[0][1];
    // 	break;
    //   case 1:
    // 	result[0]=grad_u[1][0];
    // 	result[1]=grad_u[1][1];
    // 	break;
    //   default:
    // 	result=0;
    //   }
    // answer = result[0]*0+result[1]*1;
  }
  return answer;
}

template <int dim>
Tensor<1,dim> FluidStressValues<dim>::gradient (const Point<dim>  &p,
						const unsigned int component) const
{
  Tensor<1,dim> result;
  if (physical_properties.simulation_type==0){
    /*
     * u1=cos(t + x)*sin(t + y) + cos(t + y)*sin(t + x);
     * u2=- cos(t + x)*sin(t + y) - cos(t + y)*sin(t + x);
     * p=2*lambda*cos(t + x)*sin(t + y) - 2*viscosity*(cos(t + x)*cos(t + y) - sin(t + x)*sin(t + y));
     * 2*viscosity*(diff(u1,x)*n1+0.5*(diff(u1,y)+diff(u2,x))*n2)-p*n1*u1
     * 2*viscosity*(diff(u2,y)*n2+0.5*(diff(u1,y)+diff(u2,x))*n1)-p*n2*u2
     *
     */
  
    const double t = this->get_time();
    const double x = p[0];
    const double y = p[1];
    const double u1_x = cos(t + x)*cos(t + y) - sin(t + x)*sin(t + y);
    const double u2_y = sin(t + x)*sin(t + y) - cos(t + x)*cos(t + y);
    const double u1_y = cos(t + x)*cos(t + y) - sin(t + x)*sin(t + y);
    const double u2_x = sin(t + x)*sin(t + y) - cos(t + x)*cos(t + y);
    const double pval = 2*physical_properties.mu*cos(t + x)*sin(t + y) - 2*physical_properties.viscosity*(cos(t + x)*cos(t + y)
													  - sin(t + x)*sin(t + y));

    switch (component)
      {
      case 0:
	result[0]=2*physical_properties.viscosity*u1_x-pval;
	result[1]=2*physical_properties.viscosity*0.5*(u1_y+u2_x);
	return result;
      case 1:
	result[0]=2*physical_properties.viscosity*0.5*(u1_y+u2_x);
	result[1]=2*physical_properties.viscosity*u2_y-pval;
	return result;
      default:
	result=0;
	return result;
      }
  } else if (physical_properties.simulation_type==2){
    // This function can be deleted later. It is just for interpolating g on the boundary
    /*
     * u1=2*sin(y-t)+3*x*t;
     * u2=3*sin(x-t)-3*y*t;
     * p=100*x;
     * 2*viscosity*(diff(u1,x)*n1+0.5*(diff(u1,y)+diff(u2,x))*n2)-p*n1*u1
     * 2*viscosity*(diff(u2,y)*n2+0.5*(diff(u1,y)+diff(u2,x))*n1)-p*n2*u2
     *
     */
    Tensor<2,dim> grad_u;
    const double t = this->get_time();
    const double x = p[0];
    const double y = p[1];
    grad_u[0][0] = 3*t;
    grad_u[1][1] = -3*t;
    grad_u[1][0] = 2*cos(y-t);
    grad_u[0][1] = 3*cos(x-t);
    Tensor<2,dim> F = get_Jacobian(x, y, t, physical_properties.move_domain);
    Tensor<2,dim> detTimesFInv = get_DetTimesJacobianInv(F);
    Tensor<2,dim> FInv = 1./determinant(F)*detTimesFInv;
    grad_u = .5* (transpose(FInv)*grad_u + transpose(grad_u)*FInv);
    grad_u *= 2*physical_properties.viscosity;
    const double pval = 0;
    grad_u[0][0] -= pval;
    grad_u[1][1] -= pval;

    switch (component)
      {
      case 0:
	result[0]=grad_u[0][0];
	result[1]=grad_u[0][1];
	break;
      case 1:
	result[0]=grad_u[1][0];
	result[1]=grad_u[1][1];
	break;
      default:
	result=0;
      }
  }
  return result;
}










template <int dim>
double StructureStressValues<dim>::value (const Point<dim>  &p,
					  const unsigned int component) const
{
  Tensor<1,dim> result;
  if (physical_properties.simulation_type==0){
    /*
      >> n1=sin(x + t)*sin(y + t);
      >> n2=cos(x + t)*cos(y + t);
    */
    const double t = this->get_time();
    const double x = p[0];
    const double y = p[1];
    const double n1_x =cos(t + x)*sin(t + y);
    const double n2_y =-cos(t + x)*sin(t + y);
    const double n1_y =cos(t + y)*sin(t + x);
    const double n2_x =-cos(t + y)*sin(t + x);
    switch (component)
      {
      case 0:
	result[0]=2*physical_properties.mu*n1_x+physical_properties.lambda*(n1_x+n2_y);
	result[1]=2*physical_properties.mu*0.5*(n1_y+n2_x);
	break;
      case 1:
	result[0]=2*physical_properties.mu*0.5*(n1_y+n2_x);
	result[1]=2*physical_properties.mu*n2_y+physical_properties.lambda*(n1_x+n2_y);
	break;
      default:
	result=0;
      }
  }
  return result[0]*0+result[1]*(-1);
}

template <int dim>
Tensor<1,dim> StructureStressValues<dim>::gradient (const Point<dim>  &p,
						    const unsigned int component) const
{
  Tensor<1,dim> result;
  if (physical_properties.simulation_type==0){
    /*
      >> n1=sin(x + t)*sin(y + t);
      >> n2=cos(x + t)*cos(y + t);
    */

    const double t = this->get_time();
    const double x = p[0];
    const double y = p[1];
    const double n1_x =cos(t + x)*sin(t + y);
    const double n2_y =-cos(t + x)*sin(t + y);
    const double n1_y =cos(t + y)*sin(t + x);
    const double n2_x =-cos(t + y)*sin(t + x);
    switch (component)
      {
      case 0:
	result[0]=2*physical_properties.mu*n1_x+physical_properties.lambda*(n1_x+n2_y);
	result[1]=2*physical_properties.mu*0.5*(n1_y+n2_x);
	return result;
      case 1:
	result[0]=2*physical_properties.mu*0.5*(n1_y+n2_x);
	result[1]=2*physical_properties.mu*n2_y+physical_properties.lambda*(n1_x+n2_y);
	return result;
      default:
	result=0;
	return result;
      }
  }
  return result;
}











template <int dim>
double FluidRightHandSide<dim>::value (const Point<dim>  &p,
				       const unsigned int component) const
{
  if (physical_properties.simulation_type==0){
    double result;
    const double t = this->get_time();
    const double x = p[0];
    const double y = p[1];
    // >> u1=cos(t + x)*sin(t + y) + cos(t + y)*sin(t + x);
    // >> u2=- cos(t + x)*sin(t + y) - cos(t + y)*sin(t + x);
    // >> p=2*mu*cos(t + x)*sin(t + y) - 2*viscosity*(cos(t + x)*cos(t + y) - sin(t + x)*sin(t + y));
    // >> rho_f*diff(u1,t)-2*viscosity*(diff(diff(u1,x),x)+0.5*(diff(diff(u1,y),y)+diff(diff(u2,x),y)))+diff(p,x) + convection
    // >> rho_f*diff(u2,t)-2*viscosity*(diff(diff(u2,y),y)+0.5*(diff(diff(u2,x),x)+diff(diff(u1,y),x)))+diff(p,y) + convection
    switch (component)
      {
      case 0:
	result = physical_properties.rho_f*(2*cos(t + x)*cos(t + y) - 2*sin(t + x)*sin(t + y)) + 4*physical_properties.viscosity 
	  * (cos(t + x)*sin(t + y) + cos(t + y)*sin(t + x)) - 2*physical_properties.mu*sin(t + x)*sin(t + y);
	// if (physical_properties.navier_stokes)
	//   {
	//     result += physical_properties.rho_f * ((-sin(t + x)*sin(t + y) + cos(t + x)*cos(t + y))
	// 					   *(sin(t + x)*cos(t + y) + sin(t + y)*cos(t + x)) + (sin(t + x)*sin(t + y) - cos(t + x)*cos(t + y))
	// 					   *(-sin(t + x)*cos(t + y) - sin(t + y)*cos(t + x)));
	//   }
	return result;
	//+ physical_properties.rho_f*(2*(cos(t + x)*sin(t + y) + cos(t + y)*sin(t + x))*(cos(t + x)*cos(t + y) - sin(t + x)*sin(t + y)));
      case 1:
	result = 2*physical_properties.mu*cos(t + x)*cos(t + y) - physical_properties.rho_f*(2*cos(t + x)*cos(t + y) - 2*sin(t + x)*sin(t + y));
	// if (physical_properties.navier_stokes)
	//   {
	//     result += physical_properties.rho_f * ((-sin(t + x)*sin(t + y) + cos(t + x)*cos(t + y))
	//   				 *(sin(t + x)*cos(t + y) + sin(t + y)*cos(t + x)) + (sin(t + x)*sin(t + y) - cos(t + x)*cos(t + y))
	//   				 *(-sin(t + x)*cos(t + y) - sin(t + y)*cos(t + x)));
	//   }
	return result;
	//+ physical_properties.rho_f*(2*(cos(t + x)*sin(t + y) + cos(t + y)*sin(t + x))*(cos(t + x)*cos(t + y) - sin(t + x)*sin(t + y)));
      case 2:
	return 0;
      default:
	return 0;
      }
  } else if (physical_properties.simulation_type==2){
    /*
     * u1=2*sin(y-t)+3*x*t;
     * u2=3*sin(x-t)-3*y*t;
     * p=100*x;
     */
    // >> rho_f*diff(u1,t)-2*viscosity*(diff(diff(u1,x),x)+0.5*(diff(diff(u1,y),y)+diff(diff(u2,x),y)))+diff(p,x) + convection
    // >> rho_f*diff(u2,t)-2*viscosity*(diff(diff(u2,y),y)+0.5*(diff(diff(u2,x),x)+diff(diff(u1,y),x)))+diff(p,y) + convection
    const double t = this->get_time();
    const double x = p[0];
    const double y = p[1];
    Tensor<1,dim> result(2);
    Tensor<1,dim> u(2);
    u[0] = 2*sin(y-t)+3*x*t;
    u[1] = 3*sin(x-t)-3*y*t;
    Tensor<1,dim> z(2);
    Tensor<2,dim> grad_z;
    if (physical_properties.moving_domain)
      {
	z[0] = 0*(-cos(y-t)*1.5); 
	z[1] = 0*(2./3*cos(x-t)*2.5);
	// z[0] = 3*x; 
	// z[1] = -5*x*y;
	grad_z[0][0]=0;
	grad_z[1][0]=0*sin(y-t)*1.5;
	grad_z[0][1]=0*(-2./3*cos(x-t)*2.5);
        grad_z[1][1]=0;
      } 
    Tensor<2,dim> determinant_derivatives;
    // x/x_1 = 1
    // y/x_2 = 1
    // x/x_2 = cos(y-t)*.1
    // y/x_1 = -2./3*cos(x-t)*.1
    if (physical_properties.move_domain)
      {
	// these powers denote partial derivatives
	// x^2/x_1^2 * y/x_2 - y^2/x_1^2 * x/x_1
	determinant_derivatives[0][0]= 0;// 0*0 - (2./3*(sin(x-t)*.1)*0; 
	// x/x_1 * y^2/x_2,x_1 - y/x_1 * x^2/x_1^2
	determinant_derivatives[0][1]= 0;//0*0 - (-2./3*cos(x-t)*.1 * 0;
	// x^2/x_1,x_2 * y/x_2 - y^2/x_1,x_2 * x/x_1
	determinant_derivatives[1][0]= 0;//0*0 - 0*0; 
	// x/x_1 * y^2/x_2^2 - y/x_1 * x^2/x_1,x_2
	determinant_derivatives[1][1]= 0;//0*0 - 0*0;
      } 

    Tensor<1,dim> grad_p(2);
    grad_p[0] = 0;//100; // pval
    grad_p[1] = 0;//-40;
    Tensor<1,dim> u_t;
    u_t[0] = -2*cos(y-t)+3*x;
    u_t[1] = -3*cos(x-t)-3*y;
    Tensor<2,dim> grad_u;
    // grad_u[p][c]=[partial][component]
    // u1_x
    grad_u[0][0] = 3*t;
    // u2_y
    grad_u[1][1] = -3*t;
    // u1_y
    grad_u[1][0] = 2*cos(y-t);
    // u2_x
    grad_u[0][1] = 3*cos(x-t);

    Tensor<2,dim> F = get_Jacobian(x, y, t, physical_properties.move_domain);
    Tensor<2,dim> detTimesFInv = get_DetTimesJacobianInv(F);
    Tensor<2,dim> FInv = 1./determinant(F)*detTimesFInv;

    Tensor<2,dim> grad_u_transformed_times_TransposeFInv = .5*(transpose(FInv)*grad_u + transpose(grad_u)*FInv)*transpose(FInv);

    Tensor<1,dim> div_deformation(2);
    Tensor<3,dim> second_partial_u;

    // grad_u[p1][p2][c]=[partial1][partial2][component]
    // u1_xx
    second_partial_u[0][0][0] = 0;
    // u2_yx
    second_partial_u[1][1][0] = -2*sin(y-t);
    // u1_yx
    second_partial_u[1][0][0] = 0;
    // u2_xx
    second_partial_u[0][1][0] = 0;
    // u1_xy
    second_partial_u[0][0][1] = -3*sin(x-t);
    // u2_yy
    second_partial_u[1][1][1] = 0;
    // u1_yy
    second_partial_u[1][0][1] = 0;
    // u
    second_partial_u[0][1][1] = 0;
    
    Tensor<2,dim> second_partial_u1;
    for (unsigned int i=0; i<2; ++i)
      for (unsigned int j=0; j<2; ++j)
	second_partial_u1[i][j]=second_partial_u[i][j][0];
    Tensor<2,dim> second_partial_u2;
    for (unsigned int i=0; i<2; ++i)
      for (unsigned int j=0; j<2; ++j)
	second_partial_u2[i][j]=second_partial_u[i][j][1];

    // product rule of jacobians * transformed gradient
    div_deformation[0]+=(determinant_derivatives[0][0]+determinant_derivatives[0][1])*grad_u_transformed_times_TransposeFInv[0][0];
    div_deformation[1]+=(determinant_derivatives[0][0]+determinant_derivatives[0][1])*grad_u_transformed_times_TransposeFInv[0][1];
    div_deformation[0]+=(determinant_derivatives[1][0]+determinant_derivatives[1][1])*grad_u_transformed_times_TransposeFInv[1][0];
    div_deformation[1]+=(determinant_derivatives[1][0]+determinant_derivatives[1][1])*grad_u_transformed_times_TransposeFInv[1][1];

    Tensor<1,dim> second_partials_transformed(2);
    
    AssertThrow(div_deformation[0]+div_deformation[1]==0, ExcInternalError()); /* just checking it is zero, since it should be zero */
    // jacobian * derivatives of transformed gradient
    // for (unsigned int i=0; i<dim; ++i)
    //   for (unsigned int j=0; j<dim; ++j) {
    // 	  div_deformation[0]+=second_partial_u[i][j][0]*(FInv[i][0]*FInv[j][0]+.5*FInv[i][1]*FInv[j][1]) + .5*second_partial_u[i][j][1]*FInv[i][0]*FInv[j][1];
    // 	  div_deformation[1]+=second_partial_u[i][j][1]*(FInv[i][1]*FInv[j][1]+.5*FInv[i][0]*FInv[j][0]) + .5*second_partial_u[i][j][0]*FInv[i][1]*FInv[j][0];
    //   }

    //   // 	{
    //   {
	// second_partials_transformed[0] =second_partial_u[0][0][0]*FInv[0][0]+second_partial_u[1][0][0]*FInv[1][0];
	// second_partials_transformed[0]+=.5*(second_partial_u[0][1][0]*FInv[0][1]+second_partial_u[1][1][0]*FInv[1][1]
	// 				   +second_partial_u[0][1][1]*FInv[0][0]+second_partial_u[1][1][1]*FInv[1][0]);
	// second_partials_transformed[1] =second_partial_u[1][1][1]*FInv[1][1]+second_partial_u[0][1][1]*FInv[0][1];
	// second_partials_transformed[1]+=.5*(second_partial_u[0][0][0]*FInv[0][1]+second_partial_u[1][0][0]*FInv[1][1]
	// 				   +second_partial_u[0][0][1]*FInv[0][0]+second_partial_u[1][0][1]*FInv[1][0]);
	second_partials_transformed[0] =second_partial_u[0][0][0]*FInv[0][0]+second_partial_u[1][0][0]*FInv[1][0];
	//second_partials_transformed[0]+=.5*(second_partial_u[0][1][0]*FInv[0][1]+second_partial_u[1][1][0]*FInv[1][1]
	//+second_partial_u[0][1][1]*FInv[0][0]+second_partial_u[1][1][1]*FInv[1][0]);
	second_partials_transformed[1] =second_partial_u[1][1][1]*FInv[1][1]+second_partial_u[0][1][1]*FInv[0][1];
	//second_partials_transformed[1]+=.5*(second_partial_u[0][0][0]*FInv[0][1]+second_partial_u[1][0][0]*FInv[1][1]
	//+second_partial_u[0][0][1]*FInv[0][0]+second_partial_u[1][0][1]*FInv[1][0]);

	second_partials_transformed=determinant(F)*second_partials_transformed*transpose(FInv);

	// APPROACH BY CHAIN RULE
	second_partials_transformed *= 0;
	//FInv = transpose(FInv);
	for (unsigned int i=0; i<dim; ++i)
	  for (unsigned int j=0; j<dim; ++j)
	    for (unsigned int k=0; k<dim; ++k) 
	      for (unsigned int l=0; l<dim; ++l)
		for (unsigned int m=0; m<dim; ++m) {
		  if (i==j && j==k) {
		    second_partials_transformed[i] += second_partial_u[l][m][i] * FInv[m][k] * FInv[l][j];  
		  } else if (j==k) {
		    second_partials_transformed[i] += .5 * second_partial_u[l][m][i] * FInv[m][k] * FInv[l][j];
		  } else {
		    second_partials_transformed[(i+1)%2] += .5 * second_partial_u[l][m][i] * FInv[m][k] * FInv[l][j];
		  }
		}
	//FInv = transpose(FInv);

 	// // APPROACH BY PIOLA TRANSFORM
	// second_partials_transformed *= 0;
	// Tensor<1,dim> temp_holder(2);
	// temp_holder[0] = scalar_product(second_partial_u1,FInv);
	// temp_holder[1] = scalar_product(second_partial_u2,FInv);
	// Tensor<1,dim> other_holder(2);
	// other_holder[0] = second_partial_u1[0][0]+second_partial_u2[0][1];
	// other_holder[0] = second_partial_u1[1][0]+second_partial_u2[1][1];
	// second_partials_transformed = FInv*(temp_holder+transpose(FInv)*other_holder);



	//div_deformation = transpose(FInv) * second_partials_transformed;
	div_deformation = second_partials_transformed;
	// div_deformation=div_deformation+second_partials_transformed;
	  // div_deformation[0]+=second_partial_u[i][j][0]*(FInv[i][0]*FInv[j][0]+.5*FInv[i][1]*FInv[j][1]) + .5*second_partial_u[i][j][1]*FInv[i][0]*FInv[j][1];
	  // div_deformation[1]+=second_partial_u[i][j][1]*(FInv[i][1]*FInv[j][1]+.5*FInv[i][0]*FInv[j][0]) + .5*second_partial_u[i][j][0]*FInv[i][1]*FInv[j][0];
      // }
	// }

    result += physical_properties.rho_f*u_t; // time term
    if (physical_properties.navier_stokes)
      result += physical_properties.rho_f*u*(FInv*grad_u); // convection term
    //if (physical_properties.moving_domain && !physical_properties.move_domain) {
    //result -= physical_properties.rho_f*z*(transpose(FInv)*grad_u); // z grad u term, best with transpose
    //result -= physical_properties.rho_f*scalar_product(grad_z,FInv)*u; // (div z)u term
      //}
    result -= 2*physical_properties.viscosity*div_deformation; // diffusion term
    result += transpose(FInv)*grad_p; 

    switch (component)
      {
      case 0:
	return determinant(F)*result[0];//determinant(F)* - 2*physical_properties.viscosity*sin(t - y);
      case 1:
	return determinant(F)*result[1];//determinant(F)* - 3*physical_properties.viscosity*sin(t - x);
      case 2:
        //return -determinant(F)*scalar_product(grad_u,FInv);//- 1e-11*100*x;
	// THERE WILL ALWAYS BE A PROBLEM HERE IF THE RHS IS INTENDED TO BE EVALUATED AT T^{n+1/2} AND THE DIVERGENCE IS BEING EVALUATED AT T^{n+1}
	return -1.0*determinant(F)*scalar_product(grad_u,FInv);
	//return -determinant(F)*trace(FInv*transpose(grad_u));// - 1e-8*100*x;
      default:
	return 0;
      }
  }
  return 0;
}













template <int dim>
double StructureRightHandSide<dim>::value (const Point<dim>  &p,
					   const unsigned int component) const
{
  if (physical_properties.simulation_type==0){
    /*
      >> n1=sin(x + t)*sin(y + t);
      >> n2=cos(x + t)*cos(y + t);
      >> rho_s*diff(diff(n1,t),t)-2*mu*(diff(diff(n1,x),x)+0.5*(diff(diff(n1,y),y)+diff(diff(n2,x),y)))-lambda*(diff(diff(n1,x),x)+diff(diff(n2,y),x))
      >> rho_s*diff(diff(n2,t),t)-2*mu*(diff(diff(n2,y),y)+0.5*(diff(diff(n2,x),x)+diff(diff(n1,y),x)))-lambda*(diff(diff(n1,x),y)+diff(diff(n2,y),y))
    */
    const double t = this->get_time();
    const double x = p[0];
    const double y = p[1];
    switch (component)
      {
      case 0:
	return physical_properties.rho_s*(2*cos(t + x)*cos(t + y) - 2*sin(t + x)*sin(t + y)) + 2*physical_properties.mu*sin(t + x)*sin(t + y);
      case 1:
	return 2*physical_properties.mu*cos(t + x)*cos(t + y) - physical_properties.rho_s*(2*cos(t + x)*cos(t + y) - 2*sin(t + x)*sin(t + y));
      default:
	return 0;
      }
  }
  return 0;
}














template <int dim>
double FluidBoundaryValues<dim>::value (const dealii::Point<dim> &p,
					const unsigned int component) const
{
  if (physical_properties.simulation_type==0){
    Assert (component < 3, ExcInternalError());
    const double t = this->get_time();
    const double x = p[0];
    const double y = p[1];
    switch (component)
      {
      case 0:
	return cos(x + t)*sin(y + t) + sin(x + t)*cos(y + t);
      case 1:
	return -sin(x + t)*cos(y + t) - cos(x + t)*sin(y + t);
      case 2:
	return 2*physical_properties.viscosity*(sin(x + t)*sin(y + t) - cos(x + t)*cos(y + t)) + 2*physical_properties.mu*cos(x + t)*sin(y + t);
      default:
	return 0;
      }
  } else if (physical_properties.simulation_type==1) {
    Assert (component < 3, ExcInternalError());
    double pi =  3.14159265358979323846;
    const double t = this->get_time();
    // const double x = p[0];
    // const double y = p[1];
    switch (component)
      {
      case 0:
	if (t<.025) return 1000 * (1 - cos(2*pi*t*40));//-1000t)*sin(y + t) + sin(x + t)*cos(y + t);
        else return 0;
      case 1:
	return 0;
      case 2:
	return 0;
      default:
	return 0;
      }    
  } else if (physical_properties.simulation_type==2) {
    Assert (component < 3, ExcInternalError());
    /*
     * u1=2*sin(y-t)+3*x*t;
     * u2=3*sin(x-t)-3*y*t;
     * p=100*x;
     */
    const double t = this->get_time();
    const double x = p[0];
    const double y = p[1];
    switch (component)
      {
      case 0:
	return 2*sin(y-t)+3*x*t;
      case 1:
	return 3*sin(x-t)-3*y*t;
      case 2:
	return 0; // pval
      default:
	return 0;
      }    
  }
  return 0;
}

template <int dim>
void
FluidBoundaryValues<dim>::vector_value (const dealii::Point<dim> &p,
					Vector<double>   &values) const
{
  for (unsigned int c=0; c<this->n_components; ++c)
    {
      values(c) = FluidBoundaryValues<dim>::value (p, c);
    }
}

template <int dim>
Tensor<1,dim> FluidBoundaryValues<dim>::gradient (const dealii::Point<dim>   &p,
						  const unsigned int component) const
{
  Tensor<1,dim> result;
  if (physical_properties.simulation_type==0){
    const double t = this->get_time();
    const double x = p[0];
    const double y = p[1];
    const double u1_x = cos(t + x)*cos(t + y) - sin(t + x)*sin(t + y);
    const double u2_y = sin(t + x)*sin(t + y) - cos(t + x)*cos(t + y);
    const double u1_y = cos(t + x)*cos(t + y) - sin(t + x)*sin(t + y);
    const double u2_x = sin(t + x)*sin(t + y) - cos(t + x)*cos(t + y);

    switch (component)
      {
      case 0:
	result[0]=u1_x;
	result[1]=u1_y;
	return result;
      case 1:
	result[0]=u2_x;
	result[1]=u2_y;
	return result;
      default:
	result=0;
	return result;
      }
  } else if (physical_properties.simulation_type==2){
    Tensor<2,dim> grad_u;
    const double t = this->get_time();
    const double x = p[0];
    const double y = p[1];
    grad_u[0][0] = 3*t;
    grad_u[1][1] = -3*t;
    grad_u[1][0] = 2*cos(y-t);
    grad_u[0][1] = 3*cos(x-t);
    Tensor<2,dim> F = get_Jacobian(x, y, t, physical_properties.move_domain);
    Tensor<2,dim> detTimesFInv = get_DetTimesJacobianInv(F);
    Tensor<2,dim> FInv = 1./determinant(F)*detTimesFInv;
    //grad_u = transpose(FInv)*grad_u;
    //grad_u = FInv*grad_u;
    // grad_u = transpose(grad_u);
    // grad_u *= determinant(F); The domain has returned to the reference domain when the error calculation that uses this is called

    switch (component)
      {
      case 0:
	result[0]=grad_u[0][0];
	result[1]=grad_u[1][0]; 
	// CHECK THIS LATER, it may be the cause of some issues with transposes. I replaced what you see with their equivalents grad_u[0][1]=u2_x
	// Basically, Deal.II expects the transpose of the gradient instead of the gradient
	return result;
      case 1:
	result[0]=grad_u[0][1];
	result[1]=grad_u[1][1];
	return result;
      default:
	result=0;
	return result;
      }
  }
  return result;
}
















template <int dim>
double StructureBoundaryValues<dim>::value (const Point<dim> &p,
					    const unsigned int component) const
{
  if (physical_properties.simulation_type==0){
    /*
      >> n1=sin(x + t)*sin(y + t);
      >> n2=cos(x + t)*cos(y + t);
    */
    Assert (component < 4, ExcInternalError());
    const double t = this->get_time();
    const double x = p[0];
    const double y = p[1];
    switch (component)
      {
      case 0:
	return sin(x + t)*sin(y + t);
      case 1:
	return cos(x + t)*cos(y + t);
      case 2:
	return sin(x + t)*cos(y + t)+cos(x + t)*sin(y + t);
      case 3:
	return -sin(x + t)*cos(y + t)-cos(x + t)*sin(y + t);
      default:
	Assert(false,ExcDimensionMismatch(5,4));
	return 0;
      }
  } else if (physical_properties.simulation_type==2) {
    return 1; // just a placeholder since log(0) causes errors for python script
  }
  return 0;
}

template <int dim>
void StructureBoundaryValues<dim>::vector_value (const Point<dim> &p,
						 Vector<double>   &values) const
{
  for (unsigned int c=0; c<this->n_components; ++c)
    {
      values(c) = StructureBoundaryValues<dim>::value (p, c);
    }
}

template <int dim>
Tensor<1,dim> StructureBoundaryValues<dim>::gradient (const Point<dim>   &p,
						      const unsigned int component) const
{
  Tensor<1,dim> result;
  if (physical_properties.simulation_type==0){
    /*
      >> n1=sin(x + t)*sin(y + t);
      >> n2=cos(x + t)*cos(y + t);
    */

    const double t = this->get_time();
    const double x = p[0];
    const double y = p[1];
    const double n1_x =cos(t + x)*sin(t + y);
    const double n2_y =-cos(t + x)*sin(t + y);
    const double n1_y =cos(t + y)*sin(t + x);
    const double n2_x =-cos(t + y)*sin(t + x);

    switch (component)
      {
      case 0:
	result[0]=n1_x;
	result[1]=n1_y;
	return result;
      case 1:
	result[0]=n2_x;
	result[1]=n2_y;
	return result;
      default:
	result=0;
	return result;//
      }
  } else if (physical_properties.simulation_type==2) {
    result[0]=1;result[1]=1; // just a placeholder since log(0) causes errors for python script
  }
  return result;
}













template <int dim>
double AleBoundaryValues<dim>::value (const Point<dim> &p,
				      const unsigned int component) const
{
  if (physical_properties.simulation_type!=2){
    Assert (component < dim, ExcInternalError());
    if (component==0)
      {
	return 0;
      }
    else
      {
	return 0;
      }
  } else {
    Assert (component < dim, ExcInternalError());
    const double t = this->get_time();
    const double x = p[0];
    const double y = p[1];

    // x_new = x_old + x*sin(y-t)*.1;
    // y_new = 2*y_old - .2./3*sin(x-t)*.1;

    if (component==0)
      {
	return x*sin(y-0*t)*1.5;
	// return 3*x*t+2*y*pow(t,2);
      }
    else
      {
	return -x*2./3*sin(x-0*t)*2.5;
	// return -5*x*y*t+4*x*pow(t,3);
      }
  }
  // x_new = x_old + sin(y-t)*.1;
  // y_new = y_old - .2./3*sin(x-t)*.1;
  return 0;
}

template <int dim>
void AleBoundaryValues<dim>::vector_value (const Point<dim> &p,
					   Vector<double>   &values) const
{
  for (unsigned int c=0; c<this->n_components; ++c)
    {
      values(c) = AleBoundaryValues<dim>::value (p, c);
    }
}








// template declarations
template class FluidStressValues<2>;
template class StructureStressValues<2>;
template class FluidRightHandSide<2>;
template class StructureRightHandSide<2>;
template class FluidBoundaryValues<2>;
template class StructureBoundaryValues<2>;
template class AleBoundaryValues<2>;

