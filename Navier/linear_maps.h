#pragma once
#include "FSI_Project.h"

namespace LinearMap {

class Wilkinson
{
 public:
 Wilkinson(const unsigned int size=100):n(size),A(size,size){
    for (unsigned int i=0; i<n; i++) {
      for (unsigned int j=0; j<n; j++) {
	if (i==j) {
	  A.set(i,i,std::fabs(.5*(n-1)-j));
	} else if (j==i+1 || j==i-1) {
	  A.set(i,j,1.0);
	}
      }
    }
  };   

  // Application of matrix to vector src.
  // Write result into dst
  void vmult (Vector<double> &dst,
	      const Vector<double> &src) const
  {
    A.vmult(dst, src);
  };
  // Application of transpose to a vector.
  // Only used by some iterative methods.
  void Tvmult (Vector<double> &dst,
	       const Vector<double> &src) const {AssertThrow(false, ExcNotImplemented())};

 private:
  const unsigned int n;
  FullMatrix<double> A;
};

/* class Vector//: public Vector<double> */
/* { */
/*  public: */
/*   // Resize the current object to have */
/*   // the same size and layout as the model_vector */
/*   // argument provided. The second argument */
/*   // indicates whether to clear the current */
/*   // object after resizing. */
/*   // The second argument must have */
/*   // a default value equal to false */
/*   void reinit (const Vector &model_vector, */
/* 	       const bool leave_elements_uninitialized = false); */
/*   // Inner product between the current object */
/*   // and the argument */
/*   double operator * (const Vector &v) const; */
/*   // Addition of vectors */
/*   void add (const Vector &x); */
/*   // Scaled addition of vectors */
/*   void add (const double a, */
/* 	    const Vector &x); */
/*   // Scaled addition of vectors */
/*   void sadd (const double a, */
/* 	     const double b, */
/* 	     const Vector &x); */
/*   // Scaled assignment of a vector */
/*   void equ (const double a, */
/* 	    const Vector &x); */
/*   // Combined scaled addition of vector x into */
/*   // the current object and subsequent inner */
/*   // product of the current object with v */
/*   double add_and_dot (const double a, */
/* 		      const Vector &x, */
/* 		      const Vector &v); */
/*   // Multiply the elements of the current */
/*   // object by a fixed value */
/*   Vector & operator *= (const double a); */
/*   // Return the l2 norm of the vector */
/*   double l2_norm () const; */
/* }; */

}

