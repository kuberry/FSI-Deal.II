cl1 = 1;

fluid = 2;
structure = .2;
circle = .2;


// Whole Fluid Domain
Point(1) = {0, 0, 0, fluid*.2};
Point(2) = {2.5, 0, 0, fluid};
Point(3) = {2.5, .41, 0, fluid};
Point(4) = {0, .41, 0, fluid*.2};

//center
Point(5) = {.2, .2, 0, circle};
Point(10) = {.15, .2, 0, circle};
//on circle
Point(6) = {.248989795, .21, 0, structure};
Point(7) = {.248989795, .19, 0, structure};

// Elastic
Point(8) = {.6, .19, 0, structure};
Point(9) = {.6, .21, 0, structure};


// lines of the outer box:
Line(1) = {1, 2};
Line(2) = {2, 3};
Line(3) = {3, 4};
Line(4) = {4, 1};

// the first cutout:
Circle(9)  = {6, 5, 10};
Circle(10) = {10, 5, 7};

// lines of elasticity:
Line(5) = {7, 8};
Line(6) = {8, 9};
Line(7) = {9, 6};
Line(8) = {6, 7};


//Circle(6) = {7, 5, 6};
//Circle(5) = {6, 6, 5};
//Circle(6) = {6, 5, 6};

Line Loop(11) = {1, 2, 3, 4, -5, -6, -7, -9, -10};
Line Loop(12) = {5, 6, 7, 8};

// these define the boundary indicators in deal.II:
Physical Line(1) = {1};
Physical Line(2) = {2};
Physical Line(3) = {3};
Physical Line(4) = {4};
Physical Line(5) = {5};
Physical Line(6) = {6};
Physical Line(7) = {7};
//Physical Line(8) = {8};
Physical Line(8) = {9,10};

// you need the physical surface, because that is what deal.II reads in
Plane Surface(9) = {11};
//Plane Surface(10) = {12};
Physical Surface(0) = {9};
//Physical Surface(1) = {10};


// some parameters for the meshing:
Mesh.Algorithm = 8;
Mesh.RecombineAll = 1;
Mesh.CharacteristicLengthFactor = 0.1;
Mesh.SubdivisionAlgorithm = 1;
Mesh.Smoothing = 20;// Show "*";