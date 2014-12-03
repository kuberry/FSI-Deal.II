cl1 = 1;

fluid = 2;
structure = .2;
circle = .2;

Point(6) = {.248989795, .21, 0, structure};
Point(7) = {.248989795, .19, 0, structure};

// Elastic
Point(8) = {.6, .19, 0, structure};
Point(9) = {.6, .21, 0, structure};

// lines of elasticity:
Line(1) = {7, 8};
Line(2) = {8, 9};
Line(3) = {9, 6};
Line(4) = {6, 7};

Line Loop(11) = {1, 2, 3, 4};

// these define the boundary indicators in deal.II:
Physical Line(1) = {1};
Physical Line(2) = {2};
Physical Line(3) = {3};
Physical Line(4) = {4};

// you need the physical surface, because that is what deal.II reads in
Plane Surface(9) = {11};
Physical Surface(0) = {9};

// some parameters for the meshing:
Mesh.Algorithm = 8;
Mesh.RecombineAll = 1;
Mesh.CharacteristicLengthFactor = 0.1;
Mesh.SubdivisionAlgorithm = 1;
Mesh.Smoothing = 20;// Show "*";