% Mathematics 2 - Octave 1: Extension Problems (Not Examinable)
clc; clear;

%% X1 - Dot and cross products
disp('==================================================')
disp('                   QUESTION X1')
disp('==================================================')

disp(' '); disp('(a) Enter V1 and V2')
V1 = [4, 2, -3]
V2 = [-1, 7, 5]

disp(' '); disp('(b) Scalar (dot) product S')
S = V1(1)*V2(1) + V1(2)*V2(2) + V1(3)*V2(3)

disp(' '); disp('(c) Vector (cross) product X')
X = [V1(2)*V2(3) - V1(3)*V2(2), ...
     V1(3)*V2(1) - V1(1)*V2(3), ...
     V1(1)*V2(2) - V1(2)*V2(1)]

disp(' '); disp('(d) Check against built-in dot and cross')
dot(V1, V2)
cross(V1, V2)

disp(' '); disp('(d) Test with different vectors')
V1 = [1, 0, 0]
V2 = [0, 1, 0]
S = V1(1)*V2(1) + V1(2)*V2(2) + V1(3)*V2(3)
X = [V1(2)*V2(3) - V1(3)*V2(2), ...
     V1(3)*V2(1) - V1(1)*V2(3), ...
     V1(1)*V2(2) - V1(2)*V2(1)]

disp(' ')

%% X2 - sigma = e^pi and sigma^i
disp('==================================================')
disp('                   QUESTION X2')
disp('==================================================')
output_precision(8);

disp(' '); disp('sigma = e^pi')
sigma = exp(pi)

disp(' '); disp('sigma^i')
sigma^1i

% sigma^i = e^(i*pi) = -1. The imaginary part (~1.2e-16) is just
% floating-point rounding error; mathematically it is exactly zero.
disp(' ')
disp('The imaginary part (~1.2e-16) is only rounding error; it is exactly 0.')
