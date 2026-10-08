% Mathematics 2 - Octave 1: Extension Problems (Not Examinable)
clc; clear;

%% X1 - Dot and cross products
disp('--- X1 ---')
V1 = [4, 2, -3];                           % (a)
V2 = [-1, 7, 5];

% (b) Scalar (dot) product - works for any 3D vectors V1, V2
S = V1(1)*V2(1) + V1(2)*V2(2) + V1(3)*V2(3)

% (c) Vector (cross) product - works for any 3D vectors V1, V2
X = [V1(2)*V2(3) - V1(3)*V2(2), ...
     V1(3)*V2(1) - V1(1)*V2(3), ...
     V1(1)*V2(2) - V1(2)*V2(1)]

% (d) Check against Octave's built-in functions
dot(V1, V2)
cross(V1, V2)

% (d) Test with different vectors
V1 = [1, 0, 0];
V2 = [0, 1, 0];
S = V1(1)*V2(1) + V1(2)*V2(2) + V1(3)*V2(3)
X = [V1(2)*V2(3) - V1(3)*V2(2), ...
     V1(3)*V2(1) - V1(1)*V2(3), ...
     V1(1)*V2(2) - V1(2)*V2(1)]

%% X2 - sigma = e^pi and sigma^i
disp('--- X2 ---')
output_precision(8);
sigma = exp(pi)
sigma^1i
% sigma^i = e^(i*pi) = -1. The imaginary part (~1.2e-16) is just
% floating-point rounding error; mathematically it is exactly zero.
