% Mathematics 2 - Octave 1: Core Problems (Examinable)
clc; clear;

%% Question 1 - Octave as a calculator
disp('--- Q1 ---')
abs(-3)                       % (a) |-3|
log10(100) / log10(10)        % (b)
(1 + tan(1.2)) / 1.2          % (c)
sqrt(3^2 + 4^2)               % (d)
exp(2)                        % (e) e^2
sind(28)                      % (f) sin(28 degrees)
phi = atan(3/4);              % (g) opposite = 3, adjacent = 4
rad2deg(phi)

%% Question 2 - Vectors
disp('--- Q2 ---')
a = [0 -2 -4 -6 -8]           % (a) row vector
b = [1; 3; 5; 7; 9]           %     column vector
a = 0:-2:-8                   % (b) using the : operator
a = linspace(0, -8, 5)        % (c) using linspace

%% Question 3
disp('--- Q3 ---')
c = a(3) + b(end-2:end)       % (a) 3rd element of a + last 3 elements of b
2 * c                         % (b)
d = a .* [5 4 3 2 1]          % (c) element-by-element product

%% Question 4
disp('--- Q4 ---')
output_precision(8);
d .^ 7                        % (a)
e = 7 .^ (1:2:9)              % (b) first 5 odd powers of 7

%% Question 5
disp('--- Q5 ---')
output_precision(5);
angles = 0:pi/6:2*pi          % (a)
f = sin(angles)               % (b)
g = cos(angles)               % (c)
h = f.^2 + g.^2               % (d)
length(h)                     % (e) number of elements
sum(h)                        % (f) sum of elements

%% Question 6 - floor and fix
disp('--- Q6 ---')
x = [2 2.7 -2 -2.7]
floor(x)
fix(x)

%% Question 7 - mod and rem
disp('--- Q7 ---')
x = [5 -5 5 -5]
y = [2 2 -2 -2]
mod(x, y)
rem(x, y)

%% Question 8 - Roots of cubics
disp('--- Q8 ---')
output_precision(8);
roots([1 1 0 -2])                          % (a) z^3 + z^2 - 2 = 0
roots([1, 6+3i, 11+18i, 33i])              % (b)
roots([1, -1+6i, -13-1i, -2-10i])          % (c)

%% Question 9 - Cartesian form of r cis(theta)
disp('--- Q9 ---')
4 * exp(1i * 3*pi/5)                       % (a) 4 cis(3pi/5)
3 * exp(1i * 17*pi/12)                     % (b) 3 cis(17pi/12)

%% Question 10 - Modulus and principal argument (degrees)
disp('--- Q10 ---')
z1 = sqrt(5) + 2i;                         % (a)
abs(z1)
rad2deg(angle(z1))
z2 = -3 + sqrt(7)*1i;                      % (b)
abs(z2)
rad2deg(angle(z2))

%% Question 11 - Fifth roots of 4 - 4i  (z^5 - (4 - 4i) = 0)
disp('--- Q11 ---')
z = roots([1 0 0 0 0 -(4-4i)])
z .^ 5                                     % check: each should equal 4 - 4i
