% Mathematics 2 - Octave 1: Core Problems (Examinable)
clc; clear;

%% Question 1 - Octave as a calculator
disp('==================================================')
disp('                   QUESTION 1')
disp('==================================================')

disp(' '); disp('(a) |-3|')
abs(-3)

disp(' '); disp('(b) log10(100) / log10(10)')
log10(100) / log10(10)

disp(' '); disp('(c) (1 + tan(1.2)) / 1.2')
(1 + tan(1.2)) / 1.2

disp(' '); disp('(d) sqrt(3^2 + 4^2)')
sqrt(3^2 + 4^2)

disp(' '); disp('(e) e^2')
exp(2)

disp(' '); disp('(f) sin(28 degrees)')
sind(28)

disp(' '); disp('(g) angle phi in degrees (opposite = 3, adjacent = 4)')
phi = atan(3/4);
rad2deg(phi)

disp(' ')

%% Question 2 - Vectors
disp('==================================================')
disp('                   QUESTION 2')
disp('==================================================')

disp(' '); disp('(a) Store a and b')
a = [0 -2 -4 -6 -8]
b = [1; 3; 5; 7; 9]

disp(' '); disp('(b) a using the : operator')
a = 0:-2:-8

disp(' '); disp('(c) a using linspace')
a = linspace(0, -8, 5)

disp(' ')

%% Question 3
disp('==================================================')
disp('                   QUESTION 3')
disp('==================================================')

disp(' '); disp('(a) 3rd element of a + last 3 elements of b')
c = a(3) + b(end-2:end)

disp(' '); disp('(b) Each element of c multiplied by 2')
2 * c

disp(' '); disp('(c) a .* [5 4 3 2 1]')
d = a .* [5 4 3 2 1]

disp(' ')

%% Question 4
disp('==================================================')
disp('                   QUESTION 4')
disp('==================================================')
output_precision(8);

disp(' '); disp('(a) Each element of d to the power 7')
d .^ 7

disp(' '); disp('(b) First 5 odd powers of 7')
e = 7 .^ (1:2:9)

disp(' ')

%% Question 5
disp('==================================================')
disp('                   QUESTION 5')
disp('==================================================')
output_precision(5);

disp(' '); disp('(a) Angles from 0 to 2*pi in steps of pi/6')
angles = 0:pi/6:2*pi

disp(' '); disp('(b) Sine of the angles')
f = sin(angles)

disp(' '); disp('(c) Cosine of the angles')
g = cos(angles)

disp(' '); disp('(d) Sum of squares f.^2 + g.^2')
h = f.^2 + g.^2

disp(' '); disp('(e) Number of elements of h')
length(h)

disp(' '); disp('(f) Sum of the elements of h')
sum(h)

disp(' ')

%% Question 6 - floor and fix
disp('==================================================')
disp('                   QUESTION 6')
disp('==================================================')

disp(' '); disp('x values')
x = [2 2.7 -2 -2.7]

disp(' '); disp('floor(x)')
floor(x)

disp(' '); disp('fix(x)')
fix(x)

disp(' ')

%% Question 7 - mod and rem
disp('==================================================')
disp('                   QUESTION 7')
disp('==================================================')

disp(' '); disp('x and y values')
x = [5 -5 5 -5]
y = [2 2 -2 -2]

disp(' '); disp('mod(x, y)')
mod(x, y)

disp(' '); disp('rem(x, y)')
rem(x, y)

disp(' ')

%% Question 8 - Roots of cubics
disp('==================================================')
disp('                   QUESTION 8')
disp('==================================================')
output_precision(8);

disp(' '); disp('(a) z^3 + z^2 - 2 = 0')
roots([1 1 0 -2])

disp(' '); disp('(b) z^3 + (6+3i)z^2 + (11+18i)z + 33i = 0')
roots([1, 6+3i, 11+18i, 33i])

disp(' '); disp('(c) z^3 + (-1+6i)z^2 + (-13-i)z - 2 - 10i = 0')
roots([1, -1+6i, -13-1i, -2-10i])

disp(' ')

%% Question 9 - Cartesian form of r cis(theta)
disp('==================================================')
disp('                   QUESTION 9')
disp('==================================================')

disp(' '); disp('(a) 4 cis(3pi/5)')
4 * exp(1i * 3*pi/5)

disp(' '); disp('(b) 3 cis(17pi/12)')
3 * exp(1i * 17*pi/12)

disp(' ')

%% Question 10 - Modulus and principal argument (degrees)
disp('==================================================')
disp('                   QUESTION 10')
disp('==================================================')

disp(' '); disp('(a) sqrt(5) + 2i : modulus, then argument in degrees')
z1 = sqrt(5) + 2i;
abs(z1)
rad2deg(angle(z1))

disp(' '); disp('(b) -3 + sqrt(7)i : modulus, then argument in degrees')
z2 = -3 + sqrt(7)*1i;
abs(z2)
rad2deg(angle(z2))

disp(' ')

%% Question 11 - Fifth roots of 4 - 4i  (z^5 - (4 - 4i) = 0)
disp('==================================================')
disp('                   QUESTION 11')
disp('==================================================')

disp(' '); disp('Fifth roots of 4 - 4i')
z = roots([1 0 0 0 0 -(4-4i)])

disp(' '); disp('Check: each root raised to the power 5 (should be 4 - 4i)')
z .^ 5
