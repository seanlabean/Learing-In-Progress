# PyClaw_examples

This repository contains several example ipython notebooks demonstrating how to use PyClaw to numerically integrate the hydro equations.

Start with PyClaw_intro to see how to set up a problem, interface with the solver, etc.

Euler_exact_Riemann_solver contains a function to exactly evaluate the solution to the Riemann problem. You can use this solver in interactive mode to gain some intution about how the initial conditions on either side of the interface affect the solution.

Limiters shows examples of how various slope limiters affect the numerical solution to the advection equation.

If you finish playing around with those three, try writing your own first-order Godunov method (using the exact Riemann solver provided in the previous notebook) and compare to the first-order Godunov solver demonstrated in Godunov_intro.
