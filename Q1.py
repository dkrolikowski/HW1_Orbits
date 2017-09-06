import numpy as np

def NewtonRaphson( g, gp, x0, tol, args = () ):
    '''Implementation of the Newton Raphson root finding technique

    Inputs
    ---------
    g    : Name of function you are trying to solve. Takes addition arguments from args.
    gp   : Name of function defining the derivative of g. Takes arguments from args.
    x0   : Initial guess for the solution to g.
    tol  : Percent error tolerance at which to stop solution loop.
    args : Tuple containing arguments for g and gp

    Outputs
    ---------
    xnew : Solution to g within error tolerance tol'''

    err = 2 * tol # Initialize error to be larger than tolerance

    while err > tol:
        xnew = x0 - g( x0, args ) / gp( x0, args ) # Calculate next x guess
        err  = ( xnew - x0 ) / x0                  # Calculate relative error
        x0   = xnew                                # Set x0 to xnew for loop

    return xnew
