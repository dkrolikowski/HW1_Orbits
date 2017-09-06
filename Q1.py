import numpy as np
import matplotlib.pyplot as plt
import pdb

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

def KeplerEq( E, args ):

    e, M = args
    
    return E - e * np.sin( E ) - M

def KeplerEqPrime( E, args ):

    e, M = args

    return 1.0 - e * np.cos( E )

Marr    = np.linspace( 0, 2 * np.pi, 10000 )
# earr    = np.array( [ 0.0, 0.2, 0.5, 0.7, 0.8, 0.9, 0.95, 0.99 ] )
# Earr    = np.zeros( ( earr.size , Marr.size ) )
# Earr[0] = Marr

# for i in range( 1, earr.size ):
#     tmp     = [ NewtonRaphson( KeplerEq, KeplerEqPrime, Earr[i-1,j] , 1e-9, ( earr[i], Marr[j] ) ) for j in range( Marr.size ) ]
#     tmp     = np.array(tmp)
#     Earr[i] = tmp

# plt.clf()
# for i in range( earr.size ):
#     plt.plot( Marr, Earr[i] )

# plt.show()

e    = 0.4; a = 1.0
Earr = np.array( [ NewtonRaphson( KeplerEq, KeplerEqPrime, x, 1e-9, ( e, x ) ) for x in Marr ] )
farr = 2.0 * np.arctan( np.sqrt( ( 1 + e ) / ( 1 - e ) ) * np.tan( Earr / 2.0 ) )
rarr = a * ( 1.0 - e * np.cos( Earr ) )

xarr = rarr * np.cos( farr )
yarr = rarr * np.sin( farr )

plt.plot( xarr, yarr )
plt.show()
