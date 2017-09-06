import numpy as np
import matplotlib.pyplot as plt
import pdb

def NewtonRaphson( g, gp, x0, tol, args = () ):
    '''Implementation of the Newton Raphson root finding technique.

    Inputs
    ---------
    g    : Name of function you are trying to solve. Takes addition arguments from args.
    gp   : Name of function defining the derivative of g. Takes arguments from args.
    x0   : Initial guess for the solution to g.
    tol  : Percent error tolerance at which to stop solution loop.
    args : Tuple containing arguments for g and gp.

    Outputs
    ---------
    xnew : Solution to g within error tolerance tol.'''

    err = 2 * tol # Initialize error to be larger than tolerance

    while err > tol:
        xnew = x0 - g( x0, args ) / gp( x0, args ) # Calculate next x guess
        err  = ( xnew - x0 ) / x0                  # Calculate relative error
        x0   = xnew                                # Set x0 to xnew for loop
        
    return xnew

def KeplerEq( E, args ):
    '''Kepler equation in root finding form to determine eccentric anomaly E from mean anomaly M.
    For input into NewtonRaphson.

    Inputs
    ---------
    E    : Eccentric anomaly in radians.
    args : Tuple containing the eccentricity e and mean anomaly M (in radians).

    Outputs
    ---------
    out  : The value of the kepler equation in root finding form.'''
    
    e, M = args
    
    return E - e * np.sin( E ) - M

def KeplerEqPrime( E, args ):
    '''Derivative of kepler equation in root finding form for use in NewtonRaphson.

    Inputs
    ---------
    E    : Eccentric anomaly in radians.
    args : Tuple containing the eccentricity e and mean anomaly M (not used here).

    Outputs
    ---------
    out  : The value of the derivative of kepler equation in root finding form.'''

    e, M = args

    return 1.0 - e * np.cos( E )

def xRotMatrix( t ):
    '''Function to return rotation matrix of angle t about the x axis.

    Inputs
    ---------
    t   : Angle of rotation

    Outputs
    ---------
    out : 3x3 rotation matrix.'''
    
    return np.array( [ [ 1, 0, 0 ], [ 0, np.cos(t), - np.sin(t) ], [ 0, np.sin(t), np.cos(t) ] ] )

def zRotMatrix( t ):
    '''Function to return rotation matrix of angle t about the z axis.

    Inputs
    ---------
    t   : Angle of rotation

    Outputs
    ---------
    out : 3x3 rotation matrix.'''

    return np.array( [ [ np.cos(t), - np.sin(t), 0 ], [ np.sin(t), np.cos(t), 0 ], [ 0, 0, 1 ] ] )

def toObsFrame( posvec, W, I, w ):

    rotmat = np.dot( np.dot( zRotMatrix(W), xRotMatrix(I) ), zRotMatrix(w) )

    return np.dot( rotmat, posvec )

e = 0.5
a = 1.0
W = np.pi / 4
I = np.pi / 5
w = np.pi / 3

Marr = np.linspace( 0, 2 * np.pi, 10000 )

Earr = np.array( [ NewtonRaphson( KeplerEq, KeplerEqPrime, x, 1e-9, ( e, x ) ) for x in Marr ] )
farr = 2.0 * np.arctan( np.sqrt( ( 1 + e ) / ( 1 - e ) ) * np.tan( Earr / 2.0 ) )
rarr = a * ( 1.0 - e * np.cos( Earr ) )

xarr = rarr * np.cos( farr )
yarr = rarr * np.sin( farr )

obscoos = np.array( [ toObsFrame( np.array( [ xarr[i], yarr[i], 0.0 ] ), W, I, w ) for i in range( xarr.size ) ] )
Xarr = obscoos[:,0]; Yarr = obscoos[:,1]; Zarr = obscoos[:,2]
Rarr = np.sqrt( Xarr ** 2.0 + Yarr ** 2.0 )

plt.plot( Xarr, Yarr, 'k-' )
plt.plot( xarr, yarr, 'r--' )
plt.show()

plt.clf()
plt.plot( Earr, Rarr, 'k-' )
plt.plot( Earr, rarr, 'r--' )
for x in [ -1, 1 ]:
    plt.axhline( y = a * ( 1.0 + x * e ), color = 'r' )
plt.show()
