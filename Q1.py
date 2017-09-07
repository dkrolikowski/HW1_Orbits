import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u

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

a = 5.0; M1 = 1.0; M2 = 0.1; P = np.sqrt( a ** 3.0 / ( M1 + M2 ) )
e = 0; W = 0.0; I = np.pi/4; w = 0.0; t0 = 0.0

a = 0.4564; M1 = 1.018; M2 = 4.114 * u.jupiterMass.to('solMass'); P = np.sqrt( a ** 3.0 / ( M1 + M2 ) )
e = 0.934; W = 0.0; I = 89.232 * np.pi / 180; w = 300.77 * np.pi / 180.0; t0 = 0.0
astar = M2 / ( M1 + M2 ) * a

tarr = np.linspace( 0.0, 2*P, 10000 )
Marr = 2.0 * np.pi * tarr / P

#Earr = np.array( [ NewtonRaphson( KeplerEq, KeplerEqPrime, x, 1e-9, ( e, x ) ) for x in Marr ] )
x0   = np.array( [ NewtonRaphson( KeplerEq, KeplerEqPrime, x, 1e-9, ( 0.7, x ) ) for x in Marr ] )
x0   = np.array( [ NewtonRaphson( KeplerEq, KeplerEqPrime, x0[i], 1e-9, ( 0.8, Marr[i] ) ) for i in range( Marr.size ) ] )
x0   = np.array( [ NewtonRaphson( KeplerEq, KeplerEqPrime, x0[i], 1e-9, ( 0.85, Marr[i] ) ) for i in range( Marr.size ) ] )
x0   = np.array( [ NewtonRaphson( KeplerEq, KeplerEqPrime, x0[i], 1e-9, ( 0.9, Marr[i] ) ) for i in range( Marr.size ) ] )
Earr = np.array( [ NewtonRaphson( KeplerEq, KeplerEqPrime, x0[i], 1e-9, ( e, Marr[i] ) ) for i in range( Marr.size ) ] )
farr = 2.0 * np.arctan( np.sqrt( ( 1 + e ) / ( 1 - e ) ) * np.tan( Earr / 2.0 ) )
rarr = a * ( 1.0 - e * np.cos( Earr ) )

plt.plot( Marr, Earr )
plt.show()

xarr = rarr * np.cos( farr )
yarr = rarr * np.sin( farr )

obscoos = np.array( [ toObsFrame( np.array( [ xarr[i], yarr[i], 0.0 ] ), W, I, w ) for i in range( xarr.size ) ] )
Xarr = obscoos[:,0]; Yarr = obscoos[:,1]; Zarr = obscoos[:,2]
Rarr = np.sqrt( Xarr ** 2.0 + Yarr ** 2.0 )

plt.figure()
plt.plot( Xarr, Yarr, 'k-' )
plt.plot( xarr, yarr, 'r--' )
plt.vlines( 0.0, -u.solRad.to('au'), 0.0, color = 'r' )

plt.figure()
plt.plot( tarr, Rarr, 'k-' )
plt.plot( tarr, rarr, 'r--' )
for x in [ -1, 1 ]:
    plt.axhline( y = a * ( 1.0 + x * e ), color = 'r' )
plt.show()

n = 2 * np.pi / ( P * u.yr.to('s') )
Zdot = n * astar * u.au.to('km') * np.sin(I) / np.sqrt( 1 - e ** 2 ) * ( np.cos( w + farr ) + e * np.cos(w) )

plt.plot( tarr, np.gradient( Zarr * astar / a * u.au.to('km'), np.diff(tarr*u.yr.to('s'))[0] ), 'k-' )
plt.plot( tarr, Zdot, 'r--' )
plt.show()

#pdb.set_trace()

