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

class OrbitPredictor():

    def __init__( self, a, e, W, I, w, t0, M1, M2 ):
        self.a  = a
        self.e  = e
        self.W  = W
        self.I  = I
        self.w  = w
        self.t0 = t0
        self.M1 = M1
        self.M2 = M2

        self.q  = M2 / M1
        self.P  = np.sqrt( a ** 3.0 / ( M1 + M2 ) )
        self.a1 = self.q / ( 1.0 + self.q ) * a
        self.a2 = 1.0 / ( 1.0 + self.q ) * a

    def toObsFrame( self, posvec ):

        rotmat = np.dot( np.dot( zRotMatrix( self.W ), xRotMatrix( self.I ) ), zRotMatrix( self.w ) )

        if posvec.shape[0] == posvec.size: return np.dot( rotmat, posvec )
        else: return np.array( [ np.dot( rotmat, posvec[:,i] ) for i in range( posvec.shape[1] ) ] ).T

    def PlotOrbit( self ):
        tarr = np.linspace( 0.0, self.P, 10000 )
        Marr = 2.0 * np.pi * tarr / self.P

        Earr = np.array( [ NewtonRaphson( KeplerEq, KeplerEqPrime, x, 1e-9, ( self.e, x ) ) for x in Marr ] )
        farr = 2.0 * np.arctan( np.sqrt( ( 1 + self.e ) / ( 1 - self.e ) ) * np.tan( Earr / 2.0 ) )

        rarr = self.a * ( 1 - self.e * np.cos( Earr ) )
        xarr = rarr * np.cos( farr )
        yarr = rarr * np.sin( farr )

        Xarr, Yarr, Zarr = np.array()
        return None
        #Xarr, Yarr, Zarr = np.array( [ toObsFrame( np.array( [ xarr[i], yarr[i], 0.0 ] ), self.W, self.I, self.w ) for i in range( xarr.size )

def PlotOrbit( a, e, W, I, w, t0, P ):
    tarr = np.linspace( 0.0, P, 10000 )
    Marr = 2.0 * np.pi * tarr / P

    Earr = np.array( [ NewtonRaphson( KeplerEq, KeplerEqPrime, x, 1e-9, ( e, x ) ) for x in Marr ] )
    farr = 2.0 * np.arctan( np.sqrt( ( 1 + e ) / ( 1 - e ) ) * np.tan( Earr / 2.0 ) )

    rarr = a * ( 1.0 - e * np.cos( Earr ) )
    xarr = rarr * np.cos( farr )
    yarr = rarr * np.sin( farr )

    rotmat = np.dot( np.dot( zRotMatrix( W ), xRotMatrix( I ) ), zRotMatrix( w ) )
    Xarr, Yarr, Zarr = np.array( [ toObsFrame( np.array( [ xarr[i], yarr[i], 0.0 ] ), W, I, w ) for i in range( xarr.size ) ] ).T
    pdb.set_trace()
    Rarr = np.sqrt( Xarr ** 2.0 + Yarr ** 2.0 )
    
    plt.figure()
    plt.plot( tarr, Rarr, 'k-' )
    plt.plot( tarr, rarr, 'r--' )
    for x in [ -1.0, 1.0 ]:
        plt.axhline( y = a * ( 1.0 + x * e ), color = 'b', linestyle = ':' )

    plt.figure()
    plt.plot( Xarr, Yarr, 'k-' )
    plt.plot( xarr, yarr, 'r--' )

    return None

##### Question 1

# Set orbital parameters
a = 0.4564; e = 0.2; W = 0.0; I = 89.232 * np.pi / 180; w = 300.77 * np.pi / 180.0; t0 = 0.0
M1 = 1.018; M2 = 4.114 * u.jupiterMass.to('solMass');

P = np.sqrt( a ** 3.0 / ( M1 + M2 ) )

t = 8 * P / 8.0
M = 2.0 * np.pi * ( t - t0 ) / P
E = NewtonRaphson( KeplerEq, KeplerEqPrime, M, 1e-9, ( e, M ) )
f = 2.0 * np.arctan( np.sqrt( ( 1 + e ) / ( 1 - e ) ) * np.tan( E / 2.0 ) )

r = a * ( 1.0 - e * np.cos( E ) )
x = r * np.cos( f )
y = r * np.sin( f )

X, Y, Z = toObsFrame( np.array( [ x, y, 0.0 ] ), W, I, w )
pdb.set_trace()
R       = np.sqrt( X ** 2.0 + Y ** 2.0 )

angle = np.arctan2( Y, X ) - np.pi / 2.0
if angle < 0.0:
    PA = np.degrees( angle + 2 * np.pi )
else:
    PA = np.degrees( angle )

PlotOrbit( a, e, W, I, w, t0, P )

# astar = M2 / ( M1 + M2 ) * a

# n     = 2 * np.pi / ( P * u.yr.to('s') )
# Zdot  = n * astar * u.au.to('km') * np.sin(I) / np.sqrt( 1 - e ** 2 ) * ( np.cos( w + farr ) + e * np.cos(w) )

# plt.plot( tarr, np.gradient( Zarr * astar / a * u.au.to('km'), np.diff(tarr*u.yr.to('s'))[0] ), 'k-' )
# plt.plot( tarr, Zdot, 'r--' )
# plt.show()

# x0   = np.array( [ NewtonRaphson( KeplerEq, KeplerEqPrime, x, 1e-9, ( 0.7, x ) ) for x in Marr ] )
# x0   = np.array( [ NewtonRaphson( KeplerEq, KeplerEqPrime, x0[i], 1e-9, ( 0.8, Marr[i] ) ) for i in range( Marr.size ) ] )
# x0   = np.array( [ NewtonRaphson( KeplerEq, KeplerEqPrime, x0[i], 1e-9, ( 0.85, Marr[i] ) ) for i in range( Marr.size ) ] )
# x0   = np.array( [ NewtonRaphson( KeplerEq, KeplerEqPrime, x0[i], 1e-9, ( 0.9, Marr[i] ) ) for i in range( Marr.size ) ] )
# Earr = np.array( [ NewtonRaphson( KeplerEq, KeplerEqPrime, x0[i], 1e-9, ( e, Marr[i] ) ) for i in range( Marr.size ) ] )
