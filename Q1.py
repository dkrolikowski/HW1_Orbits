import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u

import pdb

def NewtonRaphson( g, gp, x0, tol, args = () ):
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

def xRotMatrix( t ):
    return np.array( [ [ 1, 0, 0 ], [ 0, np.cos(t), - np.sin(t) ], [ 0, np.sin(t), np.cos(t) ] ] )

def zRotMatrix( t ):
    return np.array( [ [ np.cos(t), - np.sin(t), 0 ], [ np.sin(t), np.cos(t), 0 ], [ 0, 0, 1 ] ] )

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

    def retSepPA( self, t ):

        M = 2.0 * np.pi * ( t - self.t0 ) / self.P
        E = NewtonRaphson( KeplerEq, KeplerEqPrime, M, 1e-9, ( self.e, M ) )
        f = 2.0 * np.arctan( np.sqrt( ( 1 + self.e ) / ( 1 - self.e ) ) * np.tan( E / 2.0 ) )

        r = self.a * ( 1 - self.e * np.cos( E ) )
        x = r * np.cos( f )
        y = r * np.sin( f )

        X, Y, Z = self.toObsFrame( np.array( [ x, y, 0.0 ] ) )
        R       = np.sqrt( X ** 2.0 + Y ** 2.0 )

        angle = np.arctan2( Y, X ) - np.pi / 2.0
        if angle < 0.0: PA = np.degrees( angle + 2.0 * np.pi )
        else:           PA = np.degrees( angle )

        return R, PA    
        
    def PlotOrbit( self ):
        tarr = np.linspace( 0.0, self.P, 10000 )
        Marr = 2.0 * np.pi * tarr / self.P

        Earr = np.array( [ NewtonRaphson( KeplerEq, KeplerEqPrime, x, 1e-9, ( self.e, x ) ) for x in Marr ] )
        farr = 2.0 * np.arctan( np.sqrt( ( 1 + self.e ) / ( 1 - self.e ) ) * np.tan( Earr / 2.0 ) )

        rarr = self.a * ( 1 - self.e * np.cos( Earr ) )
        xarr = rarr * np.cos( farr )
        yarr = rarr * np.sin( farr )

        Xarr, Yarr, Zarr = self.toObsFrame( np.array( [ xarr, yarr, np.zeros( xarr.size ) ] ) )
        Rarr             = np.sqrt( Xarr ** 2.0 + Yarr ** 2.0 )

        plt.figure()
        plt.plot( tarr, Rarr, 'k-' )
        plt.plot( tarr, rarr, 'r--' )
        for x in [ -1, 1 ]: plt.axhline( y = self.a * ( 1 + x * self.e ), color = 'b', linestyle = ':' )

        plt.figure()
        plt.plot( Xarr, Yarr, 'k-' )
        plt.plot( xarr, yarr, 'r--' )

        plt.show()

        return None

##### Question 1

# Set orbital parameters
a = 0.4564; e = 0.2; W = 0.0; I = 89.232 * np.pi / 180; w = 300.77 * np.pi / 180.0; t0 = 0.0
M1 = 1.018; M2 = 4.114 * u.jupiterMass.to('solMass');

P = np.sqrt( a ** 3.0 / ( M1 + M2 ) )

t = 2 * P / 5

orbit = OrbitPredictor( a, e, W, I, w, t0, M1, M2 )
r, t  = orbit.retSepPA(t)
print r, t

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
