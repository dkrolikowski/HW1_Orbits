import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
import matplotlib as mpl

from scipy import signal
from astropy.time import Time

import pdb

def NewtonRaphson( g, gp, x0, tol, args = () ):
    err = 2 * tol # Initialize error to be larger than tolerance
    while err > tol:
       xnew = x0 - g( x0, args ) / gp( x0, args ) # Calculate next x guess
       err  = ( xnew - x0 ) / x0                  # Calculate relative error
       x0   = xnew                                # Set x0 to xnew for loop
        
    return x0

def KeplerEq( E, args ):
    e, M = args
    
    return E - e * np.sin( E ) - M

def KeplerEqPrime( E, args ):
    e, M = args

    return 1.0 - e * np.cos( E )

def RotMatrix( t, axis ):
    if axis == 'x': return np.array( [ [ 1, 0, 0 ], [ 0, np.cos(t), - np.sin(t) ], [ 0, np.sin(t), np.cos(t) ] ] )
    elif axis == 'z': return np.array( [ [ np.cos(t), - np.sin(t), 0 ], [ np.sin(t), np.cos(t), 0 ], [ 0, 0, 1 ] ] )

def getPA( x, y ):

    angle = np.arctan2( y, x ) - np.pi / 2.0

    if angle < 0.0: PA = np.degrees( angle + 2.0 * np.pi )
    else:           PA = np.degrees( angle )

    return PA

class OrbitPredictor():

    def __init__( self, a, e, W, I, w, t0, M1, M2, Rs, p ):
        self.a  = a  # Semimajor axis planet wrt star
        self.e  = e  # Eccentricity
        self.W  = W  # Longitude of ascending node
        self.I  = I  # Inclination
        self.w  = w  # Argument of periapse
        self.t0 = t0 # Time of periapse
        self.M1 = M1 # Mass of star
        self.M2 = M2 # Mass of planet
        self.Rs = Rs # Radius of star
        self.p  = p  # Parallax of system

        self.ws = w - np.pi # w of star, offset from planet by pi
        self.d  = 1 / self.p * u.pc.to('au')
        self.q  = M2 / M1
        self.P  = np.sqrt( a ** 3.0 / ( M1 + M2 ) )
        self.a1 = self.q / ( 1.0 + self.q ) * a
        self.a2 = 1.0 / ( 1.0 + self.q ) * a
        
    def getf_r( self, t ):

        M = 2.0 * np.pi * ( t - self.t0 ) / self.P

        if np.isscalar( M ):
            if e > 0.8: x0 = NewtonRaphson( KeplerEq, KeplerEqPrime, M, 1e-9, ( 0.75, M ) )
            else:       x0 = M

            E = NewtonRaphson( KeplerEq, KeplerEqPrime, x0, 1e-9, ( self.e, M ) )
            
        else:
            if e > 0.8: x0 = np.array( [ NewtonRaphson( KeplerEq, KeplerEqPrime, x, 1e-9, ( 0.75, x ) ) for x in M ] )
            else:       x0 = M

            E = np.array( [ NewtonRaphson( KeplerEq, KeplerEqPrime, x0[i], 1e-9, ( self.e, M[i] ) ) for i in range( M.size ) ] )

        f = 2.0 * np.arctan( np.sqrt( ( 1 + self.e ) / ( 1 - self.e ) ) * np.tan( E / 2.0 ) )
        r = self.a * ( 1 - self.e * np.cos( E ) )
        
        return f, r

    def toObsFrame( self, posvec, w ):

        rotmat = np.dot( np.dot( RotMatrix( self.W, 'z' ), RotMatrix( self.I, 'x' ) ), RotMatrix( w, 'z' ) )

        if posvec.shape[0] == posvec.size: return np.dot( rotmat, posvec )
        else: return np.array( [ np.dot( rotmat, posvec[:,i] ) for i in range( posvec.shape[1] ) ] ).T

    def getXYZ( self, a, w, t, retInOrbit = False ):

        f, r = self.getf_r( t )
        r    = r * a / self.a

        x = r * np.cos( f )
        y = r * np.sin( f )

        X, Y, Z = self.toObsFrame( np.array( [ x, y, 0.0 ] ), w )

        if retInOrbit: return X, Y, Z, x, y
        else:          return X, Y, Z

    def getSepPA( self, t ):

        X, Y, Z    = self.getXYZ( self.a, self.w, t )
        Xp, Yp, Zp = self.getXYZ( self.a2, self.w, t )
        Xs, Ys, Zs = self.getXYZ( self.a1, self.ws, t )

        wrtStar = [ np.sqrt( X ** 2.0 + Y ** 2.0 ) / self.d * u.rad.to('mas'), getPA( X, Y ) ]
        PwrtCoM = [ np.sqrt( Xp ** 2.0 + Yp ** 2.0 ) / self.d * u.rad.to('mas'), getPA( Xp, Yp ) ]
        SwrtCoM = [ np.sqrt( Xs ** 2.0 + Ys ** 2.0 ) / self.d * u.rad.to('mas'), getPA( Xs, Ys ) ]

        return wrtStar, PwrtCoM, SwrtCoM

    def getRV( self, t ):

        M   = 2.0 * np.pi * ( t - self.t0 ) / self.P
        x0  = NewtonRaphson( KeplerEq, KeplerEqPrime, M, 1e-9, ( 0.7, M ) )
        E   = NewtonRaphson( KeplerEq, KeplerEqPrime, x0, 1e-9, ( self.e, M ) )
        f   = 2.0 * np.arctan( np.sqrt( ( 1 + self.e ) / ( 1 - self.e ) ) * np.tan( E / 2.0 ) )

        A   = - ( 2.0 * np.pi * np.sin( self.I ) ) / ( self.P * np.sqrt( 1 - self.e ** 2.0 ) ) / u.km.to('AU') / u.yr.to('s')
        RVp = A * self.a2 * ( np.cos( self.w + f ) + self.e * np.cos( self.w ) )
        RVs = A * self.a1 * ( np.cos( self.ws + f ) + self.e * np.cos( self.ws ) )

        return RVp, RVs
                                    
    def PlotOrbit( self ):

        tarr = np.linspace( 0.0, self.P, 10000 )

        X, Y, Z, x, y = self.getXYZ( self.a, self.w, tarr, retInOrbit = True )
        R = np.sqrt( X ** 2.0 + Y ** 2.0 )
        transit = np.where( R <= u.solRad.to('au') )[0]

        Xp, Yp, Zp, xp, yp = self.getXYZ( self.a2, self.w, tarr, retInOrbit = True )
        Xs, Ys, Zs, xs, ys = self.getXYZ( self.a1, self.ws, tarr, retInOrbit = True )

        posvec     = np.array( [ xs, ys, np.zeros( xs.size ) ] )
        xs, ys, zs = np.array( [ np.dot( RotMatrix( np.pi, 'z' ), posvec[:,i] ) for i in range( posvec.shape[1] ) ] ).T

        plt.figure()
        plt.plot( x, y, 'k-' )
        ax = plt.gca()
        ax.add_patch(mpl.patches.Circle((0,0),radius=u.solRad.to('au'),color='r'))
        plt.figure()
        plt.plot( xp, yp, 'r--' )
        plt.plot( xp[0], yp[0], 'ro' )
        plt.plot( xs, ys, 'b--' )
        plt.plot( xs[0], ys[0], 'bo' )
        plt.show()

        plt.figure()
        plt.plot( X, Y, 'k-' )
        ax = plt.gca()
        ax.add_patch(mpl.patches.Circle((0,0),radius=1*u.solRad.to('au'),color='r'))
        for t in transit: ax.add_patch( mpl.patches.Circle( (X[t],Y[t]),radius=u.jupiterRad.to('au'), color='b' ) )
        plt.figure()
        plt.plot( Xp, Yp, 'r--' )
        plt.plot( Xs, Ys, 'b--' )
        plt.show()
        
        return None

    def PlotRV( self, tarr ):

        RVp, RVs = np.array( [ self.getRV( t ) for t in tarr ] ).T
        tarrplot = tarr * u.yr.to('d') - 2451900.0

        plt.clf()
        fig, ( axs, axp ) = plt.subplots( 2, 1, sharex = True )

        axs.plot( tarrplot, RVs, 'k-' )
        axs.set_ylabel( 'Stellar Velocity (km/s)' )

        axp.plot( tarrplot, RVp, 'r-' )
        axp.set_ylabel( 'Planet Velocity (km/s)' )
        axp.set_xlabel( 'JD - 2451900' )

        fig.subplots_adjust( hspace = 0 )
        fig.suptitle( 'HD 80606 Radial Velocity Curves' )

        plt.savefig('Plots/HD80606_RV.pdf')

        plt.clf()
        plt.plot( tarrplot, RVs, 'k-' )
        plt.axhline( y = 0.0, color = 'r', ls = ':' )
        plt.ylabel( 'Radial Velocity (km/s)' )
        plt.xlabel( 'JD - 2451900' )
        plt.title( 'HD 80606 Stellar Radial Velocity Curve' )
        plt.savefig( 'Plots/HD80606_StarRV.pdf' )
        plt.clf()
        
        return RVp, RVs

    def FindTransits( self, tarr ):

        X, Y, Z = self.getXYZ( self.a, self.w, tarr / u.yr.to('d') )
        R       = np.sqrt( X ** 2.0 + Y ** 2.0 )

        transit = np.where( ( R <= self.Rs ) & ( Z > 0.0 ) )[0]

        return tarr[transit]
    
##### Question 2, 3, 4 -- HD 80606 #####
a     = 0.4564 # AU
e     = 0.934
W     = 160.98 * np.pi / 180.0 # Rad
I     = 89.232 * np.pi / 180.0 # Rad
w     = 300.77 * np.pi / 180.0 # Rad
t0    = 2451973.72 / u.yr.to('d') # Year
M1    = 1.018 # Sol Mass
M2    = 4.114 * u.jupiterMass.to('solMass') # Sol Mass
Rs    = 1.037 * u.solRad.to('au') # AU
p     = 15.3e-3 # Arcsec
P     = np.sqrt( a ** 3.0 / ( M1 + M2 ) )

doQ2 = False
doQ3 = False

# Get class instance for HD 80606 orbit
orbit = OrbitPredictor( a, e, W, I, w, t0, M1, M2, Rs, p )
#orbit.PlotOrbit() # Show orbits in orbit frame (both relative and absolute)

### Question 2 - Radial Velocity ###

if doQ2:
    # Set dates to compute RV for
    aug1     = Time( '2017-08-01 00:00:00', scale = 'utc' ).jd
    jan1     = Time( '2018-01-01 00:00:00', scale = 'utc' ).jd
    tarr     = np.linspace( aug1, jan1, 100000 )

    # Plot and return planet and stellar RVs
    RVp, RVs = orbit.PlotRV( tarr / u.yr.to('d') )

    mins     = signal.argrelmin( RVs )[0]
    maxs     = signal.argrelmax( RVs )[0]

    print 'RV minima occur at: '
    print Time( tarr[mins], format = 'jd' ).iso

    print '\nRV maxima occur at: '
    print Time( tarr[maxs], format = 'jd' ).iso

### Question 3 - Transits ###

if doQ3:
    transits = orbit.FindTransits( tarr )
    transits = np.split( transits, np.where( np.diff( transits ) > 10.0 * np.median( np.diff( transits ) ) )[0] + 1 )
    for t in transits:
        print Time( t.min(), format = 'jd' ).iso
        print Time( np.median(t), format = 'jd' ).iso
        print Time( t.max(), format = 'jd' ).iso
        print '\n'

dec = 50 + 36.0 / 60.0 + 13.43 / 3600.0
ra  = 15 * ( 9 + 22.0 / 60.0 + 37.5769 / 3600.0 )
pma = 45.76 * u.mas.to('deg') * u.d.to('yr')
pmd = 16.56 * u.mas.to('deg') * u.d.to('yr')

aug1     = Time( '2017-08-01 00:00:00', scale = 'utc' ).jd
aug5     = Time( '2023-08-01 00:00:00', scale = 'utc' ).jd
tarr     = np.linspace( aug1, aug5, 100000 )

X, Y, Z = orbit.getXYZ( orbit.a1, orbit.ws, tarr / u.yr.to('d') )
X = X / orbit.d * u.rad.to('deg')
Y = Y / orbit.d * u.rad.to('deg')

CoMx = ra + pma * tarr * u.d.to('yr')
CoMy = dec + pmd * tarr * u.d.to('yr')

totalx = ( X + CoMx - ra ) * u.deg.to('mas')
totaly = ( Y + CoMy - dec ) * u.deg.to('mas')

plt.clf()
plt.plot( totalx, totaly )
plt.show()
