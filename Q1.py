import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u

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
    
    e, M = args # Read in eccentricity and mean anomaly
    
    return E - e * np.sin( E ) - M

def KeplerEqPrime( E, args ):
    
    e, M = args # Read in eccentricity and mean anomaly

    return 1.0 - e * np.cos( E )

def RotMatrix( t, axis ):

    # Read in t (angle in rad) and axis about which to rotate
    if axis == 'x': return np.array( [ [ 1, 0, 0 ], [ 0, np.cos(t), - np.sin(t) ], [ 0, np.sin(t), np.cos(t) ] ] )
    elif axis == 'z': return np.array( [ [ np.cos(t), - np.sin(t), 0 ], [ np.sin(t), np.cos(t), 0 ], [ 0, 0, 1 ] ] )

def getPA( x, y ):

    angle = np.arctan2( y, x ) - np.pi / 2.0

    if angle < 0.0: PA = np.degrees( angle + 2.0 * np.pi )
    else:           PA = np.degrees( angle )

    return PA

class OrbitPredictor():

    def __init__( self, a, e, W, I, w, t0, M1, M2, Rs, Rp, p ):
        self.a  = a  # Semimajor axis planet wrt star
        self.e  = e  # Eccentricity
        self.W  = W  # Longitude of ascending node
        self.I  = I  # Inclination
        self.w  = w  # Argument of periapse
        self.t0 = t0 # Time of periapse
        self.M1 = M1 # Mass of star
        self.M2 = M2 # Mass of planet
        self.Rs = Rs # Radius of star
        self.Rp = Rp # Radius of planet
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
            if self.e > 0.8: x0 = NewtonRaphson( KeplerEq, KeplerEqPrime, M, 1e-9, ( 0.75, M ) )
            else:       x0 = M

            E = NewtonRaphson( KeplerEq, KeplerEqPrime, x0, 1e-9, ( self.e, M ) )
            
        else:
            if self.e > 0.8: x0 = np.array( [ NewtonRaphson( KeplerEq, KeplerEqPrime, x, 1e-9, ( 0.75, x ) ) for x in M ] )
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

        f, r = self.getf_r( t ) # Just need f as a function of time

        # Negative sign added to A so that moving in the +z direction (towards us) means RV is negative (as per convention)
        A   = - ( 2.0 * np.pi * np.sin( self.I ) ) / ( self.P * np.sqrt( 1 - self.e ** 2.0 ) ) / u.km.to('AU') / u.yr.to('s')
        RVp = A * self.a2 * ( np.cos( self.w + f ) + self.e * np.cos( self.w ) )   # Planet's RV
        RVs = A * self.a1 * ( np.cos( self.ws + f ) + self.e * np.cos( self.ws ) ) # Star's RV

        return RVp, RVs

    def FindTransits( self, tarr ):

        X, Y, Z = self.getXYZ( self.a, self.w, tarr )
        R       = np.sqrt( X ** 2.0 + Y ** 2.0 ) # Calculate projected sep wrt star

        # Look for when the projected sep is less than the stellar radius + planet radius (would account for grazing transits)
        # AND when the planet is in front of the star ( z > 0.0 )
        transit = np.where( ( R <= self.Rs + self.Rp ) & ( Z > 0.0 ) )[0]

        # Return the times at which there is a transit (re-convert to days)
        return tarr[transit] * u.yr.to('d')

    def PlotOrbit( self ):

        tarr = np.linspace( 0.0, self.P, 10000 )

        X, Y, Z, x, y = self.getXYZ( self.a, self.w, tarr, retInOrbit = True )

        Xp, Yp, Zp, xp, yp = self.getXYZ( self.a2, self.w, tarr, retInOrbit = True )
        Xs, Ys, Zs, xs, ys = self.getXYZ( self.a1, self.ws, tarr, retInOrbit = True )

        posvec     = np.array( [ xs, ys, np.zeros( xs.size ) ] )
        xs, ys, zs = np.array( [ np.dot( RotMatrix( np.pi, 'z' ), posvec[:,i] ) for i in range( posvec.shape[1] ) ] ).T

        plt.figure()
        plt.plot( x, y, 'k-' )
        plt.figure()
        plt.plot( xp, yp, 'r--', xs, ys, 'b--' )
        plt.show()

        plt.figure()
        plt.plot( X, Y, 'k-' )
        plt.figure()
        plt.plot( Xp, Yp, 'r--', Xs, Ys, 'b--' )
        plt.show()
        
        return None
    
##### Question 2, 3, 4 -- HD 80606 #####
a     = 0.4564 # AU
e     = 0.934
W     = np.radians( 160.98 ) # Rad
I     = np.radians( 89.232 ) # Rad
w     = np.radians( 300.77 ) # Rad
t0    = 2451973.72 / u.yr.to('d') # Year
M1    = 1.018 # Sol Mass
M2    = 4.114 * u.jupiterMass.to('solMass') # Sol Mass
Rs    = 1.037 * u.solRad.to('au') # AU
Rp    = 1.029 * u.jupiterRad.to('au')
p     = 15.3e-3 # Arcsec
P     = np.sqrt( a ** 3.0 / ( M1 + M2 ) )

a     = 0.213 # AU
e     = 0.1
W     = np.radians( 25 ) # Rad
I     = np.radians( 84 ) # Rad
w     = np.radians( 339 ) # Rad
t0    = 2451973.72 / u.yr.to('d') # Year
M1    = 0.32 # Sol Mass
M2    = 1.9 * u.jupiterMass.to('solMass') # Sol Mass
Rs    = 1.037 * u.solRad.to('au') # AU
Rp    = 1.029 * u.jupiterRad.to('au')
p     = 215e-3 # Arcsec
P     = np.sqrt( a ** 3.0 / ( M1 + M2 ) )

d = 1 / p * u.pc.to('au')

doQ2 = False
doQ3 = False
doQ4 = False

# Get class instance for HD 80606 orbit
orbit = OrbitPredictor( a, e, W, I, w, t0, M1, M2, Rs, Rp, p )

# orbit.PlotOrbit() # Show orbits in orbit frame (both relative and absolute)

### Question 2 - Radial Velocity ###

if doQ2:
    # Set dates to compute RV for in units of days
    aug1     = Time( '2017-08-01 00:00:00', scale = 'utc' ).jd
    jan1     = Time( '2018-01-01 00:00:00', scale = 'utc' ).jd
    tarr     = np.linspace( aug1, jan1, 100000 )

    # Get planet and stellar RVs
    RVp, RVs = orbit.getRV( tarr * u.d.to('yr') ) # Give tarr in yrs as that's what is used

    plt.clf()
    plt.plot( tarr - 2400000.5, RVs, 'k-' )
    plt.axhline( y = 0.0, color = 'r', ls = ':' )
    plt.ylabel( 'Radial Velocity (km/s)' )
    plt.xlabel( 'MJD (JD - 2400000.5)' )
    plt.savefig( 'Plots/HD80606_StarRV.pdf' )

    # Now find mins and maxs of the radial velocity curve
    mins = signal.argrelmin( RVs )[0] # mins
    maxs = signal.argrelmax( RVs )[0] # maxs

    print '\nRV minima occur at: '
    print Time( tarr[mins], format = 'jd' ).iso

    print '\nRV maxima occur at: '
    print Time( tarr[maxs], format = 'jd' ).iso

### Question 3 - Transits ###

if doQ3:
    # Set dates to search for transits over in units of days (same as Q2)
    aug1     = Time( '2017-08-01 00:00:00', scale = 'utc' ).jd
    jan1     = Time( '2018-01-01 00:00:00', scale = 'utc' ).jd
    tarr     = np.linspace( aug1, jan1, 100000 )

    transits = orbit.FindTransits( tarr * u.d.to('yr') ) # Give tarr in yrs as that's what is used
    
    # Split the transit time array into each separate transit
    transits = np.split( transits, np.where( np.diff( transits ) > 10.0 * np.median( np.diff( transits ) ) )[0] + 1 )

    # Print out start, mid, and end time of transit in UTC
    for t in transits:
        print '\n'
        print Time( t.min(), format = 'jd' ).iso
        print Time( np.median(t), format = 'jd' ).iso
        print Time( t.max(), format = 'jd' ).iso

def RADec_Ecliptic( ra, dec ):

    eps = 23.43699 * np.pi / 180.0

    lat = np.arcsin( np.sin(dec) * np.cos(eps) - np.cos(dec) * np.sin(eps) * np.sin(ra) )
    lon = np.arccos( np.cos(ra) * np.cos(dec) / np.cos(lat) )

    return lon, lat

def Ecliptic_RADec( lon, lat ):

    eps = 23.43699 * np.pi / 180.0

    dec = np.arcsin( np.sin(lat) * np.cos(eps) + np.cos(lat) * np.sin(eps) * np.sin(lon) )
    ra  = np.arccos( np.cos(lon) * np.cos(lat) / np.cos(dec) )

    return ra, dec

def getSolLon( jd ):

    N   = jd - 2451545.0;
    L   = ( 280.460 + 0.9856474 * N ) % 360.0
    g   = ( 357.528 + 0.9856003 * N ) % 360.0

    lon = ( L + 1.915 * np.sin( g * np.pi / 180.0 ) + 0.02 * np.sin( g * np.pi / 90.0 ) ) % 360.0

    return np.radians( lon )

ra   = np.radians( 15 * ( 9 + 22.0 / 60.0 + 37.5769 / 3600.0 ) )
dec  = np.radians( 50 + 36.0 / 60.0 + 13.43 / 3600.0 )

lon, lat = RADec_Ecliptic( ra, dec ) # Get ecliptic coordinates of the star

pma = 45.76 * u.mas.to('rad') * u.d.to('yr') # Convert pm from mas/yr to rad/day
pma  = pma / np.cos( dec )                   # Go from mua cosd to mua (do I need this???)
pmd  = 16.56 * u.mas.to('rad') * u.d.to('yr') # Convert pm from mas/yr to rad/day

# Set array of time over which to calculate astrometric motion in days
aug1   = Time( '2017-08-01 00:00:00', scale = 'utc' ).jd
aug5   = Time( '2022-08-01 00:00:00', scale = 'utc' ).jd
tarr   = np.linspace( aug1, aug5, 100000 )
p      = p * u.arcsec.to('rad')

obstarr   = np.sort( np.random.uniform( aug1, aug5, 100 ) )
obssollon = getSolLon( obstarr )
obsdellon = p * np.sin( obssollon - lon ) / np.cos( lat )
obsdellat = - p * np.sin( lat ) * np.cos( obssollon - lon )

obslon = lon + obsdellon
obslat = lat + obsdellat

obsra, obsdec = Ecliptic_RADec( obslon, obslat )
obsra += pma * ( obstarr - tarr[0] ) - ra
obsdec += pmd * ( obstarr - tarr[0] ) - dec

sigma = 0.2 * u.mas.to('rad')

obsra += np.random.normal( 0.0, sigma, 100 )
obsdec += np.random.normal( 0.0, sigma, 100 )

# Calculate motion due to parallax
sollon = getSolLon( tarr ) # Get the ecliptic longitude of the sun as a function of time

dellon   = p * np.sin( sollon - lon ) / np.cos( lat )
dellat   = - p * np.sin( lat ) * np.cos( sollon - lon )

lon += dellon
lat += dellat

raoft, decoft = Ecliptic_RADec( lon, lat )

rapmt  = pma * ( tarr - tarr[0] )
decpmt = pmd * ( tarr - tarr[0] )

raoft  += rapmt - ra
decoft += decpmt - dec

plt.clf()
plt.plot( raoft * u.rad.to('mas'), decoft * u.rad.to('mas'), 'k--' )
plt.plot( rapmt * u.rad.to('mas'), decpmt * u.rad.to('mas'), 'r:' )
plt.errorbar( obsra * u.rad.to('mas'), obsdec * u.rad.to('mas'), xerr = sigma * u.rad.to('mas'), yerr = sigma * u.rad.to('mas'), fmt = 'k.', ls = 'none' )
plt.show()

X, Y, Z = orbit.getXYZ( orbit.a1, orbit.ws, tarr * u.d.to('yr') )
X = X / orbit.d
Y = Y / orbit.d

raoft += Y
decoft += X

plt.clf()
plt.plot( X * u.rad.to('mas'), Y * u.rad.to('mas'), 'k-' )
plt.show()

plt.clf()
plt.plot( raoft * u.rad.to('mas'), decoft * u.rad.to('mas'), 'k-' )
plt.show()

#plt.clf()
#plt.plot( raoft * u.rad.to('mas'), decoft * u.rad.to('mas'), 'k-' )
#plt.show()

# raarr += X
# decarr += Y


# CoMx = ra + pma * tarr * u.d.to('yr')
# CoMy = dec + pmd * tarr * u.d.to('yr')

# totalx = ( X + CoMx - ra ) * u.deg.to('mas')
# totaly = ( Y + CoMy - dec ) * u.deg.to('mas')

# plt.clf()
# plt.plot( totalx, totaly )
# plt.plot( (CoMx - ra)*u.deg.to('mas'), (CoMy - dec)*u.deg.to('mas') )
# plt.show()
