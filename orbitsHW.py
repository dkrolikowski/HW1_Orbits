import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
from scipy import signal
from astropy.time import Time

########## FUNCTIONS NEEDED FOR CORE ORBITPREDICTOR CLASS ##########

def NewtonRaphson( g, gp, x0, tol, args = () ):
    # Newton Raphson root finding method
    # g is equation to find root of, gp is its derivative, x0 is an initial guess
    # tol is the relative error tolerance, args is a tuple of arguments for g/gp
    
    err = 2 * tol # Initialize error to be larger than tolerance
    while err > tol:
       xnew = x0 - g( x0, args ) / gp( x0, args ) # Calculate next x guess
       err  = ( xnew - x0 ) / x0                  # Calculate relative error
       x0   = xnew                                # Set x0 to xnew for loop
        
    return x0

def KeplerEq( E, args ):
    # Kepler Equation to solve for eccentric anomaly
    # E is the eccentric anomaly, args contains eccentricity and mean anomaly
    
    e, M = args
    
    return E - e * np.sin( E ) - M

def KeplerEqPrime( E, args ):
    # Derivative of Kepler Equation to solve for eccentric anomaly
    # E is the eccentric anomaly, args contains eccentricity and mean anomaly
    
    e, M = args

    return 1.0 - e * np.cos( E )

def RotMatrix( t, axis ):
    # Rotation matrix. t is angle in rad to rotate, and axis is axis to rotate about

    if axis == 'x': return np.array( [ [ 1, 0, 0 ], [ 0, np.cos(t), - np.sin(t) ], [ 0, np.sin(t), np.cos(t) ] ] )
    elif axis == 'y': return np.array( [ [ np.cos(t), 0, np.sin(t) ], [ 0, 1, 0 ], [ - np.sin(t), 0, np.cos(t) ] ] )
    elif axis == 'z': return np.array( [ [ np.cos(t), - np.sin(t), 0 ], [ np.sin(t), np.cos(t), 0 ], [ 0, 0, 1 ] ] )

########## ORBITPREDICTOR CLASS - CORE OF CODE ##########

class OrbitPredictor():

    def __init__( self, a, e, W, I, w, t0, M1, M2 ):
        
        self.a   = a   # Semimajor axis planet wrt star
        self.e   = e   # Eccentricity
        self.W   = W   # Longitude of ascending node
        self.I   = I   # Inclination
        self.w   = w   # Argument of periapse
        self.t0  = t0  # Time of periapse
        self.M1  = M1  # Mass of star
        self.M2  = M2  # Mass of planet

        self.ws  = w - np.pi # w of star, offset from planet by pi
        self.q   = M2 / M1
        self.P   = np.sqrt( a ** 3.0 / ( M1 + M2 ) )
        self.a1  = self.q / ( 1.0 + self.q ) * a
        self.a2  = 1.0 / ( 1.0 + self.q ) * a
        
    def getf_r( self, t ):
        # Get f and r at a given t or array of t for input parameters

        M = 2.0 * np.pi * ( t - self.t0 ) / self.P # Mean Anomaly

        if np.isscalar( M ): # If given only a single time
            if self.e > 0.8: x0 = NewtonRaphson( KeplerEq, KeplerEqPrime, M, 1e-9, ( 0.75, M ) ) # Test for high eccentricity case
            else:       x0 = M

            E = NewtonRaphson( KeplerEq, KeplerEqPrime, x0, 1e-9, ( self.e, M ) )
            
        else: # If given an array
            if self.e > 0.8: x0 = np.array( [ NewtonRaphson( KeplerEq, KeplerEqPrime, x, 1e-9, ( 0.75, x ) ) for x in M ] )
            else:       x0 = M

            E = np.array( [ NewtonRaphson( KeplerEq, KeplerEqPrime, x0[i], 1e-9, ( self.e, M[i] ) ) for i in range( M.size ) ] )

        f = 2.0 * np.arctan( np.sqrt( ( 1 + self.e ) / ( 1 - self.e ) ) * np.tan( E / 2.0 ) ) # Calculate f
        r = self.a * ( 1 - self.e * np.cos( E ) ) # Calculate r
        
        return f, r

    def toObsFrame( self, posvec, w ):
        # Rotate given x,y to observer frame. Note w is an input because it can be either w or wstar

        rotmat = np.dot( np.dot( RotMatrix( self.W, 'z' ), RotMatrix( self.I, 'x' ) ), RotMatrix( w, 'z' ) ) # Rotation matrix

        if posvec.shape[0] == posvec.size: return np.dot( rotmat, posvec ) # Return single position vector case
        else: return np.array( [ np.dot( rotmat, posvec[:,i] ) for i in range( posvec.shape[1] ) ] ).T # Return array of position vectors case

    def getXYZ( self, a, w, t, retInOrbit = False ):
        # Get cartesian coordinates of orbit at a given time. a and w are inputs for all 3 different orbit cases (planet wrt star, star/planet wrt CoM)
        # retInOrbit is a condition that if True will return the orbit frame x, y in addition to the observer frame X, Y, Z
        
        f, r = self.getf_r( t )
        r    = r * a / self.a # Correct r for whichever orbit you are calculating

        x = r * np.cos( f )
        y = r * np.sin( f )

        X, Y, Z = self.toObsFrame( np.array( [ x, y, 0.0 ] ), w )

        if retInOrbit: return X, Y, Z, x, y
        else:          return X, Y, Z

    def getSepPA( self, a, w, t ):

        # Get PA from +X to -Y, and correct for any negative angles
        def getPA( x, y ): return ( np.arctan2( -y, x ) * 180 / np.pi ) % 360.0

        X, Y, Z = self.getXYZ( a, w, t )

        return np.sqrt( X ** 2.0 + Y ** 2.0 ), getPA( X, Y )

    def getRV( self, t ):

        f, r = self.getf_r( t ) # Just need f as a function of time

        # Negative sign added to A so that moving in the +z direction (towards us) means RV is negative (as per convention)
        A   = - ( 2.0 * np.pi * np.sin( self.I ) ) / ( self.P * np.sqrt( 1 - self.e ** 2.0 ) ) / u.km.to('AU') / u.yr.to('s')
        RVp = A * self.a2 * ( np.cos( self.w + f ) + self.e * np.cos( self.w ) )   # Planet's RV
        RVs = A * self.a1 * ( np.cos( self.ws + f ) + self.e * np.cos( self.ws ) ) # Star's RV

        return RVp, RVs

    def PlotOrbit( self ):
        # Simple orbit plotting routine over 1 period
        
        tarr = np.linspace( 0.0, self.P, 10000 )

        X, Y, Z, x, y = self.getXYZ( self.a, self.w, tarr, retInOrbit = True )

        Xp, Yp, Zp, xp, yp = self.getXYZ( self.a2, self.w, tarr, retInOrbit = True )
        Xs, Ys, Zs, xs, ys = self.getXYZ( self.a1, self.ws, tarr, retInOrbit = True )

        posvec     = np.array( [ xs, ys, np.zeros( xs.size ) ] )
        xs, ys, zs = np.array( [ np.dot( RotMatrix( np.pi, 'z' ), posvec[:,i] ) for i in range( posvec.shape[1] ) ] ).T

        plt.clf()
        fig, ( orb, obs ) = plt.subplots( 1, 2 )

        orb.plot( y, x, 'k-' )
        orb.plot( yp, xp, 'r--', ys, xs, 'b--' )
        orb.set_ylabel( 'North-South Offset (au)' ); orb.set_xlabel( 'East-West Offset (au)' )

        obs.plot( Y, X, 'k-' )
        obs.plot( Yp, Xp, 'r--', Ys, Xs, 'b--' )
        obs.set_xlabel( 'East-West Offset (au)' )
        plt.savefig( 'orbit.pdf' )
        
        return None

########## FUNCTIONS FOR ASTROMETRIC PREDICTIONS (PARALLAX MOTION AND RA/DEC/TIME PLOTS ##########

def getSolLon( jd ):
    # Calculate ecliptic longitude of the sun given a time in julian days
    # Return solar longitude in radians

    N   = jd - 2451545.0;
    L   = ( 280.460 + 0.9856474 * N ) % 360.0
    g   = ( 357.528 + 0.9856003 * N ) % 360.0

    lon = ( L + 1.915 * np.sin( g * np.pi / 180.0 ) + 0.02 * np.sin( g * np.pi / 90.0 ) ) % 360.0

    return np.radians( lon )

def RADec_Ecliptic( ra, dec ):
    # Convert from RA/Dec to ecliptic longitude and latitude
    # All angles in radians

    eps = 23.43699 * np.pi / 180.0 # Earth's obliquity

    lat = np.arcsin( np.sin(dec) * np.cos(eps) - np.cos(dec) * np.sin(eps) * np.sin(ra) )
    lon = np.arccos( np.cos(ra) * np.cos(dec) / np.cos(lat) )

    return lon, lat

def Ecliptic_RADec( lon, lat ):
    # Convert from ecliptic longitude/latitue to RA and Dec
    # All angles in radians

    eps = 23.43699 * np.pi / 180.0 # Earth's obliquity

    dec = np.arcsin( np.sin(lat) * np.cos(eps) + np.cos(lat) * np.sin(eps) * np.sin(lon) )
    ra  = np.arccos( np.cos(lon) * np.cos(lat) / np.cos(dec) )

    return ra, dec

def getPlxMotion( tarr, ra, dec, p ):
        # tarr in JD, ra/dec/p in radians

        lon, lat = RADec_Ecliptic( ra, dec ) # Get ecliptic coordinates of the star
        
        sollon   = getSolLon( tarr )               # Get ecliptic longitude of sun as function of time
        dellon   = p * np.sin( sollon - lon )      # Get change in longitude as function of time
        dellat   = - p * np.sin( lat ) * np.cos( sollon - lon ) # Get change in latitude as function of time

        lon += dellon # Add dellon to lon of star
        lat += dellat # Add dellat to lat of star

        rat, dect = Ecliptic_RADec( lon, lat ) # Get ra and dec as functions of t from lon/lat

        # Return movement from CoM RA/Dec
        return rat - ra, dect - dec

def RADecPlots( tarr, ra, dec, obst, obsra, obsdec, sig, plotname, unit = 'mas', rapm = None, decpm = None ):

    plt.clf()
    fig, ( ( ravt, decvt ), ( radec, ax4 ) ) = plt.subplots( 2, 2 )

    ravt.plot( tarr - 2400000.5, ra, 'r-', lw = 0.5 )
    if rapm != None: ravt.plot( tarr - 2400000.5, rapm, 'b:', lw = 0.5 )
    ravt.errorbar( obst - 2400000.5, obsra, yerr = sig, fmt = 'k.', ls = 'none' )
    ravt.set_xlabel( 'MJD (JD - 2400000.5)' )
    ravt.set_ylabel( '$\Delta$RA ('+ unit +')' )
    ravt.xaxis.set_label_position( 'top' )
    ravt.xaxis.set_ticks_position( 'top' )

    decvt.plot( tarr - 2400000.5, dec, 'r-', lw = 0.5 )
    if decpm != None: decvt.plot( tarr - 2400000.5, decpm, 'b:', lw = 0.5 )
    decvt.errorbar( obst - 2400000.5, obsdec, yerr = sig, fmt = 'k.', ls = 'none' )
    decvt.set_xlabel( 'MJD (JD - 2400000.5)' )
    decvt.set_ylabel( '$\Delta$Declination ('+ unit +')' )
    decvt.yaxis.set_label_position( 'right' )
    decvt.yaxis.set_ticks_position( 'right' )
    decvt.xaxis.set_label_position( 'top' )
    decvt.xaxis.set_ticks_position( 'top' )

    radec.plot( ra, dec, 'r-', lw = 0.5 )
    if rapm!= None: radec.plot( rapm, decpm, 'b:', lw = 0.5 )
    radec.errorbar( obsra, obsdec, xerr = sig, yerr = sig, fmt = 'k.', ls = 'none' )
    radec.set_xlabel( '$\Delta$RA ('+ unit +')' )
    radec.set_ylabel( '$\Delta$Declination ('+ unit +')' )

    ax4.axis( 'off' )

    fig.subplots_adjust( hspace = 0.1, wspace = 0.1 )
    
    plt.savefig( plotname ); plt.clf()

    return None

#############################################################################################################

########## APPLYING ORBIT PREDICTOR CLASS TO HD 80606 ##########
    
##### Input parameters for HD 80606 #####
a    = 0.4564 # AU
e    = 0.934
W    = np.radians( 160.98 ) # Rad
I    = np.radians( 89.232 ) # Rad
w    = np.radians( 300.77 ) # Rad
t0   = 2451973.72 / u.yr.to('d') # Year
M1   = 1.018 # Sol Mass
M2   = 4.114 * u.jupiterMass.to('solMass') # Sol Mass

# Boolean flags for whether to do each question
plot = True # Whether or not to plot orbit
doQ2 = True # RV Curve
doQ3 = False # Transit Prediction
doQ4 = True # Astrometric Prediction

# Get class instance for HD 80606 orbit
orbit = OrbitPredictor( a, e, W, I, w, t0, M1, M2 )

if plot: orbit.PlotOrbit() # Plot orbits

##### Question 2 - Radial Velocity #####

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
    plt.savefig( 'HD80606_StarRV.pdf' )

    # Now find mins and maxs of the radial velocity curve
    mins = signal.argrelmin( RVs )[0] # mins
    maxs = signal.argrelmax( RVs )[0] # maxs

    print '\nRV minima occur at: '
    print Time( tarr[mins], format = 'jd' ).iso

    print '\nRV maxima occur at: '
    print Time( tarr[maxs], format = 'jd' ).iso

##### Question 3 - Transits #####

if doQ3:

    def FindTransits( tarr, Rs, Rp ):
        # Takes tarr in years and radii in au

        # Get projected separation and Z component of orbit
        X, Y, Z = orbit.getXYZ( orbit.a, orbit.w, tarr )
        R       = np.sqrt( X ** 2.0 + Y ** 2.0 )

        # Look for when the projected sep is less than the stellar radius + planet radius (would account for grazing transits)
        # AND when the planet is in front of the star ( z > 0.0 )
        transit = np.where( ( R <= Rs + Rp ) & ( Z > 0.0 ) )[0]

        # Return the times at which there is a transit (re-convert to days)
        return tarr[transit] * u.yr.to('d')

    # Set dates to search for transits over in units of days (same as Q2)
    aug1 = Time( '2017-08-01 00:00:00', scale = 'utc' ).jd
    jan1 = Time( '2018-01-01 00:00:00', scale = 'utc' ).jd
    tarr = np.linspace( aug1, jan1, 100000 )

    Rs   = 1.037 * u.solRad.to('au')
    Rp   = 1.003 * u.jupiterRad.to('au')

    transits = FindTransits( tarr * u.d.to('yr'), Rs, Rp )

    if len(transits) == 0: print '\nNo transits predicted to occur in this time frame.\n'
    else:
        # Split the transit time array into each separate transit
        transits = np.split( transits, np.where( np.diff( transits ) > 10.0 * np.median( np.diff( transits ) ) )[0] + 1 )

        # Print out start, mid, and end time of transit in UTC
        for t in transits:
            print '\n'
            print Time( t.min(), format = 'jd' ).iso
            print Time( np.median(t), format = 'jd' ).iso
            print Time( t.max(), format = 'jd' ).iso

##### Question 4 - Astrometry #####

if doQ4:

    ra   = np.radians( 140.6569379644042 ) # RA from Gaia
    dec  = np.radians( 50.60377501745222 ) # Dec from Gaia
    p    = 15.3e-3 # arcsec from Gaia
    d    = 1 / p * u.pc.to('au')

    pma = 56.01 * u.mas.to('rad') * u.d.to('yr') # Convert pm from mas/yr to rad/day
    pmd = 10.64 * u.mas.to('rad') * u.d.to('yr') # Convert pm from mas/yr to rad/day

    # Set array of time over which to calculate astrometric motion in days
    aug1   = Time( '2017-08-01 00:00:00', scale = 'utc' ).jd
    aug5   = Time( '2022-08-01 00:00:00', scale = 'utc' ).jd
    tarr   = np.linspace( aug1, aug5, 100000 )
    p      = p * u.arcsec.to('rad')
    sigma  = 7.5e-3 # Gaia uncertainty mas

    obstarr = np.sort( np.random.uniform( aug1, aug5, 100 ) ) # Random tarr for observations

    # First just planet on star

    X, Y, Z = orbit.getXYZ( orbit.a1, orbit.ws, tarr * u.d.to('yr') )    # Get X, Y, Z in au
    X = X / d * u.rad.to('mas'); Y = Y / d * u.rad.to('mas') # Get X, Y in radians

    obsX, obsY, obsZ = orbit.getXYZ( orbit.a1, orbit.ws, obstarr * u.d.to('yr') )
    obsX = obsX / d * u.rad.to('mas'); obsY = obsY / d * u.rad.to('mas')
    obsX += np.random.normal( 0.0, sigma, 100 ); obsY += np.random.normal( 0.0, sigma, 100 )
    
    RADecPlots( tarr, -Y * 1e3, X * 1e3, obstarr, -obsY * 1e3, obsX * 1e3, sigma * 1e3, 'plt.pdf', '$\mu$as' )
    
    # Now planet plus parallax motion

    raplx, decplx = getPlxMotion( tarr, ra, dec, p )
    raplx *= u.rad.to('mas'); decplx *= u.rad.to('mas')
    rat = raplx - Y; dect = decplx + X

    obsraplx, obsdecplx = getPlxMotion( obstarr, ra, dec, p )
    obsraplx *= u.rad.to('mas'); obsdecplx *= u.rad.to('mas')
    obsrat = obsraplx - obsY; obsdect = obsdecplx + obsX

    RADecPlots( tarr, rat, dect, obstarr, obsrat, obsdect, sigma, 'pltplx.pdf' )

    # Now planet plus parallax plus proper motion

    rapmt  = pma * ( tarr - tarr[0] ) * u.rad.to('mas')
    decpmt = pmd * ( tarr - tarr[0] ) * u.rad.to('mas')
    rat += rapmt; dect += decpmt

    obsrapmt  = pma * ( obstarr - tarr[0] ) * u.rad.to('mas')
    obsdecpmt = pmd * ( obstarr - tarr[0] ) * u.rad.to('mas')
    obsrat += obsrapmt; obsdect += obsdecpmt

    RADecPlots( tarr, rat, dect, obstarr, obsrat, obsdect, sigma, 'pltplxpm.pdf', rapm = rapmt, decpm = decpmt )
