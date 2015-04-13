import numpy
from scipy import integrate
from matplotlib import pyplot, rcParams
rcParams['font.family'] = 'StixGeneral'

size = 10
xl = -0.5
xr = 2
ar = 0.20

class Panel:
    """Contains information related to one panel."""
    def __init__(self, xa, ya, xb, yb):
        """Creates a panel.
        
        Arguments
        ---------
        xa, ya -- Cartesian coordinates of the first end-point.
        xb, yb -- Cartesian coordinates of the second end-point.
        """
        self.xa, self.ya = xa, ya
        self.xb, self.yb = xb, yb
        
        self.xc, self.yc = (xa+xb)/2, (ya+yb)/2       # control-point (center-point)
        self.length = numpy.sqrt((xb-xa)**2+(yb-ya)**2)     # length of the panel
        
        # orientation of the panel (angle between x-axis and panel's normal)
        if xb-xa <= 0.:
            self.beta = numpy.arccos((yb-ya)/self.length)
        elif xb-xa > 0.:
            self.beta = numpy.pi + numpy.arccos(-(yb-ya)/self.length)
        
        # location of the panel
        if self.beta <= numpy.pi:
            self.loc = 'extrados'
        else:
            self.loc = 'intrados'
        
        self.sigma = 0.   # source strength
        self.vt = 0.      # tangential velocity
        self.cp = 0.      # pressure coefficient


class Freestream:
    """Freestream conditions."""
    def __init__(self, u_inf=1.0, alpha=0.0):
        """Sets the freestream conditions.
        
        Arguments
        ---------
        u_inf -- Farfield speed (default 1.0).
        alpha -- Angle of attack in degrees (default 0.0).
        """
        self.u_inf = u_inf
        self.alpha = alpha*numpy.pi/180    # degrees --> radians



def gen_panels(x, y, N=40):
    """Discretizes the geometry into panels using 'cosine' method.
    
    Arguments
    ---------
    x, y -- Cartesian coordinates of the geometry (1D arrays).
    N - number of panels (default 40).
    
    Returns
    -------
    panels -- Numpy array of panels.
    """
    R = (x.max()-x.min())/2     # radius of the circle
    x_center = (x.max()+x.min())/2    # x-coord of the center
    x_circle = x_center + R*numpy.cos(numpy.linspace(0, 2*numpy.pi, N+1))  # x-coord of the circle points
    
    x_ends = numpy.copy(x_circle)      # projection of the x-coord on the surface
    y_ends = numpy.empty_like(x_ends)  # initialization of the y-coord Numpy array

    x, y = numpy.append(x, x[0]), numpy.append(y, y[0])    # extend arrays using numpy.append
    
    # computes the y-coordinate of end-points
    I = 0
    for i in xrange(N):
        while I < len(x)-1:
            if (x[I] <= x_ends[i] <= x[I+1]) or (x[I+1] <= x_ends[i] <= x[I]):
                break
            else:
                I += 1
        a = (y[I+1]-y[I])/(x[I+1]-x[I])
        b = y[I+1] - a*x[I+1]
        y_ends[i] = a*x_ends[i] + b
    y_ends[N] = y_ends[0]
    
    panels = numpy.empty(N, dtype=object)
    for i in xrange(N):
        panels[i] = Panel(x_ends[i], y_ends[i], x_ends[i+1], y_ends[i+1])
    
    return panels


def integral(x, y, panel, dxdz, dydz):
    """Evaluates the contribution of a panel at one point.
    
    Arguments
    ---------
    x, y -- Cartesian coordinates of the point.
    panel -- panel which contribution is evaluated.
    dxdz -- derivative of x in the z-direction.
    dydz -- derivative of y in the z-direction.
    
    Returns
    -------
    Integral over the panel of the influence at one point.
    """
    def func(s):
        return ( ((x - (panel.xa - numpy.sin(panel.beta)*s))*dxdz 
                  + (y - (panel.ya + numpy.cos(panel.beta)*s))*dydz)
                / ((x - (panel.xa - numpy.sin(panel.beta)*s))**2 
                   + (y - (panel.ya + numpy.cos(panel.beta)*s))**2) )
    return integrate.quad(lambda s:func(s), 0., panel.length)[0]


def plot_panels(panels, figname):
	"""Plot the panels."""
	pyplot.figure(figsize=(size, size*ar))
	pyplot.grid(True)
	pyplot.axis('equal')
	pyplot.xlabel(r'$x$')
	pyplot.ylabel(r'$y$')
	pyplot.title(figname)
	pyplot.xlim(xl, xr)
	pyplot.plot([panel.xa for panel in panels], [panel.ya for panel in panels], 'g-')
	pyplot.plot([panel.xa for panel in panels], [panel.ya for panel in panels], 'go', markersize=5.0, alpha=0.5)
	pyplot.plot([panel.xc for panel in panels], [panel.yc for panel in panels], 'rx', markersize=4.0);