import numpy
from matplotlib import pyplot
from scipy import integrate

class Panel:
    def __init__(self, xa, ya, xb, yb):
        self.xa, self.ya = xa, ya
        self.xb, self.yb = xb, yb
        self.xc, self.yc = (xa+xb)/2, (ya+yb)/2
        self.length = numpy.sqrt((xb-xa)**2+(yb-ya)**2)
        if xb-xa <= 0.:
            self.beta = numpy.arccos((yb-ya)/self.length)
        elif xb-xa > 0.:
            self.beta = numpy.pi + numpy.arccos(-(yb-ya)/self.length)

        # project to normal and tangential direction
        self.nx = numpy.cos(self.beta)
        self.ny = numpy.sin(self.beta)
        self.tx = -numpy.sin(self.beta)
        self.ty = numpy.cos(self.beta)

        if self.beta <= numpy.pi:
            self.loc = 'extrados'
        else:
            self.loc = 'intrados'
        
        self.sigma = 0.
        self.vt = 0.
        self.cp = 0.




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


def integral(p_i, panel, a, b):
    """Evaluates the contribution of a panel at one point.
    
    Arguments
    ---------
    p_i -- the target panel
    x, y -- Cartesian coordinates of the point.
    panel -- panel which contribution is evaluated.
    
    Returns
    -------
    Integral over the panel of the influence at one point.
    """
    def f(s):
        return ( ((p_i.xc - (panel.xa - numpy.sin(panel.beta)*s))*a 
                  + (p_i.yc - (panel.ya + numpy.cos(panel.beta)*s))*b)
                / ((p_i.xc - (panel.xa - numpy.sin(panel.beta)*s))**2 
                   + (p_i.yc - (panel.ya + numpy.cos(panel.beta)*s))**2) )
    return integrate.quad(lambda s:f(s), 0., panel.length)[0]
