# Geubelle research group
# Authors:  Qibang Liu (qibang@illinois.edu)
#           Michael Zakoworotny (mjz7@illinois.edu)
#           Philippe Geubelle (geubelle@illinois.edu)
#           Aditya Kumar (aditya.kumar@ce.gatech.edu)
#
# Contains a library of commonly used cure kinetics functions
# 

from abc import ABC, abstractmethod

class Kinetics_Model(ABC):
    """
    Base class for cure kinetics models
    """
    def __init__(self):
        super().__init__()

    @abstractmethod
    def eval(self, alpha):
        pass


class First_Order(Kinetics_Model):
    """
    First order reaction model of the form:

    dalpha/dt = A*exp(-E/R*T)*g(alpha)
    g(alpha) = (1 - alpha)
    """

    def __init__(self):
        """
        No parameters needed
        """
        super().__init__()

    def eval(self, alpha):
        """
        Evaluate cure kinetics

        Parameters
        ----------
        alpha - dolfin function
            The function with values of alpha to evalue g(alpha)
        """
        g = 1 - alpha
        return g

class Nth_Order(Kinetics_Model):
    """
    nth order reaction model of the form:

    dalpha/dt = A*exp(-E/R*T)*g(alpha)
    g(alpha) = (1 - alpha)^n
    """

    def __init__(self, n):
        """
        Parameters
        ----------
        n - float
            Order of reaction
        """
        self.n = n
        super().__init__()

    def eval(self, alpha):
        """
        Evaluate cure kinetics

        Parameters
        ----------
        alpha - dolfin function
            The function with values of alpha to evalue g(alpha)
        """
        g = (1 - alpha)**self.n
        return g

class Prout_Tompkins(Kinetics_Model):
    """
    Prout Tompkins reaction model of the form:

    dalpha/dt = A*exp(-E/R*T)*g(alpha)
    g(alpha) = (1 - alpha)^n * alpha^m
    """

    def __init__(self, n, m):
        """
        Parameters
        ----------
        n - float
            Order of reaction
        m - float
            Order of reaction
        """
        self.n = n
        self.m = m
        super().__init__()
        
    def eval(self, alpha):
        """
        Evaluate cure kinetics

        Parameters
        ----------
        alpha - dolfin function
            The function with values of alpha to evalue g(alpha)
        """
        g = (1 - alpha)**self.n * alpha**self.m
        return g

class Prout_Tompkins_Diffusion(Kinetics_Model):
    """
    Prout Tompkins with diffusion term reaction model of the form:

    dalpha/dt = A*exp(-E/R*T)*g(alpha)
    g(alpha) = (1 - alpha)^n * alpha^m * 1/exp(Ca*(alpha - alpha_c))
    """

    def __init__(self, n, m, Ca, alpha_c):
        """
        Parameters
        ----------
        n - float
            Order of reaction
        m - float
            Order of reaction
        Ca - float
            Diffusion constant
        alpha_c - float
            Critical conversion at onset of diffusion dominance
        """
        self.n = n
        self.m = m
        self.Ca = Ca
        self.alpha_c = alpha_c
        super().__init__()
        
    def eval(self, alpha):
        """
        Evaluate cure kinetics

        Parameters
        ----------
        alpha - dolfin function
            The function with values of alpha to evalue g(alpha)
        """
        from ufl import exp
        g = (1 - alpha)**self.n * alpha**self.m * 1/(1 + exp(self.Ca*(alpha - self.alpha_c)))
        return g
