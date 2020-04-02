"""
Copyright (C) 2018 Michele Ginesi

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.    See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.    If not, see <http://www.gnu.org/licenses/>.
"""

import numpy as np

class Obstacle_Ellipse():

    """
    Implementation of an obstacle for Dynamic Movement Primitives written as a
    general n-ellipsoid
      / x - x_c \ 2n     / y - y_c \ 2n      / z - z_c \ 2n
      |---------|     +  |---------|     +   |---------|     =  1
      \    a    /        \    b    /         \    c    /
    """

    def __init__(self, n_dim = 2, n = 1, center = np.zeros(2), axis = np.ones(2),
            **kwargs):
        """
        n_dim int     : dimension of the space (usually 2 or 3)
        n int         : order of the ellipsoid
        center float  : array containing the coordinates of the center of the ellipsoid
        axis float    : array containing the lenghts of the ais of the ellipsoid
        """
        if ((np.shape(center)[0] != n_dim) or (np.shape(axis)[0] != n_dim)):
            raise ValueError ("The dimensions of center or axis are not compatible with n_dim")
        else:
            self.n_dim = n_dim
            self.n = n
            self.center = center
            self.axis = axis
        return



    def compute_forcing_term(self, x, A = 1., eta = 1.):
        """
        Compute the forcing term generated by the potential U as
          \varphi(x) = -\nabla U
        """
        phi = np.zeros(self.n_dim)
        for i in range(self.n_dim):
            phi[i] = (((x[i] - self.center[i]) ** (2 * self.n - 1)) /
                (self.axis[i] ** (2 * self.n)))
        K = self.compute_isopotential(x)

        phi *= (A * np.exp(-eta*K)) * (eta / K + 1. / K / K) * (2 * self.n)
        return phi

    def compute_potential(self, x, A = 1., eta = 1.):
        """
        Compute the potential
                 exp (-\eta K)
          U = A ---------------
                       K
        """
        K = self.compute_isopotential (x)
        U = A * np.exp(-eta * K) / K
        return U

    def compute_isopotential(self, x):
        """
        Compute the isopotential of the obstacle
                / x - x_c \ 2n     / y - y_c \ 2n      / z - z_c \ 2n
          K  =  |---------|     +  |---------|     +   |---------|     -  1
                \    a    /        \    b    /         \    c    /
        """
        K = 0.
        for i in range(self.n_dim):
            K += ((x[i] - self.center[i]) / self.axis[i]) ** (2 * self.n)
        K -= 1
        return K