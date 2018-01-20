#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 10 10:36:03 2018

@author: rosa
"""


from future import standard_library
standard_library.install_aliases()
from builtins import super

import numpy as np

from odl.discr import Gradient
from odl.operator import Operator
from odl.space import ProductSpace
from odl.discr import ResizingOperator
from odl.operator import DiagonalOperator
from odl.trafos import FourierTransform
import odl

__all__ = ('Shooting',  )





class Shooting(Operator):
    
    def __init__(self, N, kernel, space):
        self.N = N  
        self.kernel = kernel
        self.space = space
        self.dim = space.ndim
    
        # Compute the FT of kernel in fitting term
        def fitting_kernel(space, kernel):
        
            kspace = ProductSpace(space, self.dim)
        
            # Create the array of kernel values on the grid points
            discretized_kernel = kspace.element(
                [space.element(kernel) for _ in range(self.dim)])
            return discretized_kernel  
        
    
        def padded_ft_op(space, padded_size):
            """Create zero-padding fft setting
            Parameters
            ----------
            space : the space needs to do FT
            padding_size : the percent for zero padding
            """
            padded_op = ResizingOperator(
                space, ran_shp=[padded_size for _ in range(space.ndim)])
            shifts = [not s % 2 for s in space.shape]
            ft_op = FourierTransform(
                padded_op.range, halfcomplex=False, shift=shifts)
            
            return ft_op * padded_op
    
         # FFT setting for data matching term, 1 means 100% padding
        padded_size = 2 * space.shape[0]
        padded_ft_fit_op = padded_ft_op(space, padded_size)
        vectorial_ft_fit_op = DiagonalOperator(*([padded_ft_fit_op] * self.dim))
        
        discretized_kernel = fitting_kernel(space, kernel)
        ft_kernel_fitting = vectorial_ft_fit_op(discretized_kernel)
        
        self.vectorial_ft_fit_op = vectorial_ft_fit_op
        self.ft_kernel_fitting = ft_kernel_fitting
        

        super().__init__(domain=ProductSpace(space,space.tangent_bundle),
                         range=ProductSpace(ProductSpace(space,self.N+1), ProductSpace(space.tangent_bundle, self.N+1), ProductSpace(space.tangent_bundle, self.N+1)),
                            linear=False)
        
        
    def _call(self,X):
        """ Shooting equations and Integration
        
        Args: X = [I0, m] with I0 template image and m vectorvalued momentum
        
        Returns: 
                I: deformed Image over time (N+1 timesteps)
                M: vectorvalued momentum in eulerian coordinates over time
                V: vectorfields over time
        """
        template = X[0].copy()
        m0 = X[1].copy()
        
        # create spaces for time-dependent Image and vectorfields
        series_image_space_integration = ProductSpace(template.space,
                                                      self.N+1)
        series_vector_space_integration = ProductSpace(m0.space, self.N+1)
      
        inv_N=1/self.N
        I=series_image_space_integration.element()
        I[0]=template.copy()
        M = series_vector_space_integration.element()
        M[0] = m0.copy()
        U = series_vector_space_integration.element()
        
        # Create the gradient op
        grad_op = Gradient(domain=self.space, method='forward',
                           pad_mode='symmetric')
        # Create the divergence op
        div_op = -grad_op.adjoint
        
        
        for i in range(self.N):
            m = M[i]
            
            # Kernel convolution to obtain vectorfields from momentum
            # v = K*m
            U[i] = (2 * np.pi) ** (self.dim / 2.0) * self.vectorial_ft_fit_op.inverse(self.vectorial_ft_fit_op(m) * self.ft_kernel_fitting)
            u = U[i]
            
            # shooting equation
            # d/dt m = - ad*_v m 
            gradu = [grad_op(u[0]), grad_op(u[1])]
            dtM0 = sum(grad_op(m[0])*u) + sum([gradu[i][0] * m[i] for i in range(self.dim)]) + div_op(u)*m[0]
            dtM1 = sum(grad_op(m[1])*u) + sum([gradu[i][1] * m[i] for i in range(self.dim)]) + div_op(u)*m[1]
            
            # Integration step
            M[i+1] = M[i] - inv_N * m0.space.element([dtM0, dtM1])
            I[i+1] = odl.deform.linear_deform(I[i], - inv_N * u)
            
        U[self.N] = (2 * np.pi) ** (self.dim / 2.0) * self.vectorial_ft_fit_op.inverse(self.vectorial_ft_fit_op(M[self.N]) * self.ft_kernel_fitting)

        return I,M,U
