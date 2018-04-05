#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Energyfunctional for the LDDMM Setting of Image Registration
E(I0, m) = norm(v) + reg_param*(I0(phi) - I1)


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

__all__ = ('Energy',  )


class Energy(Operator):
    
    def __init__(self, Shooting, forwardOp, norm, data, space, kernel, reg_param):
        self.Shooting = Shooting
        self.forwardOp = forwardOp
        self.norm = norm
        self.data = data
        self.space = space
        self.kernel = kernel
        self.reg_param = reg_param
        self.attach = self.norm*(self.data - self.forwardOp)
        
        super().__init__(domain=ProductSpace(self.space, self.space.tangent_bundle),
                         range=norm.range,
                            linear=False)
        
        
    def _call(self, X):    
        
        norm_v = odl.solvers.L2NormSquared(X[1].space)
        
        Imv = self.Shooting(X)
        I1 = Imv[0]
        
        return self.reg_param/2 * self.attach(I1[-1]) + norm_v(X[1])/2
        
    
    @property
    def gradient(self):
        
        operator = self
        
  
        return EnergyGradient(operator)
    
class EnergyGradient(Operator):
    
    def __init__(self, operator):
        
        self.operator = operator
        
        # Kernel convolution v = K*m is done through Fouriertransform
        def fitting_kernel(space, kernel):
            
            kspace = ProductSpace(space, dim)
        
            # Create the array of kernel values on the grid points
            discretized_kernel = kspace.element(
                [space.element(kernel) for _ in range(dim)])
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
        dim = 2
        padded_size = 2 * operator.space.shape[0]
        padded_ft_fit_op = padded_ft_op(operator.space, padded_size)
        vectorial_ft_fit_op = DiagonalOperator(*([padded_ft_fit_op] * dim))
        
        discretized_kernel = fitting_kernel(operator.space, operator.kernel)
        ft_kernel_fitting = vectorial_ft_fit_op(discretized_kernel)
        
        self.vectorial_ft_fit_op = vectorial_ft_fit_op
        self.ft_kernel_fitting = ft_kernel_fitting
        
        super().__init__(domain=ProductSpace(operator.space, operator.space.tangent_bundle),
                 range=operator.space.tangent_bundle,
                    linear=False)
        
        
        
    def _call(self, X):
        """
        Compute Gradient of the Energy functional
        Gradient of regularization term norm(v): K*m
        Gradient of data term: mhat
        """

        dim=2
        
        Imv = self.operator.Shooting(X)
        I = Imv[0]
        m = Imv[1]
        v = Imv[2]
        N = I.shape[0] - 1
        
        mhat = m[0].space.zero()
        Ihat = self.operator.reg_param * self.operator.attach.gradient(I[-1])
        
        # Create the gradient op
        grad_op = Gradient(domain=self.operator.space, method='forward',pad_mode='constant', pad_const = 0)
        # Create the divergence op
        div_op = -grad_op.adjoint

        
        for i in range(N):
            gradmhat0 = grad_op(mhat[0])
            gradmhat1 = grad_op(mhat[1])
            gradmhat = [gradmhat0, gradmhat1]
            gradI = grad_op(I[N-i])
            coad0 = sum(grad_op(m[N-i][0])*mhat) + sum([gradmhat[j][0] * m[N-i][j] for j in range(dim)]) + div_op(mhat)*m[N-i][0]
            coad1 = sum(grad_op(m[N-i][1])*mhat) + sum([gradmhat[j][1] * m[N-i][j] for j in range(dim)]) + div_op(mhat)*m[N-i][1]
            coad_mhat_m = mhat.space.element([coad0, coad1])
            
            vhat = (2 * np.pi) ** (dim / 2.0) * self.vectorial_ft_fit_op.inverse(self.vectorial_ft_fit_op(-coad_mhat_m + Ihat * gradI) * self.ft_kernel_fitting)
            
            Ihat = Ihat + 1/N * div_op(Ihat * v[N-i])
            advm0 = sum(grad_op(v[N-i][0])*mhat) - sum(grad_op(mhat[0])*v[N-i])
            advm1 = sum(grad_op(v[N-i][1])*mhat) - sum(grad_op(mhat[1])*v[N-i])   
            advm = mhat.space.element([advm0, advm1])
            mhat = mhat - 1/N*(advm + vhat)
        
        Km = (2 * np.pi) ** (dim / 2.0) * self.vectorial_ft_fit_op.inverse(self.vectorial_ft_fit_op(m[0]) * self.ft_kernel_fitting)
        
        return Km + mhat