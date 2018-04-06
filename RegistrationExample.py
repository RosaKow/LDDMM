#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 11 17:00:16 2018

@author: rosa
"""

import numpy as np
from matplotlib import pylab as plt
import odl
from odl.discr import ResizingOperator
from future import standard_library
standard_library.install_aliases()
import ShootingOperator_vectormomentum as so
import EnergyFunctional



def loadimages(images):
    if images == 'jv':
        # J to V
        I1name = 'v.png'
        I0name = 'j.png'
        I0 = np.rot90(plt.imread(I0name).astype('float'), -1)
        I1 = np.rot90(plt.imread(I1name).astype('float'), -1)
        ran = (100,100)
        
    elif images == 'shapes':
        I0 = 1- plt.imread('shape1.jpeg').astype('float')[:,:,0]/255
        I1 = 1- plt.imread('shape2.jpeg').astype('float')[:,:,0]/255
        ran = (150,150)
        
    elif images == 'translation':
        I0 = np.zeros((100,100))
        I1 = I0.copy()
        I0[30:50,30:55] = 1
        I1[30:50,40:65] = 1
        ran = (100,100)
        
    elif images == 'square':
        I0 = np.zeros((100,100))
        I1 = I0.copy()
        I0[40:60,40:60] = 1
        I1[40:65,40:75] = 1
        ran = (100,100)
            
    elif images == 'shepp-logan':
        # Shepp-Logan
        rec_space = odl.uniform_discr(
            min_pt=[-16,-16], max_pt=[16,16], shape=[100, 100],
            dtype='float32', interp='linear')
        # Create the template as the deformed Shepp-Logan phantom
        I0 = odl.phantom.transmission.shepp_logan(rec_space, modified=True)
        # Create the template for Shepp-Logan phantom
        deform_field_space = rec_space.tangent_bundle
        disp_func = [
            lambda x: 16.0 * np.sin(0.2*np.pi * x[0] / 10.0),
            lambda x: 1*16.0 * np.sin(np.pi * x[1] / 16.0)]
        deform_field = deform_field_space.element(disp_func)
        rec_space.element(odl.deform.linear_deform(I0, 0.2*deform_field)).show()
    
        I1 = rec_space.element(
            odl.deform.linear_deform(I0, 0.2*deform_field))
        ran = (150,150)
    
    space = odl.uniform_discr(
            min_pt=[-16, -16], max_pt=[16, 16], shape=I0.shape,
            dtype='float32', interp='linear')
    
    # The images need to be resized in case the deformation will be too big and exceed the image boundary
    resize_op = ResizingOperator(space, ran_shp=ran)
    I0 = resize_op(I0)
    I1 = resize_op(I1)

    return I0, I1


def kernel(x):
    scaled = [xi ** 2 / (2 * sigma ** 2) for xi in x]
    return np.exp(-sum(scaled))


def DefForwardOp(space, operator='Raytransform', num_angles=100):
    
    if operator == 'Raytransform':
        # Create 2-D parallel projection geometry
        angle_partition = odl.uniform_partition(0.0, np.pi, num_angles,
                                            nodes_on_bdry=[(True, True)])
        # Create 2-D projection domain
        # The length should be 1.5 times of that of the reconstruction space
        detector_partition = odl.uniform_partition(-24, 24, int(round(space.shape[0]*np.sqrt(2))))
        # Create 2-D parallel projection geometry
        geometry = odl.tomo.Parallel2dGeometry(angle_partition, detector_partition)
        
        # Ray transform aka forward projection. We use ASTRA CUDA backend.
        forwardOp = odl.tomo.RayTransform(space, geometry, impl='astra_cpu')
    elif operator == 'Identity':
        forwardOp = odl.IdentityOperator(space)
    return forwardOp


def armijo(E, gradE):
    """ Armijo Stepsize Calculation 
    Args: E: functionvalue at current x
        gradE: gradient at current x
    Returns: alpha: stepsize
    """
    alpha = 1
    p = 0.8
    c1=10**-4
    norm = odl.solvers.L2NormSquared(X[1].space)
    while energy([X[0],X[1] - alpha*gradE]) > E - c1*alpha*norm(gradE):
        alpha *= p
    return alpha

def gill_murray_wright(E, Enew, gradE,X, Xnew, k, kmax = 1000):
    """ Convergence Criteria of Gill, Murray, Wright """
    tau = 10**-8
    eps = 10**-8
    norm = odl.solvers.L2NormSquared(X[1].space)
    cond0 = E < Enew
    cond1 = E - Enew < tau*(1 + E)
    cond2 = norm(Xnew[1] - X[1])<np.sqrt(tau)*(1 + (norm(X[1]))**(1/2))
    cond3 = norm(gradE) < tau**(1/3)*(1 + E)
    cond4 = norm(gradE)<eps
    cond5 = k > kmax
    if (cond1 and cond2 and cond3) or cond4 or cond5 or cond0:
        return True
    return False

def gradientdescent(X):
    
    k = 0
    convergence = False
    
    while convergence == False:
        gradE = energy.gradient(X)    
        E = energy(X)
        alpha = 0.00001 #armijo(E, gradE)
        Xnew = [X[0], X[1] - alpha*gradE]
        print(" iter : {}  ,  attachment term : {}".format(k,E))
        convergence = gill_murray_wright(E, energy(Xnew), gradE, X, Xnew, k)
        X = Xnew
        k+=1
    return X

        
def visualize_momentum(m,step=5):
    
    m = m.__array__()
    
    u = m[0,0:-1:step, 0:-1:step]
    v = m[1,0:-1:step,0:-1:step]
    x = np.linspace(0,1,u.shape[0])
    y = np.linspace(0,1,u.shape[1])
    
    plt.quiver(x,y,u,v, scale=1)

if __name__ == '__main__':
    
    #load and prepare data
    I0, I1 = loadimages('shapes')    
    space = I0.space
    I1_noise = odl.phantom.noise.salt_pepper_noise(I1)
    
    forwardOp = DefForwardOp(space)
    data = forwardOp(I1)
    
    norm = odl.solvers.L2NormSquared(forwardOp.range)
    
    # set parameters
    N = 10      #number of integration steps
    reg_param = 5# regularisation parameter
    sigma = 5   # gaussian convolution kernel parameter
    
    # define energy functional
    shoot = so.Shooting(N, kernel, space)
    energy = EnergyFunctional.Energy(shoot, forwardOp, norm, data, space, kernel, reg_param)
    
    # do registration
    m = space.tangent_bundle.zero()
    X = [I0, m]
    X = gradientdescent(X)
    
    # visualisation
    I0.show('template image')
    I1.show('reference image')
    (I1-shoot(X)[0][-1]).show('difference between deformed template / reference')        
    shoot(X)[0][-1].show('deformed image')
    visualize_momentum(X[1], step=5)

def savedata():
    import scipy.io as io
    a = {}
    a['I0'] = X[0].__array__()
    a['m'] = X[1].__array__()
    a['I1'] = I1.__array__()
    a['Iresult'] = shoot(X)[0][-1].__array__()
    a['ShootingOperator'] = 'vectormomentum'
    a['ForwardOperator'] = 'Raytransform 100angles'
    io.savemat('JtoV', a)

