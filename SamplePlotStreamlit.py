#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 25 09:37:52 2025

@author: luiszerpa
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt




def objectiveFunction(x,iF):
    
    # Branin-Hoo
    if iF == 1:
        f = (x[1]-(5.1/(4*(np.pi)**2))*x[0]**2+ 5*x[0]/np.pi - 6)**2 + 10*(1-(1/(8*np.pi)))*np.cos(x[0])+10
    
    # Six-Hump Camel
    elif iF == 2:
        f = (4 - 2.1*x[0]**2 +(4*x[0]**4)/3)*x[0]**2 + x[0]*x[1] + (-4+4*x[1]**2)*x[1]**2
    
    # Rosenbrock Banana
    elif iF == 3:
        f = 1 + 100*(x[1] - x[0]**2)**2 + (1 - x[0]**2)
    
    # Hock-Schittkowski 5
    elif iF == 4:
        f = np.sin(x[0] + x[1]) + (x[0] - x[1])**2 - 1.5*x[0] + 2.5*x[1] + 1
    
    # Example class notes
    elif iF == 5:
        f =  -20 + 3*x[0]**2 + x[1]**2
    
    else:
        print('error: function not defined')
    
    return f

def Gradient(fun,x,iF):
    dx = x*1e-3
    xi = np.zeros(np.size(x))
    xj = np.zeros(np.size(x))
    vectorGradient = np.zeros(np.size(x))
    for i in range (0,np.size(x)):
        if dx[i] == 0:
            dx[i] = x[i]+1e-12
    
    for i in range(0,np.size(x)):
        xi = x.copy()
        xj = x.copy()
        xi[i] = x[i]+dx[i]
        xj[i] = x[i]-dx[i]
        vectorGradient[i] = (fun(xi,iF) - fun(xj,iF))/(2*dx[i])
    
    return vectorGradient

def Hessian(fun,x,iF):
    dx = x*1e-3
    xp = np.zeros(np.size(x))
    xn = np.zeros(np.size(x))
    xipp = np.zeros(np.size(x))
    xipn = np.zeros(np.size(x))
    xjnp = np.zeros(np.size(x))
    xjnn = np.zeros(np.size(x))
    matrizHessian = np.zeros((np.size(x),np.size(x)))
    for i in range(0,np.size(x)):
        for j in range (0,np.size(x)):
            if i == j:
                xp = x.copy()
                xn = x.copy()
                xp[i] = x[i]+dx[i]
                xn[j] = x[i]-dx[i]
                matrizHessian[i,j] = (fun(xp,iF) - 2*fun(x,iF)+ fun(xn,iF))/(dx[i]**2+1e-12)
            else:
                xipp = x.copy()
                xipn = x.copy()
                xjnp = x.copy()
                xjnn = x.copy()
                xipp[i] = x[i]+dx[i]
                xipp[j] = x[j]+dx[j]
                xipn[i] = x[i]+dx[i]
                xipn[j] = x[j]-dx[j]
                xjnp[i] = x[i]-dx[i]        
                xjnp[j] = x[j]+dx[j]
                xjnn[i] = x[i]-dx[i]         
                xjnn[j] = x[j]-dx[j]
                matrizHessian[i,j] = (fun(xipp,iF) - fun(xipn,iF) - fun(xjnp,iF) + fun(xjnn,iF))/(4*dx[i]*dx[j]+1e-12)
    return matrizHessian

def SteepestDescent(fun,x0,iF,tolerance,ax):
    maxIter = 100
    
    f0 = fun(x0,iF)
    # Descent direction
    gradient0 = Gradient(fun,x0,iF)
    descentDirection = gradient0/np.linalg.norm(gradient0)
    
    iteration = 0
    criterioParada = 0
    
    while criterioParada == 0:
        # Line Search method - Newton
        dpx = np.dot(Gradient(fun,x0,iF),descentDirection)
        ddpx = np.inner(np.inner(descentDirection,Hessian(fun,x0,iF)),descentDirection)
        xLineSearch = x0 - dpx/ddpx*descentDirection
        # xLineSearch = x0 - 0.01*dpx*descentDirection
        fLineSearch = fun(xLineSearch,iF)
                
        if (np.linalg.norm(x0 - xLineSearch) < tolerance or abs(f0-fLineSearch) < tolerance or \
            np.linalg.norm(gradient0) < tolerance or iteration > maxIter):
            criterioParada = 1
        else:
            criterioParada = 0

        ax.plot(([x0[0],xLineSearch[0]]),([x0[1],xLineSearch[1]]),marker='o')
        x0 = xLineSearch.copy()
        f0 = fLineSearch.copy()
        gradient0 = Gradient(fun,x0,iF)
        descentDirection = gradient0/np.linalg.norm(gradient0)
        iteration = iteration + 1
        
   
    return x0, f0, iteration


st.title("Function Optimization - Steepest Descent")

x0 = np.zeros((2))
x0[0] = st.slider("x", -5.0, 10.0, 2.5)
x0[1] = st.slider("y", 0.0, 15.0, 7.5)


''' Plotting the Branin-Hoo function'''

x = np.linspace(-5, 10, 100)
y = np.linspace(0, 15, 100)

xx, yy = np.meshgrid(x, y)
iF = 1
zz = objectiveFunction(([xx, yy]),iF)


fig, ax = plt.subplots(figsize=(6, 6))  # Create a figure containing a single axes.
ax.contour(xx,yy,zz,levels=50) # Plot data on the axes.
ax.plot(x0[0],x0[1],marker='o',color='b')
ax.set_xlabel('Variable x',fontsize='12',fontweight='bold', fontname='Verdana')
ax.set_ylabel('Variable y',fontsize='12',fontweight='bold',fontname='Verdana')
ax.tick_params(direction='in',width=1.5,length=8.0, labelsize=12.0)
# ax.set(xlim=(-5, 10), ylim=(0, 15))
for tick in ax.get_xticklabels():
    tick.set_fontname('Verdana')
for tick in ax.get_yticklabels():
    tick.set_fontname('Verdana')
ax.grid(True)  

st.pyplot(fig)

tolerance = 1e-12
x0, f0, interations = SteepestDescent(objectiveFunction,x0,iF,tolerance,ax)

st.write("Function value", f0)
st.write("Optimum variable", x0)
ax.plot(x0[0],x0[1],marker='o',color='r')

st.pyplot(fig)