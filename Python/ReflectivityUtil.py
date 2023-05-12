# -*- coding: utf-8 -*-
"""
This utility class contains all the method related to estimating the dielectric properties from the reflected THz beam.

@author: Bo Wang
@file: ReflectivityUtil.py
@time: 2021/12/2 17:52
"""
#此文件由王博开发，主要计算反射式吸收系数与折射率
import numpy as np
from numpy.lib.scimath import sqrt as csqrt
from PIL import  Image
def kkPhaseShift(freq, rScale):
    """Calculate the phase shift caused by the absorption using the Kramers-Kronig relation.

    Args:
        freq: the range of frequency
        rScale: the scale of reflectivity corresponding to freq
    """
    phaseShift = []
    length = freq.shape[0]
    freqStep = freq[1] - freq[0]
    
    # Cauchy principle integral

    for i in range(length):
        # Calculate the integrand of Cauchy principle integral, avoid the peculiar point at w=wi.
        integrands = np.log(rScale[rScale != rScale[i]]) - np.log(rScale[i])
        integrands /= freq[rScale != rScale[i]] ** 2 - freq[i] ** 2
        integrands *= freqStep
    
        # Cite from Yamamoto 1994.
        currPhaseShift = -np.sum(integrands) * 2 * freq[i] / np.pi
        phaseShift.append(currPhaseShift)

    return np.array(phaseShift)

def permittivityReal(n, k):
    """The real part of the dielectric permittivity."""
    return np.power(n, 2) - np.power(k, 2)

def permittivityImag(n, k):
    """The imaginary part of the dielectric permittivity, representing absorption."""
    return 2 * n * k

def refracIndexS(theta, nIn, r):
    """Calculate the complex refractive index of S polarization.

    Args:
        theta: the incidence angle
        nIn: the refractive of the incident substance.
        r: the reflectivity.
    """
    refracIndexS = np.power(1 - r, 2) * (np.cos(theta) ** 2) / np.power(1 + r, 2)
    refracIndexS += np.sin(theta) ** 2
    refracIndexS = np.sqrt(refracIndexS)
    refracIndexS *= nIn

    return refracIndexS

def refracIndexP(theta, nIn, r):
    """Assume the refractive index of air is 1, calculate the complex refractive index of S polarization.

    Args:
        theta: the incidence angle.
        nIn: the refractive of the incident substance.
        r: the reflectivity.
    """
    factor = np.power(1 + r, 2)  * nIn ** 2 / np.power(1 - r, 2)
    refracIndexP = factor + np.sqrt(np.power(factor, 2) - factor * nIn ** 2 * np.sin(2 * theta) ** 2)
    refracIndexP /= 2 * np.cos(theta) ** 2
    refracIndexP = np.sqrt(refracIndexP)

    return refracIndexP

def reflectivityS(theta, nIn, nOut):
    """Calculate the reflectivity of S polarization according to Fresnell law.

    Args:
        theta: the incidence angle.
        nIn: the refractive index of the incident substance.
        nOut: the refractive index of the refractive substance.
    """
    refReflect = nIn * np.cos(theta) - csqrt(nOut ** 2 - nIn ** 2 * np.sin(theta) ** 2)
    refReflect /=  nIn * np.cos(theta) + csqrt(nOut ** 2 - nIn ** 2 * np.sin(theta) ** 2)

    return refReflect

def reflectivityP(theta, nIn, nOut):
    """Calculate the reflectivity of P polarization according to Fresnell law."""
    refReflect = nOut ** 2 * np.cos(theta) - nIn * csqrt(nOut ** 2 - nIn ** 2 * np.sin(theta) ** 2)
    refReflect /= nOut ** 2 * np.cos(theta) + nIn * csqrt(nOut ** 2 - nIn ** 2 * np.sin(theta) ** 2)

    return refReflect