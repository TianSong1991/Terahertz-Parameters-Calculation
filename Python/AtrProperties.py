# -*- coding: utf-8 -*-
#此文档由王博开发
import numpy as np
from ReflectivityUtil import kkPhaseShift, refracIndexS, refracIndexP, reflectivityS, reflectivityS, reflectivityP
from FrequencyAnalysisUtil import concatenateTimeSeries, calRangeOfFrequency


class AtrProperties:
    """
    This class is to calculate the complex refractive index for ATR lens. @see ReflectionProperties fro detail.

    Attributes:
        nSamp: the refractive index of the sample solution, i.e., the solution is water based, nSamp = 1.33.

    @author: Bo Wang
    @file: AtrProperties.py
    @time: 2021/12/2 22:40
    """
    def __init__(self, time, refTime, sampTime, nSamp, lowFreq = 0.25, highFreq = 2, polarity = 0):
        if polarity not in [0, 1]:
            raise ValueError("The polarity should be 0 or 1.")

        self.time = time
        self.sampTime = sampTime
        self.refTime = refTime
        self.lowFreq = lowFreq
        self.highFreq = highFreq
        self.polarity = polarity
        self.nSamp = nSamp
        self.nSi = 3.45

        # This is the incident angle of Quenda's ATR prism.
        self.theta = 51.59 / 180 * np.pi

    def refracIndex(self):
        # Concatenate to 100 ps
        concatT, concatRef = concatenateTimeSeries(self.time, self.refTime, 100)
        freq, _ = calRangeOfFrequency(concatT, len(concatT), False)
        refFreq = np.fft.fft(concatRef)

        # Concatenate to 100 ps
        _, concatSamp = concatenateTimeSeries(self.time, self.sampTime, 100)
        sampFreq = np.fft.fft(concatSamp)

        # Calculate the transfer function
        h = sampFreq / refFreq
        nAir = 1

        # Eliminate the reference reflectivity from the transfer function
        if self.polarity == 0:
            h *= reflectivityS(self.theta, self.nSi, nAir)
        else:
            h *= reflectivityP(self.theta, self.nSi, nAir)

        freq = np.array(freq)
        indexLimits = (freq >= self.lowFreq) & (freq <= self.highFreq)
        h = h[indexLimits]
        freq = freq[indexLimits]
        self.freq = freq

        # Reconstruct the reflectivity
        r = self._reflectivity(freq, h)

        if self.polarity == 0:
            return refracIndexS(self.theta, self.nSi, r)
        else:
            return refracIndexP(self.theta, self.nSi, r)

    def _reflectivity(self, freq, h):
        """Calculate the reflectivity from the scale of the transfer function.
        Args:
            freq: the frequency of the transfer function.
            h: the transfer function.
        """
        # The transfer function is equivalent ot reflectivity, but its phase is not stable enough to estimate. Here, we
        # use theory cited from Yamamoto 1994 to calculate the phase and hence construct the reflectivity.
        r = abs(h)

        # The phase shift induced by absorption, derived by Kramer-Kronig relation with the scale of reflectivity.
        phaseShift = kkPhaseShift(freq, r)

        # The total phase shift equals the Kramer-Kronig phase shift reduced by pi
        phaseShift += self._correctionAngle()

        # The reconstructed reflectivity
        reflectivity = r * np.exp(1j * phaseShift)

        return reflectivity

    def _correctionAngle(self):
        """The phase shift induced by the total internal reflection. Cite from Yamamoto 1994."""

        # The refractive index of silicon
        corrAngle = np.sqrt(self.nSi ** 2 * np.sin(self.theta) ** 2 - self.nSamp ** 2)
        corrAngle /= self.nSi * np.cos(self.theta)
        corrAngle = 2 * np.arctan(corrAngle) - np.pi

        return corrAngle