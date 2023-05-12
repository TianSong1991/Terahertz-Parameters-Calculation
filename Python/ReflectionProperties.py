# -*- coding: utf-8 -*-
import numpy as np
from ReflectivityUtil import kkPhaseShift, refracIndexS, refracIndexP
from FrequencyAnalysisUtil import concatenateTimeSeries, calRangeOfFrequency
#此文件由王博开发，主要计算反射式吸收系数与折射率

class ReflectionProperties:
    """
    This class is to calculate the complex refractive index of a substance by measuring the reflective THz beam from the
    interface between the sample and the air.

    Attributes:
        sampTime: the time signal of the beam reflected from the sample.
        refTime: the time signal of the beam reflected from the metal mirror.
        time: the time during of the signal. Note that the reflected pulse from the sample should locate at the time of the
        main pulse of the reference signal. sampTime should only include the first order reflection, and truncate at least
        10 ps before the second order reflection to avoid the disturbance of other interfaces.
        theta: the incident angle of the beam, unit radian. For reflection measurement, an angle smaller than 30 degree
        should be used.
        lowFreq: the lower limit of the frequency band.
        highFreq: the higher limit of the frequency band.
        polarity: the polarity of the incident beam. 0 represents S polarization, 1 represents P polarization.

    @author: Bo Wang
    @file: ReflectionProperties.py
    @time: 2021/12/2 20:24
    """
    def __init__(self, time, refTime, sampTime, theta, lowFreq = 0.25, highFreq = 2, polarity = 0):
        if polarity not in [0, 1]:
            raise ValueError("The polarity should be 0 or 1.")

        if theta > np.pi / 6:
            raise RuntimeWarning("A incident angle of less than 30 degree is preferred.")

        self.time = time
        self.sampTime = sampTime
        self.refTime = refTime
        self.lowFreq = lowFreq
        self.highFreq = highFreq
        self.theta = theta
        self.polarity = polarity

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
        freq = np.array(freq)
        indexLimits = (freq >= self.lowFreq) & (freq <= self.highFreq)
        h = h[indexLimits]
        freq = freq[indexLimits]
        self.freq = freq

        # Reconstruct the reflectivity
        r = self._reflectivity(freq, h)
        nAir = 1

        if self.polarity == 0:
            return refracIndexS(self.theta, nAir, r)
        else:
            return refracIndexP(self.theta, nAir, r)

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
        phaseShift -= np.pi

        # The reconstructed reflectivity
        reflectivity = r * np.exp(1j * phaseShift)

        return reflectivity