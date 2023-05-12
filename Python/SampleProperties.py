# -*- coding: utf-8 -*-
"""
Create Time: 2021/1/29 8:36
Author: Bo Wang
"""
import numpy as np
from numpy.fft import fft
from FrequencyAnalysisUtil import concatenateTimeSeries, calRangeOfFrequency
from scipy import signal
#此文件由王博开发，主要实现吸收系数与折射率的计算
class SampleProperties:
    """
    This class is to calculate the refractive index and the absorption rate of 
    a subject scanned by the THz-TDS. An initial measurement without loading the 
    sample has to be made as reference signal.

    Attributes:
        refT: the time points of the reference signal
        refX: the amplitudes of the reference signal
        sampT: the time point of the sample signal
        sampX: the amplitudes of the sample signal
        d: the depth of the sample
        mode: the way to fit the phase change and the frequency. 'linear_fit', use linear regression to fit the phase change.
        'phase_comp', subtract the error induced by the anterior portion of the signal from the phase change. The default
        is 'none'.
        lowLimit: the lower limit of the partition of high SNR for the linear fit and the phase compensation.
        upLimit: the upper limit of the partition of high SNR for the linear fit and the phase compensation.
        denoise: if true, remove the reflection peaks by deconvolution
        f(np.array): the frequency points of the absorption rate and refractive index
    
    @author: Bo Wang
    """

    def __init__(self, refT, refX, sampT, sampX, d, mode = 'none', lowLimit = 0.3, upLimit = 0.5, denoise = False):
        if mode not in ['linear_fit', 'phase_comp', 'none']:
            raise ValueError("The mode should be linear_fit, phase_comp")

        self.refT = refT[:]
        self.refX = refX[:]
        self.sampT = sampT[:]
        self.sampX = sampX[:]
        self.d = d
        self.mode = mode
        self.lowLimit = lowLimit
        self.upLimit = upLimit
        self.denoise = denoise
        self.__calTransFunction()


    def calRefractiveIndex(self):
        """
        Calculate the sample's refractive index.

        Returns:
            refractiveIndex(np.array): a 1d np array of type float64
        """
        # The light speed
        c = 3e8
        refractiveIndex = self.__phaseTransFunction * c
        refractiveIndex /= (2 * np.pi * 1e9)
        refractiveIndex /= self.f
        refractiveIndex /= self.d
        refractiveIndex += 1

        # refractiveIndex = self.DealRefractive(refractiveIndex)

        return refractiveIndex

    def calExtinction(self):
        """
        Calculate the sample's absorption rate.

        Returns:
            absorptionRate(np.array): a 1d np array of float64 representing the absorption rates
        """

        # refractiveIndex = self.__linearFitRefractIndex()
        refractiveIndex = self.calRefractiveIndex()
        refractiveIndex = np.mean(refractiveIndex[(self.f > 0.6) & (self.f < 1.6)])
        absorptionRate = 4 * refractiveIndex
        absorptionRate /= self.__ampTransFunction
        absorptionRate /= np.power(refractiveIndex + 1, 2)
        if np.min(absorptionRate) < 0:
            absorptionRate = absorptionRate - np.min(absorptionRate) + 1
        absorptionRate = np.log(absorptionRate)

        # Convert the unit to cm^-1
        absorptionRate = absorptionRate*0.3/(self.d*self.f)

        return absorptionRate

    def calAbsorptionRate(self):
        """
        Calculate the sample's absorption rate.

        Returns:
            absorptionRate(np.array): a 1d np array of float64 representing the absorption rates
        """

        refractiveIndex = self.calRefractiveIndex()
        refractiveIndex = np.mean(refractiveIndex[(self.f > 0.6) & (self.f < 1.6)])
        absorptionRate = 4 * refractiveIndex
        absorptionRate /= self.__ampTransFunction
        absorptionRate /= np.power(refractiveIndex + 1, 2)
        if np.min(absorptionRate) < 0:
            absorptionRate = absorptionRate - np.min(absorptionRate) + 1
        absorptionRate = np.log(absorptionRate)

        # Convert the unit to cm^-1
        absorptionRate *= 2 / (self.d * 0.1)

        return absorptionRate

    #####################################Private Methods#######################

    def __calPhaseAndAmplitude(self, t, refX, sampX, indices):
        """
        Calculate the phase and amplitude of a frequency spectrum by
        converting the time spectrum

        Args:
            refX: the time spectrum of the reference signal
            sampX: the time spectrum of the sample signal
            indices: the indices representing the effective range
        """

        refXf = fft(refX)
        sampXf = fft(sampX)

        xf = sampXf / refXf
        amplitudes = np.abs(xf)
        phase = np.angle(xf)

        return phase[indices], amplitudes[indices]

    def __calTransFunction(self):
        """
        Calculate the sample's transmission function using the time spectrum
        of the sample and the reference signal
        """
        # The default length of the time signal 
        defDuration = 100.0
        sampDuration = self.sampT[-1] - self.sampT[0]
        refDuration = self.refT[-1] - self.refT[0]

        # Concatenate the time series to the default duration or the longer duration 
        # of the reference series and the sample series
        duration = max(defDuration, sampDuration, refDuration)
        newSampT, newSampX = concatenateTimeSeries(self.sampT, self.sampX, duration)
        self.sampX = newSampX
        _, newRefX = concatenateTimeSeries(self.refT, self.refX, duration)
        self.refX = newRefX
        f, _ = calRangeOfFrequency(newSampT, len(newSampT), False)
        f = np.array(f)

        # The frequency spectrum below 0.1 THz may contain peculiar values, therefore remove the portion below 0.1 THz
        effectLowLimit = 0.1
        indices = f >= effectLowLimit
        self.f = f[indices]
        self.refraction_ratio = np.abs(np.fft.fft(self.refX))/np.abs(np.fft.fft(self.sampX))
        self.refraction_ratio = self.refraction_ratio[indices]


        phaseTrans, self.__ampTransFunction = self.__calPhaseAndAmplitude(newSampT, newRefX, newSampX, indices)

        # Correct the jumps between consecutive phase shifts to be within pi.
        phaseTrans = -np.unwrap(phaseTrans)
        f1 = np.argmin(np.abs(f - 0.2))
        f2 = np.argmin(np.abs(f - 0.6))

        t = range(len(f[f1:f2])) + f1
        params = np.polyfit(t, phaseTrans[f1:f2], 1)
        phaseTrans = phaseTrans - params[1]

        if self.mode == 'phase_comp':
            indices, theorPhase = self.__calTheoreticalPhase()
            phaseOffset = np.mean(phaseTrans[indices] - theorPhase) / np.pi
            phaseOffset = np.round(phaseOffset) * np.pi
            self.__phaseTransFunction = phaseTrans - phaseOffset
        else:
            self.__phaseTransFunction = phaseTrans

    def __calculateSlope(self, phase, f):
        """ Use y = x * k to estimate k, thereby trans(x) * y = trans(x) * x * k, to assure that the fitted line goes
        through the origin."""
        phase = np.array(phase)
        f = np.array(f)
        x = np.sum(np.power(phase,2))
        y = np.sum(np.multiply(phase, f))
        k = y / x
        return k

    def __linearFitRefractIndex(self):
        """Make linear fitting to the sample's refractive index

        Args:
            see calAbsorptionRate

        Returns:
            a float representing the corrected refractive index
        """
        # Fit the phase shift
        indicef = np.array(np.where((self.f >= self.lowLimit) & (self.f <= self.upLimit)))
        indicef = indicef.reshape(indicef.shape[1],)
        indices = indicef.tolist()

        f = self.f[indices]
        phaseTrans = self.__phaseTransFunction[indices]

        # Make linear fitting
        slope = self.__calculateSlope(phaseTrans, f)

        # The light speed
        c = 3e8
        factor = c / self.d / (2 * np.pi * 1e9)
        refractiveIndex = factor * slope + 1

        return refractiveIndex

    def __calTheoreticalPhase(self):
        """ Calculate the theoretical phase assuming no dispersion occurs in the sample
        """
        indicef = np.array(np.where((self.f >= self.lowLimit) & (self.f <= self.upLimit)))
        indicef = indicef.reshape(indicef.shape[1],)
        indices = indicef.tolist()

        f = self.f[indices]
        # f = self.f[(self.f >= self.lowLimit) & (self.f <= self.upLimit)]
        delay = self.__calTimeDelay()
        theorPhase = 2 * np.pi * f * delay

        return indices, theorPhase

    def __calTimeDelay(self):
        """Calculate the time delay of the primary peak of the sample signal by comparing with the reference signal."""
        refMaxIndex = np.argmax(np.array(self.refX))
        refPeakTime = self.refT[refMaxIndex]
        sampMaxIndex = np.argmax(np.array(self.sampX))
        sampPeakTime = self.sampT[sampMaxIndex]
        delay = sampPeakTime - refPeakTime

        return delay
