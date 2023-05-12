import numpy as np
from scipy import signal
#此文件由王博开发，主要计算反射式寻峰
class ReflectionPeaksHelper:
    """
        This class is to find the reflection peaks using the cross-correlation method.

        Attributes:
            t: the time points
            x: the amplitudes
            mPeakWidth: the width of the main peak
            cThreshold: the minimum value of the correlation by which a peak is determined
            as a reflection of the main peak
            s: if true, the curve of the time series is smooth

        @author Bo Wang
        """
    def __init__(self, t, x, mPeakWidth, cThreshold, s):
        self.t = t
        self.x = x
        self.mainPeakWidth = mPeakWidth
        self.corrThreshold = cThreshold
        self.isSmooth = s

    def findMainPeak(self):
        """
        Find the range of the main peak, which is from mainPeakWidth prior
        to the peak to mainPeakWidth after the valley

        y: the values of the main peak
        startIndex: the start position of the main peak
        endIndex: the end position of the main peak
        peakIndex: the position of the peak
        """
        maxValue = max(self.x)
        minValue = min(self.x)

        # The height of the main pulse might be lower than that of the first reflected pulse, i.e. the absorption of the sample is
        # sufficiently low and the bottom surface has very strong reflection.
        peakIndex = signal.find_peaks(self.x, maxValue / 2, threshold=None, distance=None)[0][0]
        valleyIndex = signal.find_peaks(-np.array(self.x), -minValue / 2)[0][0]

        # peakIndex = self.x.index(peakValue)[0]
        # valleyIndex = self.x.index(valleyValue)[0]

        # The peak is before the valley
        startIndex = peakIndex - self.mainPeakWidth
        endIndex = valleyIndex + self.mainPeakWidth + 1

        # The peak is after the valley
        if peakIndex > valleyIndex:
            startIndex = valleyIndex - self.mainPeakWidth
            endIndex = peakIndex + self.mainPeakWidth + 1

        if startIndex < 0:
            startIndex = 0

        if endIndex > len(self.x):
            endIndex = len(self.x)

        y = self.x[startIndex: endIndex]

        return y, startIndex, endIndex, peakIndex

    def findMainPeakMedian(self):
        """
        Find the range of the main peak, which is from mainPeakWidth prior
        to the peak to mainPeakWidth after the valley

        y: the values of the main peak
        startIndex: the start position of the main peak
        endIndex: the end position of the main peak
        peakIndex: the position of the peak
        """
        maxValue = max(self.x)
        minValue = min(self.x)

        # The height of the main pulse might be lower than that of the first reflected pulse, i.e. the absorption of the sample is
        # sufficiently low and the bottom surface has very strong reflection.
        peakIndex = signal.find_peaks(self.x, maxValue / 2)[0][0]
        valleyIndex = signal.find_peaks(-np.array(self.x), -minValue / 2)[0][0]


        # The peak is after the valley
        if peakIndex > valleyIndex:
            y = self.x[valleyIndex: peakIndex]
        else:
            y = self.x[peakIndex: valleyIndex]

        return y, peakIndex, valleyIndex

    def findReflectionPeaks(self):
        """
        Find the starts and the ends of all the reflection peaks by correlating
        the main peak to the rest of the time series.

        y[:, 0] are the starts of the reflection peaks, y[:, 1] are the ends of the
        reflection peaks, y = [] if there is no reflection peak.
        """
        (xMp, mStartIndex, mEndIndex, mPeakIndex) = self.findMainPeak()

        # Substitute the section of the main peak to zeros
        xRest = np.array(self.x[:])
        xRest[mStartIndex: mEndIndex] = 0

        # Concatenate zeros to the end of the main peak to make it the same length
        # as xRest
        xMpComp = np.zeros(xRest.size)
        xMpComp[: len(xMp)] = xMp

        # Normalize the inputs to make the correlation signal normalized
        xRest = (xRest - np.mean(xRest)) / (np.std(xRest) * xRest.size)
        xMpComp = (xMpComp - np.mean(xMpComp)) / (np.std(xMpComp))
        xCorr = signal.correlate(xRest, xMpComp, mode='full')

        # Only take the positive lags
        xCorr = xCorr[-xRest.size:]

        # The minimum distance between two consecutive peaks
        # TODO for thin films, the main pulse and the reflection pulse may overlap
        minPeakDistance = self.mainPeakWidth * 4
        corrPeaks = signal.find_peaks(np.abs(xCorr), height=self.corrThreshold, distance=minPeakDistance)
        corrPeakIndices = corrPeaks[0]

        # Write the starts and the ends of the reflection peaks
        y = self._writeReflectionPeaks(xCorr, corrPeakIndices, mStartIndex, mEndIndex)

        return y
        
    def findReflectionPeaksMedian(self,select_peak):
        maxValue = max(self.x)
        minValue = min(self.x)

        peakIndex = signal.find_peaks(self.x, maxValue / 4,distance=50)
        valleyIndex = signal.find_peaks(-np.array(self.x), -minValue / 4,distance=50)

        valleyIndex_num = len(valleyIndex[0])

        if len(peakIndex[0]) >= 2:
            peakIndex = peakIndex[0][1]
        else:
            peakIndex = signal.find_peaks(self.x, maxValue / 16, distance=50)
            peakIndex = peakIndex[0][1]


        if len(valleyIndex[0]) >= 2:
            valleyIndex1 = valleyIndex[0][1]
        else:
            valleyIndex = signal.find_peaks(-np.array(self.x), -minValue / 16,distance=50)
            valleyIndex1 = valleyIndex[0][1]

        if peakIndex > valleyIndex1:
            y = self.x[valleyIndex1:peakIndex]
            startIndex = valleyIndex1
            endIndex = peakIndex
        else:
            y = self.x[peakIndex:valleyIndex1]
            startIndex = peakIndex
            endIndex = valleyIndex1

        # if select_peak == 0:
        #     if valleyIndex1 < peakIndex:
        #         for i in range(2,valleyIndex_num):
        #             if valleyIndex[0][i] > peakIndex:
        #                 valleyIndex1 = valleyIndex[0][i]
        #                 break

        # if select_peak == 1:
        #     if valleyIndex1 < peakIndex:
        #         for i in range(2,valleyIndex_num):
        #             if valleyIndex[0][i] < peakIndex:
        #                 valleyIndex1 = valleyIndex[0][i]
        #             else:
        #                 break

        # # The peak is after the valley
        # if select_peak == 1:
        #     y = self.x[valleyIndex1: peakIndex]
        #     startIndex = valleyIndex1
        #     endIndex = peakIndex
        # else:
        #     y = self.x[peakIndex: valleyIndex1]
        #     endIndex = valleyIndex1
        #     startIndex = peakIndex

        return y, startIndex, endIndex

        ##################### Private methods ###########################

    def _writeReflectionPeaks(self, xCorr, corrPeakIndices, mStartIndex, mEndIndex):
        """
        Args:
            mEndIndex: the end index of the main peak
        """
        nRefPeaks = corrPeakIndices.size

        # The start and the end indices
        y = []

        for i in range(nRefPeaks):
            # The position of a correlation peak is the beginning of
            # a reflection peak
            startIndex = corrPeakIndices[i]
            endIndex = corrPeakIndices[i] + mEndIndex - mStartIndex

            # The reflection peaks should be after the main peak in a transmissive system. The reflection peak should not
            # overlap with the main peak
            if startIndex <= mEndIndex:
                continue

            if endIndex > len(self.x):
                endIndex = len(self.x)

            corrValue = xCorr[startIndex]
            peakValues = self.x[startIndex: endIndex]
            if corrValue > 0:
                # No phase shift of the wave package
                peakIndex = np.argmax(peakValues) + startIndex
            else:
                # The phase shift of the wave package is pi and thus the profile of the wave is reversed.
                peakIndex = np.argmin(peakValues) + startIndex

            y.append([startIndex, endIndex, peakIndex])

        return y