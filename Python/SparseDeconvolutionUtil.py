# -*- coding: utf-8 -*-
"""
This utility class includes the primitive method to estimate the sparse representation of a THz signal. It can estimate
a meaningful result but costs extensive computation. The method based on machine learning is preferred.

@author: Bo Wang
@file: SparseDeconvolutionUtil.py
@time: 2021/12/31 20:19
"""
此文件由王博开发，主要计算稀疏反卷积计算
import time
from functools import partial

import numpy as np
from joblib import parallel_backend, delayed, Parallel
from scipy import signal
from ReflectionPeaksHelper import ReflectionPeaksHelper
from pywt import threshold

def calTemplate(time, ref, fd=1):
    """Calculate the template for the transmission matrix of sparse deconvolution.

    Args:
        fd: the division of frequency.
    """
    # The width of the pulse should be around 1 ps, by setting peak width as 0.8 ps, the subsectioned main pulse
    # should be close to 1.5 ps.
    rph = ReflectionPeaksHelper(time, ref, mPeakWidth= 80 // fd, cThreshold=0.1, s=False)
    mainPeak, _, _, _ = rph.findMainPeak()

    # Make the template function for Sparse deconvolution.
    # template = mainPeak / np.linalg.norm(mainPeak)
    template = mainPeak
    return template

def makeConvMatrix(m, template):
    """
    Calculate the transmission matrix of the system. The matrix is multiplied by the sparse representation of the THz signal
    to reconstruct the original signal.

    Params:
        m: the number of rows
        template: the temporal profile of the response filter
    """
    # Construct the convolutional matrix
    convMatrix = []
    n = m + template.size - 1

    for i in range(m):
        row = np.zeros(n, )
        row[i: i + template.size] = template[::-1]
        convMatrix.append(row)

    convMatrix = np.array(convMatrix)
    return convMatrix

def sparseDeconvolution(t, x, template):
    """Make sparse deconvolution to x according to the system impulse template.

    Args:
        t: timings of the THz signal.
        x: the original THz signal.
        template: the system impulse describing the temporal profile of the incident THz pulse.py

    Returns:
        sparseX: the sparse representation of x.
    """
    filteredX = _removeBackground(x)

    # Eliminate coordinate offset caused by convolution.
    startPos = template.shape[0] // 2
    endPos = startPos + t.shape[0]

    # Each row of the convolutional matrix is a template delayed by the row index.
    convMatrix = makeConvMatrix(t.size, template)

    # The step size of the shrinkage iteration.
    # stepSize = _calStepSize(convMatrix)
    stepSize = 0.0016

    # The first 60 sampling points belong to noise.
    noiseSigma = np.std(filteredX[: 60])

    # The factor of the regularization term, which is 3 * std(H * noise).
    regParam = _calRegParam(template, noiseSigma)

    # Optimize the step size until the meaningful sparse representation is found.
    sparseX = _calSparseVector(filteredX, convMatrix, regParam, stepSize)
    del convMatrix

    return sparseX[startPos: endPos]

def _removeBackground(sig):
    """
    Remove the background caused by baseline fluctuation.
    """
    # The size of the sliding window.
    n = 300
    baseline = (np.convolve(sig, np.ones((n,)) / n, mode="same"))
    return sig - baseline

def _calOptimizer(y, convMatrix, x, regParam):
    """
    The optimizer with L1 regularization

        Args:
            y: the noisy signal
            convMatix: the impulse matrix of the system
            x: the sparse vector containing the discrete impulses
            regParam: the factor of the regularization term
        Returns:
            The value of the optimization function of the current iteration
    """
    optimizer = 0.5 * np.sum(np.abs(y - np.dot(convMatrix, x).reshape(-1, )) ** 2)
    optimizer += regParam * np.sum(np.abs(x))
    return optimizer

def _calStepSize(convMatrix):
    """Calculate the step size for the shrinkage iteration.

    Args:
        convMatrix: the convolutional matrix for SD.
    Returns:
        the step size shrinkage iteration.
    """
    if any(np.array(convMatrix.shape) > 10000):

        # In case the matrix is too large for matmul.
        transConvMat = np.transpose(convMatrix)
        convMatCols = []
        for i in range(convMatrix.shape[1]):
            convMatCols.append(convMatrix[:, i])

        target = partial(_calColumnNorm, transConvMat=transConvMat)
        with parallel_backend(backend='threading', n_jobs=-1):
            colNorms = Parallel()(map(delayed(target), convMatCols))
        del transConvMat, convMatCols # Save Ram.

        # Cited from Dong 2007.
        stepSize = 1 / np.sqrt(np.sum(colNorms))
    else:
        stepSize = np.linalg.norm(np.matmul(np.transpose(convMatrix), convMatrix))
        stepSize = 2 / stepSize

    return stepSize

def _updateSparseVector(y, convMatrix, x, stepSize, regParam):
    """
    Update the sparse vector with shrinkage method.

    Args:
        convMatrix: the transmission matrix of the system.
        stepSize: the step size of the iteration.
    """
    residual = np.dot(convMatrix, x).reshape(-1, ) - y
    step = stepSize * np.dot(np.transpose(convMatrix), residual).reshape(-1, )
    x = threshold(x - step, stepSize * regParam, mode='soft')
    optimizer = _calOptimizer(y, convMatrix, x, regParam)

    return x, optimizer

def _calColumnNorm(column, transConvMat):
    """Target function for calStepSize"""
    newCol = np.dot(transConvMat, column)
    colNorm = np.linalg.norm(newCol)
    del newCol

    return colNorm ** 2

def _calRegParam(template, sigma):
    """
    Calculate the regularization parameter.

    Args:
        sigma: the standard deviation of the THz signal.
    """
    return 3 * sigma * np.linalg.norm(template)

def _calSparseVector(signal, convMatrix, regParam, stepSize):
    """Calculate the sparse vector representing the impulses of the original signal."""
    # The sparse vector
    x = np.zeros(convMatrix.shape[1])
    lastOptimizer = _calOptimizer(signal, convMatrix, x, regParam)

    # Recycle x.
    x, currOptimizer = _updateSparseVector(signal, convMatrix, x, stepSize, regParam)
    stepCount = 0

    # Iterate until the optimizer converges
    while currOptimizer < lastOptimizer and stepCount < 200:
        lastOptimizer = currOptimizer
        x, currOptimizer = _updateSparseVector(signal, convMatrix, x, stepSize, regParam)
        stepCount += 1
    # print(f"Number of iterations is {stepCount}.")

    return x

# def _optimizeSparseVector(signal, convMatrix, regParam):
#     """
#     Apply the iterative shrinkage algorithm and change the step size of the iteration to find the optimized sparse
#     representation of signal.
#
#     Returns:
#         sparseX: the final result the sparse vector.
#     """
#
#     # Step size of the shrinkage.
#     stepSize = 1 / np.sqrt(np.sum(matNorms))
#     sparseX = _calSparseVector(signal, convMatrix, regParam, stepSize)
#
#     # shrinkStep：Represents the shrinkage factor of stepSize
#     # shrinkStep = 0.95
#
#     # while np.max(np.abs(sparseX)) < 1e-2:
#     #     print(f"Step size: {shrinkStep}")
#     #     stepSize = shrinkStep * stepSize
#     #     sparseX = _calSparseVector(signal, convMatrix, regParam, stepSize)
#
#     return sparseX