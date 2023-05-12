# -*- coding: utf-8 -*-
"""
Create Time: 2021/1/29 8:36
Author: Kevin
"""
import numpy as np
from numpy.fft import fft
from scipy.interpolate import griddata
from scipy import signal
import os
import re
import time
import pandas as pd
import yaml
from ruamel.yaml import YAML
from scipy.signal import savgol_filter
import psutil

#频谱转换函数
def convertToFrequency(t, x, fs = -1, fe = -1,duration = 100, isInDb = True, denoise = False):
    """
    Transform the time domain signal to frequency domain signal

    Args:
        duration: concatenate the time series to the duration to increase the precision
        isInDb: if convert the amplitudes of frequency to dB
        fs: the start position of the frequency
        fe: the end position of the frequency
        fRange: the frequency of the converted signal
        xfRange: the amplitudes of the converted signal
        denoise: remove the oscillations induced by the reflection peaks
    """
    if np.all(x == 0):
        return np.array([]), np.array([])
    (f, xf) = timeToFrequency(t, x, duration, denoise)
    if fs == -1:
        fs = min(f)
        fe = max(f)

    if isInDb:
        xf = amplitudeToDb(xf)

    fRange, xfRange = selectRangeOfFrequency(f, xf, fs, fe)

    return fRange, xfRange

#时域转频谱
def timeToFrequency(t, x, duration, denoise = False):
    (tc, xc) = concatenateTimeSeries(t, x, duration)
    timeLength = len(xc)

    f, _ = calRangeOfFrequency(t, timeLength)
    N = len(xc)
    xf = fft(xc,N)

    xf /= timeLength
    xf *= 2

    # Normalization
    xf = np.abs(xf)
    f_index = f[0:int(np.floor(f.shape[0] / 2))]
    xf_index = xf[0:int(np.floor(xf.shape[0] / 2))]

    fre_scale = obtain_Frescale()
    if fre_scale == 1:
        xf_max = np.max(xf_index[f_index >= 0.1])
        xf = xf / xf_max
    return f, xf

#时间扩充100ps
def concatenateTimeSeries(t, x, duration):
    """Concatenate time series to a specified duration."""
    # The original duration of the time series
    origDuration = t[-1] - t[0]
    deltaT = calAverageTimeStep(t)
    timeLength = len(t)

    tc = t[:]
    xc = x[:]

    # Concatenate time series if it is shorter than the duration
    if origDuration < duration:
        newTimeLength = int(round(duration / deltaT) + 1)
        tc = np.arange(newTimeLength) * deltaT + t[0]
        tc = tc.tolist()
        xc = [0] * newTimeLength
        xc[0 : timeLength] = x[:]

    return tc, xc

#求f
def obtain_f(t,n):
    t = np.array(t)
    dt = t[1:] - t[0:-1]
    dt = dt.mean()
    fRange = 1 / dt
    df = fRange / n
    f = np.arange(n) * df
    return f

#计算平均时间
def calAverageTimeStep(t):
    """Calculate the average step of a time series."""
    t = np.array(t)
    dt = t[1:] - t[0:-1]
    dt = dt.mean()
    return dt


#计算频谱范围
def calRangeOfFrequency(t, n, isFromZero = True):
    """
    Calculate the range of frequency from the step increment of t

    n the number of steps of the time series
    isFromZero if the frequency start from 0, otherwise the first element of the frequency series is df
    f the frequency series
    df the step increment of f
    """
    dt = calAverageTimeStep(t)
    fRange = 1 / dt
    df = fRange / n
    f = np.arange(n) * df
    return f, df

#取对数
def amplitudeToDb(xf):
    """Convert intensities to decibels."""
    xfDb = 20 * np.log10(xf) 
    return xfDb


#选择频谱范围
def selectRangeOfFrequency(f, xf, fs, fe):
    """
    Select the range of frequency, starting from fs and endding at fe.
    """
    f = np.array(f)
    xf = np.array(xf)

    fRange = f[(f >= fs) & (f <= fe)]
    xfRange = xf[(f >= fs) & (f <= fe)]

    return fRange, xfRange

#解包算法
def unpackData(dataList, nPoints, fp,delay_line):
    """Unpack time series of every point of the scan and the corresponding x and y
    from the serialized data by applying specific decoding mechanism. """
    marker = np.array([(163, 163, 165, 165), (164, 164, 166, 166)])
    markerPositions0 = __findMarkerPositions(dataList, marker[0, :])
    markerPositions1 = __findMarkerPositions(dataList, marker[1, :])
    markerPositions = markerPositions0.copy()
    markerPositions.extend(markerPositions1)
    markerPositions = np.array(markerPositions)
    markerPositions.sort()

    if len(markerPositions) <= 1:
        return np.array([]),np.array([])

    # Remove all the marker positions that aren't distanced by 2 * nPoints + 20
    markerPositions = __removeIllegalPositions(dataList, markerPositions, nPoints,marker)
    if len(markerPositions) <= 1:
        return np.array([]),np.array([])

    nSeries = len(markerPositions) - 1
    delay_length,shine_length = read_delayline()
    fast_thz_time = np.arange(nPoints) * delay_length * fp * shine_length / 300

    # Decode the time series, x, and y
    timeSeriesList = []
    if len(markerPositions1) > 0:
        for i in range(nSeries):
            if delay_line == 1 and markerPositions[i] in markerPositions0:
                continue
            if delay_line == 0 and markerPositions[i] in markerPositions1:
                continue
            series = __decodeTimeSeries(dataList, markerPositions, i, nPoints)
            series = np.array(series)
            if delay_line == 2 and markerPositions[i] not in markerPositions1:
                series = series[::-1]
            if delay_line == 0:
                series = series[::-1]
            denoised = __removeNoise(series, nPoints, 5)
            timeSeriesList.append(denoised)
    else:
        for i in range(nSeries):
            series = __decodeTimeSeries(dataList, markerPositions, i, nPoints)
            series = np.array(series)
            denoised = __removeNoise(series, nPoints, 5)
            timeSeriesList.append(denoised)

    fast_thz_signal = align_signal(timeSeriesList)

    return fast_thz_time, fast_thz_signal

#信号对齐函数
def align_signal(timeSeriesList):

    align = read_align()

    if align == 0:
        fast_thz_signal = np.mean(np.array(timeSeriesList), axis=0)
    else:
        peak0 = np.argmax(timeSeriesList,axis=1)
        mean_peak0 = int(np.mean(peak0))
        for i in range(len(peak0)):
            if peak0[i] < mean_peak0:
                timeSeriesList[i] = np.append(np.zeros(mean_peak0 - peak0[i]),
                                              timeSeriesList[i][0:len(timeSeriesList[i]) - mean_peak0 + peak0[i]])
            elif peak0[i] > mean_peak0:
                timeSeriesList[i] = np.append(timeSeriesList[i][peak0[i] - mean_peak0:],
                                              np.zeros(peak0[i] - mean_peak0))
        fast_thz_signal = np.mean(np.array(timeSeriesList), axis=0)

    return fast_thz_signal

#数据转换，三维使用
def convertData(xList, yList, ppval, td, timeSeriesList):
    M_pp = np.array([xList, yList, ppval])
    M_td = np.array([xList, yList, td])
    M_thz_sig = np.vstack((np.array(xList).reshape(-1,), np.array(yList).reshape(-1,)))
    M_thz_sig = np.vstack((M_thz_sig, np.array(timeSeriesList).T))
    return M_pp, M_td, M_thz_sig

#范围求取t和x
def truncateTimeSeries(t, x, ts, te):
    """ Truncate a time series from the start, ts, the the end, te"""
    startIndex = 0
    endIndex = 0

    for i, item in enumerate(t):
        if item >= ts and startIndex == 0:
            startIndex = i

        if item > te and endIndex == 0:
            endIndex = i
            break

    # In case te is above the range of t
    if endIndex == 0:
        endIndex = len(t)

    tt = t[startIndex: endIndex]
    xt = x[startIndex: endIndex]

    return tt, xt

############################Private methods####################################
#解包找标记位
def __findMarkerPositions(x, marker):
    """Find the positions of the marker, which is used to separate the adjacent time series, in x."""
    positions = []
    arr = np.where(x == marker[0])

    for i in range(len(arr[0])):
        if arr[0][i] + 3 >= len(x):
            continue
        if x[arr[0][i] + 1] == marker[1] and x[arr[0][i] + 2] == marker[2] and x[arr[0][i] + 3] == marker[3]:
            positions.append(arr[0][i])

    return positions

#去掉不合理的标记位
def __removeIllegalPositions(datalist, positions, nPoints,marker):
    x = datalist.copy()
    position = np.array(positions).copy()
    position_distance = np.diff(position)
    defaultDist = 2 * nPoints + 20
    lostPointsIndices = np.array(np.where(position_distance != defaultDist))
    lostPointsIndices = lostPointsIndices.reshape(lostPointsIndices.shape[1], )

    for index in lostPointsIndices:
        if position_distance[index] > defaultDist:
            num = position_distance[index] - defaultDist
            data = x[position[index] + 4:position[index] + 4 + num]
            if marker.shape[0] == 1:
                if np.sum(data == marker[-1]) == num:
                    positions[index] = positions[index] + num
                    position_distance[index] = position_distance[index] - num
            else:
                if np.sum(data == marker[0,-1]) == num or np.sum(data == marker[1,-1]) == num:
                    positions[index] = positions[index] + num
                    position_distance[index] = position_distance[index] - num

    lostPointsIndices1 = np.array(np.where(position_distance != defaultDist))
    lostPointsIndices1 = lostPointsIndices1.reshape(lostPointsIndices1.shape[1], )
    positions = np.delete(positions,lostPointsIndices1)
    return positions

#进行去底噪
def __removeNoise(x, nPoints, order):
    """ Remove noise from the input time series by polynomial fitting.
        TO DO Improve the method to denoise the time series
        order: the order of the polynomial fitting
    """
    t = range(nPoints)
    params = np.polyfit(t, x, order)
    noise = np.polyval(params, t)
    denoise = np.array(x) - np.array(noise)

    return denoise

#进行去底噪，使用均值
def __removeNoise1(x):
    denoised = np.array(x) - np.mean(x)
    ppval1 = max(denoised) - min(denoised)
    return denoised, ppval1

#编码解析
def __decodeTimeSeries(dataList, positions, index, nPoints):
    position = positions[index]
    startPos = position + 4
    endPos = position + 2 * nPoints + 4
    seriesInList = dataList[startPos: endPos]
    seriesInList = np.array(seriesInList)
    seriesInListOdd = seriesInList[::2]
    seriesInListEven = seriesInList[1::2]
    decoded = seriesInListOdd + seriesInListEven * 256

    return decoded

#x和y坐标解析
def __decodeXY(dataList, positions, index, nPoints):
    position = positions[index]

    # x and y are respectively combined by four elements
    xIndices = position + 2 * nPoints + np.arange(4,8)
    yIndices = position + 2 * nPoints + np.arange(8,12)

    # Bitwise shift the components of X and Y to the left by integer times of nBits
    nBits = 8

    x = np.arange(len(xIndices))
    shiftBits = nBits * x
    xComponent = dataList[xIndices[x]]
    yComponent = dataList[yIndices[x]]
    xComponent = xComponent << shiftBits
    yComponent = yComponent << shiftBits
    xValue = sum(xComponent)
    yValue = sum(yComponent)

    # Convert the unit of x and y from micron to mm
    umPerMm = 1000.0
    xValue = np.int32(xValue)
    xValue /= umPerMm
    yValue = np.int32(yValue)
    yValue /= umPerMm

    return xValue, yValue

#获取算法配置文件参数，延迟线参数
def obtain_config():
    delay_line = 0
    path = r'C:\thztools\thz\Algconfig.yaml'
    try:
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                data = yaml.load(f, Loader=yaml.FullLoader)
        delay_line = data['System_configs']['delay_line']
        if delay_line not in [0, 1, 2]:
            delay_line = 0
    except Exception as e:
        with open(os.path.join("C:\\thztools", 'alglog.log'), 'a+', encoding='utf-8') as ff:
            ff.write(f'{time.strftime("%Y-%m-%d %X", time.localtime())} The error is : {e} \n')

    return delay_line


#获取算法配置文件参数，特征长度
def classification_width():
    width = 190
    path = r'C:\thztools\thz\Algconfig.yaml'
    try:
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                data = yaml.load(f, Loader=yaml.FullLoader)
        width = data['Classification_configs']['width_cla']
    except Exception as e:
        with open(os.path.join("C:\\thztools", 'alglog.log'), 'a+', encoding='utf-8') as ff:
            ff.write(f'{time.strftime("%Y-%m-%d %X", time.localtime())} The error is : {e} \n')

    return width

#获取算法配置文件参数，物质识别
def classification_config():
    classification = 0
    protein_classy = 0
    samples = []
    path = r'C:\thztools\thz\Algconfig.yaml'
    try:
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                data = yaml.load(f, Loader=yaml.FullLoader)
        classification = data['Classification_configs']['classification']
        protein_classy = data['Classification_configs']['protein_classy']
        sample = data['Classification_configs']['samples_cla']
        sample1 = sample.replace(' ','')
        samples = sample1.split(',')

        if classification not in [0, 1]:
            classification = 0
        if protein_classy not in [0, 1]:
            protein_classy = 0

    except Exception as e:
        with open(os.path.join("C:\\thztools", 'alglog.log'), 'a+', encoding='utf-8') as ff:
            ff.write(f'{time.strftime("%Y-%m-%d %X", time.localtime())} The error is : {e} \n')

    return samples,classification,protein_classy


#获取算法配置文件频谱最大值
def obtain_Fre():
    thz_max = 3
    path = r'C:\thztools\thz\Algconfig.yaml'
    try:
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                data = yaml.load(f, Loader=yaml.FullLoader)
        thz_max = data['Signal_configs']['thz_max']
        if thz_max < 3:
            thz_max = 3
    except Exception as e:
        with open(os.path.join("C:\\thztools", 'alglog.log'), 'a+', encoding='utf-8') as ff:
            ff.write(f'{time.strftime("%Y-%m-%d %X", time.localtime())} The error is : {e} \n')

    return thz_max

#获取算法配置文件，是否进行频谱归一化
def obtain_Frescale():
    fre_scale = 0
    path = r'C:\thztools\thz\Algconfig.yaml'
    try:
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                for line in f.readlines():
                    if '#' in line:
                        continue
                    if 'fre_scale' in line:
                        fre_scale = line.split(':')[-1]
                        fre_scale = fre_scale.replace(' ', '')
                        fre_scale = int(fre_scale)
        if fre_scale not in [0,1]:
            fre_scale = 0
    except Exception as e:
        with open(os.path.join("C:\\thztools", 'alglog.log'), 'a+', encoding='utf-8') as ff:
            ff.write(f'{time.strftime("%Y-%m-%d %X", time.localtime())} The error is : {e} \n')

    return fre_scale


#是否进行频谱平滑
def open_frequency():
    use_Fresmooth = 0
    path = r'C:\thztools\thz\Algconfig.yaml'
    try:
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                data = yaml.load(f, Loader=yaml.FullLoader)
            use_Fresmooth = data['Signal_configs']['use_Fresmooth']
        if use_Fresmooth not in [0,1]:
            use_Fresmooth = 0

    except Exception as e:
        with open(os.path.join("C:\\thztools", 'alglog.log'), 'a+', encoding='utf-8') as ff:
            ff.write(f'{time.strftime("%Y-%m-%d %X", time.localtime())} The error is : {e} \n')
    return use_Fresmooth


#读取算法信号对齐配置参数
def read_align():
    signal_align = 0
    path = r'C:\thztools\thz\Algconfig.yaml'
    try:
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                data = yaml.load(f, Loader=yaml.FullLoader)
            signal_align = data['Signal_configs']['align_signal']
        if signal_align not in [0,1]:
            signal_align = 0

    except Exception as e:
        with open(os.path.join("C:\\thztools", 'alglog.log'), 'a+', encoding='utf-8') as ff:
            ff.write(f'{time.strftime("%Y-%m-%d %X", time.localtime())} The error is : {e} \n')

    return signal_align


#保存txt
def save_txt():
    save_open = 0
    path = r'C:\thztools\thz\Algconfig.yaml'
    try:
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                data = yaml.load(f, Loader=yaml.FullLoader)
            save_open = data['Signal_configs']['save_txt']
        if save_open not in [0,1]:
            save_open = 0

    except Exception as e:
        with open(os.path.join("C:\\thztools", 'alglog.log'), 'a+', encoding='utf-8') as ff:
            ff.write(f'{time.strftime("%Y-%m-%d %X", time.localtime())} The error is : {e} \n')

    return save_open

#显示滤波后的信号
def show_filterSignal():
    show_filter = 0
    path = r'C:\thztools\thz\Algconfig.yaml'
    try:
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                data = yaml.load(f, Loader=yaml.FullLoader)
            show_filter = data['Signal_configs']['show_filter']
        if show_filter not in [0,1]:
            show_filter = 0

    except Exception as e:
        with open(os.path.join("C:\\thztools", 'alglog.log'), 'a+', encoding='utf-8') as ff:
            ff.write(f'{time.strftime("%Y-%m-%d %X", time.localtime())} The error is : {e} \n')

    return show_filter

#是否关闭算法滤波的读取配置函数
def zeroYaml():
    path = r'C:\thztools\thz\Algconfig.yaml'

    yaml = YAML()
    with open(path, "r", encoding='utf-8') as f:
        data = yaml.load(f)

    if data['Signal_configs']['filter_restart'] == 0:

        data['Signal_configs']['filter_type'] = 0

        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(data, f)


#算法配置文件滤波类型与是否显示
def freq_params():
    filter_type = 0
    show_filter = 0
    path = r'C:\thztools\thz\Algconfig.yaml'
    try:
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                for line in f.readlines():
                    if '#' in line:
                        continue
                    if 'filter_type' in line:
                        filter_type = line.split(':')[-1]
                        filter_type = filter_type.replace(' ', '')
                        filter_type = int(filter_type)
                    if 'show_filter' in line:
                        show_filter = line.split(':')[-1]
                        show_filter = show_filter.replace(' ', '')
                        show_filter = int(show_filter)
            if show_filter not in [0, 1]:
                show_filter = 0
            if filter_type not in range(10):
                filter_type = 0
    except Exception as e:
        with open(os.path.join("C:\\thztools", 'alglog.log'), 'a+', encoding='utf-8') as ff:
            ff.write(f'{time.strftime("%Y-%m-%d %X", time.localtime())} The error is : {e} \n')

    return filter_type,show_filter


#选择滤波类型参数
def select_filter():
    filter_type = 0
    path = r'C:\thztools\thz\Algconfig.yaml'
    try:
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                data = yaml.load(f, Loader=yaml.FullLoader)
            filter_type = data['Signal_configs']['filter_type']
        if filter_type not in range(10):
            filter_type = 0

    except Exception as e:
        with open(os.path.join("C:\\thztools", 'alglog.log'), 'a+', encoding='utf-8') as ff:
            ff.write(f'{time.strftime("%Y-%m-%d %X", time.localtime())} The error is : {e} \n')

    return filter_type

#算法配置文件移除二次反射峰
def move_secondPeakconfig():
    move_second = 0
    path = r'C:\thztools\thz\Algconfig.yaml'
    try:
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                data = yaml.load(f, Loader=yaml.FullLoader)
            move_second = data['Signal_configs']['move_second']
        if move_second not in [0,1]:
            move_second = 0

    except Exception as e:
        with open(os.path.join("C:\\thztools", 'alglog.log'), 'a+', encoding='utf-8') as ff:
            ff.write(f'{time.strftime("%Y-%m-%d %X", time.localtime())} The error is : {e} \n')

    return move_second

#移除二次反射峰
def move_secondPeak(t0,ref0):
    index_max = np.argmax(np.abs(ref0))
    t0_max = np.max(t0)
    if t0_max - t0[index_max] > 41:
        index, value = signal.find_peaks(ref0, height=0.25 * np.max(ref0), distance=len(t0[(t0 > 0) & (t0 < 3)]))

        index_new = []
        for i in range(len(index)):
            if index[i] > index_max + len(t0[(t0 > 0) & (t0 < 4)]):
                index_new.append(index[i])

        if len(index_new) > 0:
            if np.max(t0) > 90:
                t_num = 4
            else:
                t_num = 3
            for j in range(len(index_new)):

                len_num = len(t0[(t0 > t_num) & (t0 < t_num * 2)]) // 2
                if index_new[j] + len_num + 10 < len(ref0) and index_new[j] - len_num - 10 >= 0 \
                        and index_new[j] - 3 * len_num >= 0:
                    ref0[index_new[j] - len_num:index_new[j] + len_num] = ref0[index_new[j] - 3 * len_num:index_new[
                                                                                                              j] - len_num]

                    ref0[index_new[j] - len_num - 10:index_new[j] - len_num + 10] = savgol_filter(
                        ref0[index_new[j] - len_num - 10:index_new[j] - len_num + 10],
                        7, 3, mode='nearest')

                    ref0[index_new[j] + len_num - 10:index_new[j] + len_num + 10] = savgol_filter(
                        ref0[index_new[j] + len_num - 10:index_new[j] + len_num + 10],
                        7, 3, mode='nearest')
                elif index_new[j] + len_num + 10 >= len(ref0):
                    ref0[index_new[j] - len_num:] = ref0[index_new[j] - len_num - len(ref0[index_new[j] - len_num:]):
                                                         index_new[j] - len_num]

    return ref0

#移除二次反射峰算法2
def move_secondPeak2(ref0):
    index_max = np.argmax(np.abs(ref0))
    t0 = np.linspace(0, 120, len(ref0))
    t0_max = np.max(t0)
    if t0_max - t0[index_max] > 44:
        index1 = np.argmin(np.abs(t0 - t0[index_max]-44))
        if index1 + 150 < len(ref0):
            ref0[index1 - 150:index1 + 150] = ref0[index1 - 450:index1 - 150]
        else:
            ref0[index1 - 150:] = ref0[index1 - 450:len(ref0) - 300]
    if t0_max - t0[index_max] > 88:
        index2 = np.argmin(np.abs(t0 - t0[index_max]-88))
        if index2 + 150 < len(ref0):
            ref0[index2 - 150:index2 + 150] = ref0[index2 - 450:index2 - 150]
        else:
            ref0[index2 - 150:] = ref0[index2 - 450:len(ref0) - 300]

    if t0_max - t0[index_max] > 41:
        index, value = signal.find_peaks(ref0, height=0.25 * np.max(ref0), distance=len(t0[(t0 > 0) & (t0 < 3)]))

        index_new = []
        for i in range(len(index)):
            if index[i] > index_max + len(t0[(t0 > 0) & (t0 < 4)]):
                index_new.append(index[i])

        if len(index_new) > 0:
            if len(ref0) > 6750:
                len_num = 150
            else:
                len_num = 100
            for j in range(len(index_new)):
                if index_new[j] + len_num + 10 < len(ref0) and index_new[j] - len_num - 10 >= 0 \
                        and index_new[j] - 3 * len_num >= 0:
                    ref0[index_new[j] - len_num:index_new[j] + len_num] = ref0[index_new[j] - 3 * len_num:index_new[
                                                                                                              j] - len_num]

                    ref0[index_new[j] - len_num - 10:index_new[j] - len_num + 10] = savgol_filter(
                        ref0[index_new[j] - len_num - 10:index_new[j] - len_num + 10],
                        7, 3, mode='nearest')

                    ref0[index_new[j] + len_num - 10:index_new[j] + len_num + 10] = savgol_filter(
                        ref0[index_new[j] + len_num - 10:index_new[j] + len_num + 10],
                        7, 3, mode='nearest')
                elif index_new[j] + len_num + 10 >= len(ref0):
                    ref0[index_new[j] - len_num:] = ref0[index_new[j] - len_num - len(ref0[index_new[j] - len_num:]):
                                                         index_new[j] - len_num]

    return ref0

#移除二次反射峰算分1
def move_secondPeak1(ref0):
    index_max = np.argmax(np.abs(ref0))
    index, value = signal.find_peaks(ref0, height=0.25 * np.max(ref0), distance=300)

    index_new = []
    for i in range(len(index)):
        if index[i] > index_max + 300:
            index_new.append(index[i])

    if len(index_new) > 0:
        if len(ref0) > 6750:
            len_num = 150
        else:
            len_num = 100
        for j in range(len(index_new)):
            if index_new[j] + len_num + 10 < len(ref0) and index_new[j] - len_num - 10 >= 0 \
                    and index_new[j] - 3 * len_num >= 0:
                ref0[index_new[j] - len_num:index_new[j] + len_num] = ref0[index_new[j] - 3 * len_num:index_new[
                                                                                                          j] - len_num]

                ref0[index_new[j] - len_num - 10:index_new[j] - len_num + 10] = savgol_filter(
                    ref0[index_new[j] - len_num - 10:index_new[j] - len_num + 10],
                    7, 3, mode='nearest')

                ref0[index_new[j] + len_num - 10:index_new[j] + len_num + 10] = savgol_filter(
                    ref0[index_new[j] + len_num - 10:index_new[j] + len_num + 10],
                    7, 3, mode='nearest')
            elif index_new[j] + len_num + 10 >= len(ref0):
                ref0[index_new[j] - len_num:] = ref0[index_new[j] - len_num - len(ref0[index_new[j] - len_num:]):
                                                     index_new[j] - len_num]
    return ref0

#读取延迟线参数
def read_delayline():
    delay_length = 0.5
    shine_length = 6
    path = r'C:\thztools\thz\Algconfig.yaml'
    try:
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                data = yaml.load(f, Loader=yaml.FullLoader)
            delay_length = float(data['System_configs']['delay_length'])
            shine_length = float(data['System_configs']['shine_length'])


    except Exception as e:
        with open(os.path.join("C:\\thztools", 'alglog.log'), 'a+', encoding='utf-8') as ff:
            ff.write(f'{time.strftime("%Y-%m-%d %X", time.localtime())} The error is : {e} \n')

    return delay_length,shine_length


#获取pid
def writePid():
    pids = psutil.pids()
    for pid in pids:
        p = psutil.Process(pid)
        if 'pythonTHZ.exe' in p.name():
            wirteYaml(p.pid)

#写pid
def wirteYaml(pid):
    path = r'C:\thztools\thz\Algconfig.yaml'

    yaml = YAML()
    with open(path, "r", encoding='utf-8') as f:
        data = yaml.load(f)

    data['System_configs']['THZ'] = pid

    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(data, f)

