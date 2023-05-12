# -*- coding: utf-8 -*-
"""
@Create Time: 2021/12/14 13:24
@Author: Kevin
@Python Version：3.7.6
"""
import os
import numpy as np
import pywt
from scipy import signal
import time
import yaml
import pandas as pd
from numpy.fft import fft,ifft
from SparseDeconvolutionUtil import  sparseDeconvolution,calTemplate

#获取算法配置文件中的参数
def getParams():
	path = r'C:\thztools\thz\Algconfig.yaml'
	sg = 0
	dwt = 0
	medianf = 0
	filter_name = 'none'
	dwt_name = 'db13'
	lowpass = 0.006
	highpass = 0.8
	try:
		if os.path.exists(path):
			with open(path, 'r', encoding='utf-8') as f:
				data = yaml.load(f, Loader=yaml.FullLoader)
			sg = data['Signal_configs']['SG']
			dwt = data['Signal_configs']['dwt']
			medianf = data['Signal_configs']['medianf']
			dwt_name = data['Signal_configs']['wave_name']
			filter_name = data['Signal_configs']['filter_name']
			lowpass = data['Signal_configs']['lowpass_value']
			highpass = data['Signal_configs']['highpass_value']
			if sg not in [0, 1, 2]:
				sg = 0
			if dwt not in [0, 1]:
				dwt = 0
			if medianf % 2 == 0:
				medianf = 0
			if filter_name not in ['highpass', 'lowpass', 'bandpass']:
				filter_name = 'none'
	except Exception as e:
		with open(os.path.join("C:\\thztools", 'alglog.log'), 'a+', encoding='utf-8') as ff:
			ff.write(f'{time.strftime("%Y-%m-%d %X", time.localtime())} The error is : {e} \n')
	return sg,medianf,dwt,dwt_name,filter_name,lowpass,highpass


#使用算法滤波函数
def useFilter(t_value,ref_value):
	sg,medianf,dwt,dwt_name,filter_name,lowpass,highpass = getParams()#获取高低通滤波与小波变换参数
	if np.max(t_value) > 500:
		#下面进行滤波，可叠加使用
		try:
			if sg == 1:
				ref_value = signal.savgol_filter(ref_value, 5, 3)
			elif sg == 2:
				ref_value = signal.savgol_filter(ref_value, 25, 5)

			if medianf != 0:
				ref_value = signal.medfilt(ref_value, medianf)

			if filter_name != 'none':
				if filter_name == 'bandpass':
					b, a = signal.butter(8, [highpass, lowpass], 'bandpass')
					ref_value = signal.filtfilt(b, a, ref_value)
				elif filter_name == 'highpass':
					b, a = signal.butter(8, highpass, 'highpass')
					ref_value = signal.filtfilt(b, a, ref_value)
				else:
					b, a = signal.butter(8, lowpass, 'lowpass')
					ref_value = signal.filtfilt(b, a, ref_value)

			if dwt == 1:
				wavelists = []
				for family in pywt.families():
					for i in range(len(pywt.wavelist(family))):
						wavelists.append(pywt.wavelist(family)[i])
				if dwt_name in wavelists:
					num = ref_value.shape[0]
					mcoeffs = pywt.wavedec(ref_value,dwt_name, mode='symmetric', level=5)
					for k in range(1, len(mcoeffs)):
						value = np.sqrt(2 * np.log(ref_value.shape[0]))
						mcoeffs[k] = pywt.threshold(np.array(mcoeffs[k]), value=value, mode='soft')
					ref_value = pywt.waverec(mcoeffs, wavelet=dwt_name, mode='symmetric')
					if ref_value.shape[0] > num:
						ref_value = ref_value[0:num]
					else:
						ref_value = np.append(ref_value, np.zeros(ref_value.shape[0] - num))

		except Exception as e:
			with open(os.path.join("C:\\thztools", 'alglog.log'), 'a+', encoding='utf-8') as ff:
				ff.write(f'{time.strftime("%Y-%m-%d %X", time.localtime())} The error is : {e} \n')

	return ref_value


#读取算法配置文件，是否进行卷积操作。
def ifOpenConv():
    open_cov = 0
    path = r'C:\thztools\thz\Algconfig.yaml'
    try:
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                data = yaml.load(f, Loader=yaml.FullLoader)
        open_cov = data['Signal_configs']['open_cov']
        if open_cov not in [0, 1]:
            open_cov = 0
    except Exception as e:
        with open(os.path.join("C:\\thztools", 'alglog.log'), 'a+', encoding='utf-8') as ff:
            ff.write(f'{time.strftime("%Y-%m-%d %X", time.localtime())} The error is : {e} \n')

    return open_cov

#使用卷积滤波
def useConv(fast_thz_signal):
    open_cov = ifOpenConv()
    if open_cov == 1:
        n = int(150 / 1)
        baseline = (np.convolve(fast_thz_signal, np.ones((n,)) / n, mode="same"))
        fast_thz_signal = fast_thz_signal - baseline
    return fast_thz_signal


#获取切比学府算法参数
def obtain_cheb():
	fp,fs,rp,rs,Fs = 1,3,3,40,100
	filter_name = 'lowpass'
	path = r'C:\thztools\thz\Algconfig.yaml'
	try:
		if os.path.exists(path):
			with open(path, 'r', encoding='utf-8') as f:
				data = yaml.load(f, Loader=yaml.FullLoader)
		fp = data['Signal_configs']['fp']
		fs = data['Signal_configs']['fs']
		rp = data['Signal_configs']['rp']
		rs = data['Signal_configs']['rs']
		Fs = data['Signal_configs']['Fs']
		filter_name = data['Signal_configs']['filter_name']

	except Exception as e:
		with open(os.path.join("C:\\thztools", 'alglog.log'), 'a+', encoding='utf-8') as ff:
			ff.write(f'{time.strftime("%Y-%m-%d %X", time.localtime())} The error is : {e} \n')

	return fp,fs,rp,rs,Fs,filter_name


#使用切比雪夫滤波
def useCheb(fast_thz_signal):
	fp,fs,rp,rs,Fs,filter_name = obtain_cheb()
	fast_thz_signal = chebyshevFilter(fast_thz_signal,fp,fs,rp,rs,Fs,filter_name)
	return fast_thz_signal


#使用切比雪夫滤波2
def useCheb2(fast_thz_signal):
	fp,fs,rp,rs,Fs,filter_name = obtain_cheb()
	fast_thz_signal = chebyshevFilter2(fast_thz_signal,fp,fs,rp,rs,Fs,filter_name)
	return fast_thz_signal


#使用椭圆滤波
def useElliptic(fast_thz_signal):
	fp,fs,rp,rs,Fs,filter_name = obtain_cheb()
	fast_thz_signal = ellipticFilter(fast_thz_signal,fp,fs,rp,rs,Fs,filter_name)
	return fast_thz_signal

#使用反卷积滤波
def obtain_dev():
	dev_mode, dev_path, dev_lf, dev_hf = 'DGIF','',0.24,0.56
	path = r'C:\thztools\thz\Algconfig.yaml'
	try:
		if os.path.exists(path):
			with open(path, 'r', encoding='utf-8') as f:
				data = yaml.load(f, Loader=yaml.FullLoader)
		dev_mode = data['Signal_configs']['dev_mode']
		dev_path = data['Signal_configs']['dev_path']
		dev_lf = data['Signal_configs']['DGIF_LF']
		dev_hf = data['Signal_configs']['DGIF_HF']

	except Exception as e:
		with open(os.path.join("C:\\thztools", 'alglog.log'), 'a+', encoding='utf-8') as ff:
			ff.write(f'{time.strftime("%Y-%m-%d %X", time.localtime())} The error is : {e} \n')

	return dev_mode, dev_path, dev_lf, dev_hf


#获取snr
def obtain_snr(data):
    max_index = np.argmax(data)
    max_value = np.max(data)
    if max_value + 200 < data.shape[0] and max_index - 200 > 0:
        snr1 = max_value - np.min(data[max_index-200:max_index+200])
    elif max_value + 200 < data.shape[0] and max_index - 200 < 0:
        snr1 = max_value - np.min(data[0:max_index+200])
    elif max_value + 200 > data.shape[0] and max_index - 200 < 0:
        snr1 = max_value - np.min(data[0:data.shape[0]])
    else:
        snr1 = max_value - np.min(data[max_index - 200:data.shape[0]])
    snr = snr1/(np.max(data[10:60]) - np.min(data[10:60]))
    return snr

#获取fwhm
def obtain_fwhm(t,data):
    max_index = np.argmax(data)
    max_value = np.max(data)
    left_exist = 0
    right_exist = 0
    for i in range(2000):
        if max_value/2 - data[max_index - i] > 0 and left_exist == 0:
            left_exist = left_exist + 1
            left_index = max_index - i
        if data[max_index + i] - max_value/2 < 0 and right_exist == 0:
            right_exist = right_exist + 1
            right_index = max_index + i
        if left_exist == 1 and right_exist == 1:
            break
    t_fwhm = t[right_index] - t[left_index]
    return t_fwhm


#使用wiener滤波
def wiener_filter(x,a):
    w = x.conjugate() /(np.power(np.abs(x),2)+1/a)
    return w


#使用反卷积
def useDev(fast_thz_signal):
	dev_mode, dev_path, lf, hf = obtain_dev()
	if os.path.exists(dev_path):
		data = pd.read_excel(dev_path)
		ref_signal = data.iloc[:,1].values
		t = data.iloc[:,0].values
		if dev_mode == 'DGIF':
			max_peak_index = np.argmax(fast_thz_signal)
			t1 = t - t[max_peak_index]
			g_filter = np.exp(-np.power(t1, 2) / np.power(hf, 2)) / hf - \
					   np.exp(-np.power(t1, 2) / np.power(lf, 2)) / lf
			dgf = ifft(fft(g_filter) * fft(fast_thz_signal) / fft(ref_signal))
			y = fast_thz_signal * dgf.real
			return y
		elif dev_mode == 'Wiener':
			y = fft(fast_thz_signal)
			x = fft(ref_signal)
			a = obtain_snr(ref_signal)
			w = wiener_filter(x, a)
			T_fwhm = obtain_fwhm(t, ref_signal)
			delta = np.exp(np.power(t, 2) / (-2 * np.power(T_fwhm / (2 * np.sqrt(2 * np.log(2))), 2)))
			h = ifft(y * w * fft(delta))
			result = h.real * ref_signal
			return result
		elif dev_mode == 'SD':
			template = calTemplate(t,ref_signal)
			sample100Sr = sparseDeconvolution(t, fast_thz_signal, template)
			return sample100Sr


	return  fast_thz_signal

#获取巴特沃斯的阶数
def obtain_order():
	order_num = 6
	path = r'C:\thztools\thz\Algconfig.yaml'
	try:
		if os.path.exists(path):
			with open(path, 'r', encoding='utf-8') as f:
				data = yaml.load(f, Loader=yaml.FullLoader)
		order_num = data['Signal_configs']['butter_order']
		if order_num not in range(1,9):
			order_num = 6
	except Exception as e:
		with open(os.path.join("C:\\thztools", 'alglog.log'), 'a+', encoding='utf-8') as ff:
			ff.write(f'{time.strftime("%Y-%m-%d %X", time.localtime())} The error is : {e} \n')

	return int(order_num)

#使用巴特沃斯滤波
def useButter(fast_thz_signal):
	sg, medianf, dwt, dwt_name, filter_name, lowpass, highpass = getParams()
	if filter_name != 'none':
		order_num = obtain_order()
		if filter_name == 'bandpass':
			if lowpass > highpass:
				lowpass,highpass = highpass,lowpass
			b, a = signal.butter(order_num, [lowpass, highpass], 'bandpass')
			fast_thz_signal = signal.filtfilt(b, a, fast_thz_signal)
		elif filter_name == 'highpass':
			b, a = signal.butter(order_num, highpass, 'highpass')
			fast_thz_signal = signal.filtfilt(b, a, fast_thz_signal)
		elif filter_name == 'lowpass':
			b, a = signal.butter(order_num, lowpass, 'lowpass')
			fast_thz_signal = signal.filtfilt(b, a, fast_thz_signal)
		elif filter_name == 'bandstop':
			if lowpass > highpass:
				lowpass,highpass = highpass,lowpass
			b, a = signal.butter(order_num, [lowpass, highpass], 'bandstop')
			fast_thz_signal = signal.filtfilt(b, a, fast_thz_signal)
	return fast_thz_signal


#使用Bessel滤波
def useBessel(fast_thz_signal):
	sg, medianf, dwt, dwt_name, filter_name, lowpass, highpass = getParams()
	if filter_name != 'none':
		order_num = obtain_order()
		if filter_name == 'bandpass':
			if lowpass > highpass:
				lowpass,highpass = highpass,lowpass
			b, a = signal.bessel(order_num, [lowpass, highpass], 'bandpass')
			fast_thz_signal = signal.filtfilt(b, a, fast_thz_signal)
		elif filter_name == 'highpass':
			b, a = signal.bessel(order_num, highpass, 'highpass')
			fast_thz_signal = signal.filtfilt(b, a, fast_thz_signal)
		elif filter_name == 'lowpass':
			b, a = signal.bessel(order_num, lowpass, 'lowpass')
			fast_thz_signal = signal.filtfilt(b, a, fast_thz_signal)
		elif filter_name == 'bandstop':
			if lowpass > highpass:
				lowpass,highpass = highpass,lowpass
			b, a = signal.bessel(order_num, [lowpass, highpass], 'bandstop')
			fast_thz_signal = signal.filtfilt(b, a, fast_thz_signal)
	return  fast_thz_signal


#使用中值滤波
def useMedian(fast_thz_signal):
	sg,medianf,dwt,dwt_name,filter_name,lowpass,highpass = getParams()

	if medianf != 0:
		fast_thz_signal = signal.medfilt(fast_thz_signal, medianf)

	return fast_thz_signal

#获取小波变换算法参数
def obtain_dwt():
	dwt_level, dwt_threshold = 5, 'soft'
	path = r'C:\thztools\thz\Algconfig.yaml'
	try:
		if os.path.exists(path):
			with open(path, 'r', encoding='utf-8') as f:
				data = yaml.load(f, Loader=yaml.FullLoader)
		dwt_level = data['Signal_configs']['wave_level']
		dwt_threshold = data['Signal_configs']['wave_threshold']
		if dwt_level not in range(1,9):
			dwt_level = 5
	except Exception as e:
		with open(os.path.join("C:\\thztools", 'alglog.log'), 'a+', encoding='utf-8') as ff:
			ff.write(f'{time.strftime("%Y-%m-%d %X", time.localtime())} The error is : {e} \n')

	return dwt_level, dwt_threshold


#使用小波变换滤波
def useDwt(fast_thz_signal):
	sg, medianf, dwt, dwt_name, filter_name, lowpass, highpass = getParams()
	if dwt == 1:
		dwt_level, dwt_threshold = obtain_dwt()
		wavelists = []
		for family in pywt.families():
			for i in range(len(pywt.wavelist(family))):
				wavelists.append(pywt.wavelist(family)[i])
		if dwt_name in wavelists:
			num = fast_thz_signal.shape[0]
			mcoeffs = pywt.wavedec(fast_thz_signal, dwt_name, mode='symmetric', level=dwt_level)
			for k in range(1, len(mcoeffs)):
				value = np.sqrt(2 * np.log(fast_thz_signal.shape[0]))
				mcoeffs[k] = pywt.threshold(np.array(mcoeffs[k]), value=value, mode=dwt_threshold)
			fast_thz_signal = pywt.waverec(mcoeffs, wavelet=dwt_name, mode='symmetric')
			if fast_thz_signal.shape[0] > num:
				fast_thz_signal = fast_thz_signal[0:num]
			else:
				fast_thz_signal = np.append(fast_thz_signal, np.zeros(fast_thz_signal.shape[0] - num))

	return fast_thz_signal

#使用SG滤波
def useSG(fast_thz_signal):
	sg, medianf, dwt, dwt_name, filter_name, lowpass, highpass = getParams()
	if sg == 1:
		fast_thz_signal = signal.savgol_filter(fast_thz_signal, 5, 3)
	elif sg == 2:
		fast_thz_signal = signal.savgol_filter(fast_thz_signal, 25, 5)

	return fast_thz_signal


#定义椭圆滤波
def ellipticFilter(x, fp, fs, Ap, As, Fs, filter_name):
	wp = 2 * np.pi * fp / Fs
	ws = 2 * np.pi * fs / Fs

	N, wc = signal.ellipord(wp, ws, Ap, As)

	z, p = signal.ellip(N, Ap, As, wc, btype=filter_name)

	# After filtering the sequence x to obtain the sequence y
	y = signal.lfilter(z, p, x)

	return y

#定义切比学府滤波2
def chebyshevFilter2(x, fp, fs, rp, rs, Fs,filter_name):
	wp = 2 * np.pi * fp / Fs
	ws = 2 * np.pi * fs / Fs

	n, wn = signal.cheb2ord(wp / np.pi, ws / np.pi, rp, rs)
	b, a = signal.cheby2(n, rp, wp / np.pi,filter_name)

	y = signal.lfilter(b, a, x)

	return y

#定义切比学府滤波
def chebyshevFilter(x, fp, fs, rp, rs, Fs,filter_name):
	wp = 2 * np.pi * fp / Fs
	ws = 2 * np.pi * fs / Fs

	# Design a Chebyshev filter
	n, wn = signal.cheb1ord(wp / np.pi, ws / np.pi, rp, rs)
	b, a = signal.cheby1(n, rp, wp / np.pi,filter_name)

	# After filtering the sequence x to obtain the sequence y
	y = signal.lfilter(b, a, x)

	return y

# 切比雪夫滤波器
def apply_chebyshev_filter(data, fs, ftype, freqs=[], order=5, rp=3):
	nyq = 0.5 * fs

	if ftype == 'low_pass':
		assert len(freqs) == 1
		cut = freqs[0] / nyq
		b, a = signal.cheby1(order, rp, cut, btype='lowpass')
	elif ftype == 'high_pass':
		assert len(freqs) == 1
		cut = freqs[0] / nyq
		b, a = signal.cheby1(order, rp, cut, btype='highpass')
	elif ftype == 'band_pass':
		assert len(freqs) == 2
		lowcut, highcut = freqs[0] / nyq, freqs[1] / nyq
		b, a = signal.cheby1(order, rp, [lowcut, highcut], btype='bandpass')
	elif ftype == 'band_stop':
		assert len(freqs) == 2
		lowcut, highcut = freqs[0] / nyq, freqs[1] / nyq
		b, a = signal.cheby1(order, rp, [lowcut, highcut], btype='bandstop')

	filtered = signal.lfilter(b, a, data)
	return filtered
