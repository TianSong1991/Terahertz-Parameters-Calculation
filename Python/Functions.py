# -*- coding: utf-8 -*-
"""
Create Time: 2021/10/26 9:19
Author: Kevin
Python Version：3.7.6
"""
import mmap
import contextlib
import struct
import time
import numpy as np
from FrequencyAnalysisUtil import *
from SampleProperties import SampleProperties
import os
import shutil
from scipy import signal
from AtrProperties import AtrProperties
from FrequencyAnalysisUtil import truncateTimeSeries
import ReflectionProperties
from PIL import Image
from Filters import *

class Functions():
	def __init__(self):
		self.timeSeriesList = np.array([])

	@staticmethod#删除缓存文件
	def rmFiles():
		path = r'C:\Windows\Temp\thzcache'
		zeroYaml()
		writePid()
		if not os.path.exists(path):
			os.makedirs(path)
		try:
			if len(os.listdir(path)) > 2:
				for file in os.listdir(path):
					delete_time = time.time()
					path1 = os.path.join(path,file)
					file_time = os.path.getctime(path1)
					if delete_time - file_time > 60 * 10:
						shutil.rmtree(path1)
		except Exception as e:
			with open(os.path.join("C:\\thztools", 'alglog.log'), 'a+', encoding='utf-8') as ff:
				ff.write(f'{time.strftime("%Y-%m-%d %X", time.localtime())} The error is : {e} \n')



	@staticmethod#获取数据
	def getData(f,ns_real,ns_imag,e_real,e_imag,thz_max):
		num_f = len(f)
		num_ns = len(ns_real)
		if num_f > num_ns:
			f = f[0:num_ns]
		elif num_ns > num_f:
			ns_real = ns_real[0:num_f]
			ns_imag = ns_imag[0:num_f]
			e_real = e_real[0:num_f]
			e_imag = e_imag[0:num_f]

		ns_real = ns_real[f <= thz_max]
		ns_imag = ns_imag[f <= thz_max]
		e_real = e_real[f <= thz_max]
		e_imag = e_imag[f <= thz_max]
		f = f[f <= thz_max]

		return f,ns_real,ns_imag,e_real,e_imag

	@staticmethod#计算频率
	def calFrequency(s, m ,s_length):
		m.seek(1)
		m.write(bytes([2]))
		m.seek(0)

		ref_value1 = np.array([])
		list_length = (len(s) - 6) // 8
		python_value = struct.unpack("d" * list_length, s[6:(6 + list_length * 8)])
		python_value = np.asarray(python_value).reshape(list_length)

		t_value = python_value[0:int(list_length / 2)]
		ref_value = python_value[int(list_length / 2):]
		filter_type, show_filer = 0,0


		try:#开始滤波计算
			filter_type,show_filer = freq_params()

			if filter_type != 0:
				if filter_type == 1:
					ref_value1 = useMedian(ref_value)
				elif filter_type == 2:
					ref_value1 = useButter(ref_value)
				elif filter_type == 3:
					ref_value1 = useDwt(ref_value)
				elif filter_type == 4:
					ref_value1 = useDev(ref_value)
				elif filter_type == 5:
					ref_value1 = useConv(ref_value)
				elif filter_type == 6:
					ref_value1 = useCheb(ref_value)
				elif filter_type == 7:
					ref_value1 = useBessel(ref_value)
				elif filter_type == 8:
					ref_value1 = useElliptic(ref_value)
				elif filter_type == 9:
					ref_value1 = useCheb2(ref_value)


		except Exception as e:
			with open(os.path.join("C:\\thztools", 'alglog.log'), 'a+', encoding='utf-8') as ff:
				ff.write(f'{time.strftime("%Y-%m-%d %X", time.localtime())} The error is : {e} \n')


		if filter_type != 0 and len(ref_value1) != 0:
			fRange, xfRange = convertToFrequency(t_value, ref_value1)
		else:
			fRange, xfRange = convertToFrequency(t_value, ref_value)
		xfRange = np.array(xfRange)
		fRange = np.array(fRange)
		xfRange = xfRange[0:int(np.floor(xfRange.shape[0] / 2))]
		fRange = fRange[0:int(np.floor(fRange.shape[0] / 2))]
		if np.max(xfRange[fRange>0.1]) < 0:
			xfRange[fRange>0.1] = xfRange[fRange>0.1] - np.max(xfRange[fRange>0.1])

		xfRange = xfRange[fRange <= 10]
		fRange = fRange[fRange <= 10]


		data = np.append(fRange, xfRange)


		if show_filer == 1 and len(ref_value1) != 0:
			ref_value = ref_value1

		data = np.append(data, ref_value)
		data2bytes = struct.pack('d' * data.shape[0], *data)
		data_length = len(data2bytes)


		if data_length < s_length:
			m.seek(2)
			m.write(data_length.to_bytes(4, byteorder='little'))
			m.seek(6)
			m.write(data2bytes)
			m.seek(1)
			m.write(bytes([3]))
			m.seek(0)
		else:
			with contextlib.closing(mmap.mmap(-1, 6 + data_length, tagname="matlab", access=mmap.ACCESS_WRITE)) as mm:
				mm.seek(2)
				mm.write(data_length.to_bytes(4, byteorder='little'))
				mm.seek(6)
				mm.write(data2bytes)
				mm.seek(1)
				mm.write(bytes([3]))
				mm.seek(0)

	@staticmethod#计算吸收系数与折射率
	def calAbsorptionAndReflection(s, m ,s_length):
		m.seek(1)
		m.write(bytes([2]))
		m.seek(0)

		choose_type =struct.unpack("d",s[6:14])
		choose_type = np.asarray(choose_type)
		alg_type = struct.unpack("d",s[14:22])
		alg_type = np.asarray(alg_type)
		d = struct.unpack("d", s[22:30])
		d = np.asarray(d) * 1000
		n_liquid = struct.unpack("d", s[30:38])
		n_liquid = np.asarray(n_liquid)
		theta = struct.unpack("d", s[38:46])
		theta = np.asarray(theta) * np.pi / 180
		p = struct.unpack("d", s[46:54])
		p = np.asarray(p)
		list_length = (len(s) - 54) // 8
		python_value = struct.unpack("d" * list_length, s[54:(54 + list_length * 8)])
		python_value = np.asarray(python_value).reshape(list_length)
		t_value = python_value[0:int(list_length / 3)]
		ref_value = python_value[int(list_length / 3): int(list_length / 3) * 2]
		samp_value = python_value[int(list_length / 3) * 2:]

		f = np.array([])
		ns_real = np.array([])
		absorption = np.array([])
		absorption_ratio = np.array([])
		refraction_ratio = np.array([])
		ns_imag = np.array([])
		e_real = np.array([])
		e_imag = np.array([])
		ff = np.array([])
		psd = np.array([])
		data = np.array([])



		if d == 0:
			d = 1
		sp = SampleProperties(t_value, ref_value, t_value, samp_value, d)
		f = np.array(sp.f)
		f = f[0:int(np.floor(f.shape[0] / 2))]

		alg_type0 = int(alg_type / 1000)
		alg_type1 = int(alg_type % 1000)

		thz_max= obtain_Fre()

		if alg_type0 in [101,105,106,107] and alg_type1 in [101,105,106,107]:

			if choose_type == 0:
				ns_real = np.array(sp.calRefractiveIndex())
				ns_imag = np.array(sp.calExtinction())

				e_real = np.power(ns_real, 2) - np.power(ns_imag, 2)
				e_imag = 2 * ns_real * ns_imag

				ns_real = ns_real[0:int(np.floor(ns_real.shape[0] / 2))]
				ns_imag = ns_imag[0:int(np.floor(ns_imag.shape[0] / 2))]
				e_real = e_real[0:int(np.floor(e_real.shape[0] / 2))]
				e_imag = e_imag[0:int(np.floor(e_imag.shape[0] / 2))]

				f, ns_real, ns_imag, e_real, e_imag = Functions.getData(f, ns_real, ns_imag, e_real, e_imag,thz_max)

			elif choose_type == 2:
				rp = AtrProperties(t_value, ref_value, samp_value, n_liquid, lowFreq=np.min(f), highFreq=np.max(f),
								   polarity=p)
				ns = rp.refracIndex()

				ns_real = np.array(np.real(ns))
				ns_imag = np.array(np.imag(ns))

				e_real = np.power(ns_real, 2) - np.power(ns_imag, 2)
				e_imag = 2 * ns_real * ns_imag

				f, ns_real, ns_imag, e_real, e_imag = Functions.getData(f, ns_real, ns_imag, e_real, e_imag,thz_max)

			else:
				lowLim = 0
				upLim = 1000
				truncT, truncRef = truncateTimeSeries(t_value, ref_value, lowLim, upLim)
				_, truncSamp = truncateTimeSeries(t_value, samp_value, lowLim, upLim)
				rp = ReflectionProperties.ReflectionProperties(truncT, truncRef, truncSamp, theta, lowFreq=np.min(f),
															   highFreq=np.max(f), polarity=p)
				ns = rp.refracIndex()

				ns_real = np.array(np.real(ns))
				ns_imag = np.array(np.imag(ns))

				e_real = np.power(ns_real, 2) - np.power(ns_imag, 2)
				e_imag = 2 * ns_real * ns_imag

				f, ns_real, ns_imag, e_real, e_imag = Functions.getData(f, ns_real, ns_imag, e_real, e_imag,thz_max)

		else:
			absorption = np.array(sp.calAbsorptionRate())
			ns_real = np.array(sp.calRefractiveIndex())

			absorption = absorption[0:int(np.floor(absorption.shape[0] / 2))]
			ns_real = ns_real[0:int(np.floor(ns_real.shape[0] / 2))]
			absorption_ratio = np.exp(-absorption * d)
			absorption_ratio[absorption_ratio > 1] = 0
			absorption_ratio = absorption_ratio * 100
			refraction_ratio = sp.refraction_ratio
			refraction_ratio = refraction_ratio[0:int(np.floor(refraction_ratio.shape[0] / 2))]
			ff, psd = signal.welch(samp_value, fs=100, nperseg=samp_value.shape[0])
			psd = 10 * np.log10(psd)

			ns_real = ns_real[f <= thz_max]
			absorption = absorption[f <= thz_max]
			absorption_ratio = absorption_ratio[f <= thz_max]
			refraction_ratio = refraction_ratio[f <= thz_max]
			f = f[f <= thz_max]
			psd = psd[ff <= thz_max]
			ff = ff[ff <= thz_max]

			max_r = np.max(refraction_ratio)
			min_r = np.min(refraction_ratio)
			refraction_ratio = (refraction_ratio - min_r) / (max_r - min_r)

		choose_lists = [100, 101, 102, 103, 104, 105, 106, 107, 200, 201]
		data_lists = []
		data_lists.append(f)
		data_lists.append(ns_real)
		data_lists.append(absorption)
		data_lists.append(absorption_ratio)
		data_lists.append(refraction_ratio)
		data_lists.append(ns_imag)
		data_lists.append(e_real)
		data_lists.append(e_imag)
		data_lists.append(ff)
		data_lists.append(psd)

		index0 = choose_lists.index(alg_type0)
		if alg_type0 < 150:
			f_index = 0
			data = np.append(data,np.array([1111, 100]))
			data = np.append(data, f)
			data = np.append(data, np.array([1111, alg_type0]))
			data = np.append(data, data_lists[index0])
		else:
			f_index = 1
			data = np.append(data, np.array([1111, 200]))
			data = np.append(data, ff)
			data = np.append(data, np.array([1111, 201]))
			data = np.append(data, psd)

		index1 = choose_lists.index(alg_type1)
		if alg_type1 < 150:
			if f_index == 1:
				data = np.append(data,np.array([1111, 100]))
				data = np.append(data, f)
				data = np.append(data, np.array([1111, alg_type1]))
				data = np.append(data, data_lists[index1])
			else:
				data = np.append(data, np.array([1111, alg_type1]))
				data = np.append(data, data_lists[index1])
		else:
			data = np.append(data,np.array([1111, 200]))
			data = np.append(data, ff)
			data = np.append(data, np.array([1111, 201]))
			data = np.append(data, psd)

		data2bytes = struct.pack('d' * data.shape[0], *data)
		data_length = len(data2bytes)

		if data_length < s_length:
			m.seek(2)
			m.write(data_length.to_bytes(4, byteorder='little'))
			m.seek(6)
			m.write(data2bytes)
			m.seek(1)
			m.write(bytes([3]))
			m.seek(0)
		else:
			with contextlib.closing(mmap.mmap(-1, 6 + data_length, tagname="matlab", access=mmap.ACCESS_WRITE)) as mm:
				mm.seek(2)
				mm.write(data_length.to_bytes(4, byteorder='little'))
				mm.seek(6)
				mm.write(data2bytes)
				mm.seek(1)
				mm.write(bytes([3]))
				mm.seek(0)

	@staticmethod#进行物质识别
	def calClassification(s, m, model,trainSampNames):
		m.seek(1)
		m.write(bytes([2]))
		m.seek(0)

		list_length = (len(s) - 6) // 8
		python_value = struct.unpack("d" * list_length, s[6:(6 + list_length * 8)])
		python_value = np.asarray(python_value).reshape(list_length)
		t_value = python_value[0:int(list_length / 2)]
		ref_value = python_value[int(list_length / 2):]

		start_frequency = 0.1
		end_frequency = 2.0

		fRange, xfRange = convertToFrequency(t_value, ref_value)
		fRange = np.array(fRange)
		xfRange = np.array(xfRange)
		xfRange = xfRange[(fRange > start_frequency) & (fRange < end_frequency)]

		width = classification_width()
		inputShape = (1, width, 1)
		xfRange = np.array(Image.fromarray(xfRange).resize((1, width))).reshape(width)
		seriesList = []
		series = xfRange.reshape(inputShape)
		seriesList.append(series)
		seriesList = np.array(seriesList)

		# Classify all the series from the input file
		predicts = model.predict(seriesList)
		predicts_index = np.argmax(predicts)
		result = trainSampNames[predicts_index] + str(' ') + str(predicts.max())

		samp_name = bytes(result,'utf-8')
		data_length = len(samp_name)

		m.seek(2)
		m.write(data_length.to_bytes(4, byteorder='little'))
		m.seek(6)
		m.write(samp_name)
		m.seek(1)
		m.write(bytes([3]))
		m.seek(0)

	@staticmethod#进行解包算法解析
	def calUnpackData(s, m ,s_length,num):
		time0 = time.time()
		m.seek(1)
		m.write(bytes([2]))
		m.seek(0)
		nPoints = struct.unpack("i", s[6:10])
		nPoints = np.asarray(nPoints)
		ps = struct.unpack("i", s[10:14])
		ps = np.asarray(ps)
		mean_peaks = struct.unpack("d", s[14:22])
		mean_peaks = np.asarray(mean_peaks)
		python_value = np.array(list(s[22:]))
		save_open = save_txt()
		if save_open == 1:
			if not os.path.exists(r'C:\thztools\data'):
				os.mkdir(r'C:\thztools\data')
			if len(os.listdir(r'C:\thztools\data')) <= 50:
				np.savetxt(os.path.join(r'C:\thztools\data',time.strftime('%Y%m%d%H%M%S',time.localtime(time.time()))+'.txt'),python_value)
		delay_line = obtain_config()
		fast_thz_time, fast_thz_signal = unpackData(python_value, int(nPoints),int(ps),delay_line)

		try:
			move_second = move_secondPeakconfig()
			if move_second == 1:
				fast_thz_signal = move_secondPeak1(fast_thz_signal)
		except Exception as e:
			with open(os.path.join("C:\\thztools", 'alglog.log'), 'a+', encoding='utf-8') as ff:
				ff.write(f'{time.strftime("%Y-%m-%d %X", time.localtime())} The move second peak error is : {e} \n')

		try:#原始信号滤波
			filter_type = select_filter()

			if filter_type != 0:

				if filter_type == 1:
					fast_thz_signal1 = useMedian(fast_thz_signal)

				elif filter_type == 2:
					fast_thz_signal1 = useButter(fast_thz_signal)

				elif filter_type == 3:
					fast_thz_signal1 = useDwt(fast_thz_signal)

				elif filter_type == 4:
					fast_thz_signal1 = useDev(fast_thz_signal)

				elif filter_type == 5:
					fast_thz_signal1 = useConv(fast_thz_signal)

				elif filter_type == 6:
					fast_thz_signal1 = useCheb(fast_thz_signal)

				elif filter_type == 7:
					fast_thz_signal1 = useBessel(fast_thz_signal)

				elif filter_type == 8:
					fast_thz_signal1 = useElliptic(fast_thz_signal)

				elif filter_type == 9:
					fast_thz_signal1 = useCheb2(fast_thz_signal)


		except Exception as e:
			with open(os.path.join("C:\\thztools", 'alglog.log'), 'a+', encoding='utf-8') as ff:
				ff.write(f'{time.strftime("%Y-%m-%d %X", time.localtime())} The error is : {e} \n')

		try:
			show_filer = show_filterSignal()
			if filter_type != 0 and show_filer == 1:
				fast_thz_signal = fast_thz_signal1
		except Exception as e:
			with open(os.path.join("C:\\thztools", 'alglog.log'), 'a+', encoding='utf-8') as ff:
				ff.write(f'{time.strftime("%Y-%m-%d %X", time.localtime())} The error is : {e} \n')

		d = np.floor(1/(0.5*ps*6/300))

		index, value = signal.find_peaks(fast_thz_signal, height=float(mean_peaks * np.max(fast_thz_signal)), distance=d)


		data = np.append(np.array(fast_thz_time), np.array(fast_thz_signal))
		data = np.append(data,np.array(index))
		data2bytes = struct.pack('d' * data.shape[0], *data)
		data_length = len(data2bytes)

		if data_length < s_length:
			m.seek(2)
			m.write(data_length.to_bytes(4, byteorder='little'))
			m.seek(6)
			m.write(data2bytes)
			m.seek(1)
			m.write(bytes([3]))
			m.seek(0)
		else:
			with contextlib.closing(mmap.mmap(-1, 6 + data_length, tagname="matlab", access=mmap.ACCESS_WRITE)) as mm:
				mm.seek(2)
				mm.write(data_length.to_bytes(4, byteorder='little'))
				mm.seek(6)
				mm.write(data2bytes)
				mm.seek(1)
				mm.write(bytes([3]))
				mm.seek(0)
		with open(os.path.join("C:\\thztools", 'alglog.log'), 'a+', encoding='utf-8') as ff:
			ff.write(f'{time.strftime("%Y-%m-%d %X", time.localtime())} Unpack cost time is : {time.time() - time0} \n')
