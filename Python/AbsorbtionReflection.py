# -*- coding: utf-8 -*-
"""
Create Time: 2021/1/29 8:36
Author: Kevin
"""
import mmap
import contextlib
import time
import os
from Functions import Functions
#整个程序的入口
#打包环境:G:\PycharmProjects\ARexe_notf\venv\Scripts
if __name__ == '__main__':

	Functions.rmFiles()#删除每次启动存储的缓存
	num = 0

	while True:
		time.sleep(0.001)

		with contextlib.closing(mmap.mmap(-1, 6, tagname="matlab", access=mmap.ACCESS_READ)) as f:
			f.tell()
			s0 = f.read()
			s_length = int(s0[2]+s0[3]*256+s0[4]*256*256+s0[5]*256*256*256)

			if s0[0] == 0:
				continue

			try:
				with contextlib.closing(mmap.mmap(-1, 6 + s_length, tagname="matlab", access=mmap.ACCESS_WRITE)) as m:
					m.tell()
					s = m.read()

					if (len(s) - 6) != int(s[2] + s[3] * 256 + s[4] * 256 * 256 + s[5] * 256 * 256 * 256):
						continue

					if s[0] == 1 and s[1] == 1:#进行频谱计算
						if len(s) > 38:
							Functions.calFrequency(s, m, s_length)

					if s[0] == 2 and s[1] == 1:#进行吸收系数与折射率计算
						Functions.calAbsorptionAndReflection(s, m ,s_length)

					if s[0] == 4 and s[1] == 1:#进行解包操作
						Functions.calUnpackData(s, m, s_length,num)
						num = num + 1

			except Exception as e:
				with open(os.path.join(r'C:\thztools', 'alglog.log'), 'a+', encoding='utf-8') as ff:
					ff.write(f'{time.strftime("%Y-%m-%d %X", time.localtime())} Run error : {e} \n')


