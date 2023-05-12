# -*- coding: utf-8 -*-
"""
@Create Time: 2022/7/27 9:09
@Author: Kevin
@Python Version：3.7.6
"""
from API_Window import Ui_MainWindow
import sys
import pywt
import numpy as np
from PyQt5 import QtCore
from PyQt5.Qt import QIcon
from PyQt5.QtWidgets import QApplication, QMainWindow,QMessageBox
from ruamel.yaml import YAML
import os
#此文件主要实现功能是算法配置文件的参数修改
class Action(QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):
        super(Action, self).__init__(parent)
        self.setWindowFlags(QtCore.Qt.WindowCloseButtonHint | QtCore.Qt.WindowMinimizeButtonHint)
        try:
            icon = QIcon(r'C:\thztools\thz\color\kevin.ico')
            self.setWindowIcon(icon)
        except Exception as e:
            pass
        self.setupUi(self)
        self.init_ui()

        self.init_ui_size()


    def init_ui(self):
        self.t = np.array([])
        self.ref = np.array([])
        self.sample = np.array([])
        self.filter_num = -1
        self.MF = 1
        self.butter_mode = ''
        self.butter_low = 0.02
        self.butter_high = 0.8
        self.butter_order = 6
        self.dwt_mode = ''
        self.dwt_order = 5
        self.dwt_threshold = 'soft'
        self.dwt_threshold_value = ''
        self.dev_mode = ''
        self.dev_path = ''
        self.DGIF_LF = 0.24
        self.DGIF_HF = 0.56
        self.cheb_fp = 1
        self.cheb_fs = 3
        self.cheb_rp = 3
        self.cheb_rs = 40
        self.cheb_Fs = 100

        self.desktop = QApplication.desktop()
        self.screenRect = self.desktop.screenGeometry()
        self.height = self.screenRect.height()
        self.width = self.screenRect.width()
        self.w = 2560
        self.h = 1440

        self.setWindowTitle('青源峰达太赫兹信号滤波软件')

        self.obtain_wave_name()

        self.pushButton.clicked.connect(self.selectionchange)

        self.pushButton_2.clicked.connect(self.rejectFilter)



    def init_ui_size(self):

        if self.width == self.w and self.height == self.h:
            self.setFixedSize(595, 981)
            return
        ratio_w = self.width/self.w
        ratio_h = self.height/self.h

        self.resize(int(595*ratio_w), int(981*ratio_h))

        self.setFixedSize(int(595 * ratio_w), int(981 * ratio_h))

        size_all = [self.radioButton,self.radioButton_2,self.radioButton_3,self.radioButton_4,self.radioButton_5,self.radioButton_6,
                    self.radioButton_7,self.radioButton_8,self.radioButton_9,
                    self.label_9,self.lineEdit_8,self.comboBox,self.pushButton,self.layoutWidget,self.layoutWidget1,
                    self.layoutWidget2,self.layoutWidget3,self.layoutWidget4,self.layoutWidget5,self.comboBox_6,
                    self.layoutWidget6,self.label,self.lineEdit_2,self.pushButton_2,self.layoutWidget7]
        for i in range(len(size_all)):
            size_rect = size_all[i].geometry().getRect()
            size_all[i].setGeometry(QtCore.QRect(int(size_rect[0]*ratio_w), int(size_rect[1]*ratio_h), int(size_rect[2]*ratio_w) -1, int(size_rect[3]*ratio_h)-1))



    def obtain_wave_name(self):
        wavelists = []

        for family in pywt.families():
            for i in range(len(pywt.wavelist(family))):
                wavelists.append(pywt.wavelist(family)[i])

        wavelists = wavelists[0:106]
        self.comboBox_2.clear()
        for j in range(len(wavelists)):
            self.comboBox_2.addItem(str(wavelists[j]))


    def rejectFilter(self):
        self.filter_num = 0
        self.wirteYaml()

    def selectionchange(self):
        try:
            if self.radioButton.isChecked():
                self.filter_num = 1
                self.medianFilter()
            elif self.radioButton_2.isChecked():
                self.filter_num = 2
                self.butterFilter()
            elif self.radioButton_3.isChecked():
                self.filter_num = 3
                self.dwtFilter()
            elif self.radioButton_4.isChecked():
                self.filter_num = 4
                self.devFilter()
            elif self.radioButton_5.isChecked():
                self.filter_num = 5
                self.conFilter()
            elif self.radioButton_6.isChecked():
                self.filter_num = 6
                self.chebFilter()
            elif self.radioButton_7.isChecked():
                self.filter_num = 7
                self.besselFilter()
            elif self.radioButton_8.isChecked():
                self.filter_num = 8
                self.ellipticFilter()
            elif self.radioButton_9.isChecked():
                self.filter_num = 9
                self.chebFilter2()
            else:
                self.filter_num = -1
        except Exception as e:
            QMessageBox.warning(self, "Error", f"{e}", QMessageBox.Ok)

    def medianFilter(self):
        self.MF = int(self.lineEdit_2.text())
        if self.MF % 2 == 0:
            self.MF = 1
            QMessageBox.warning(self, "Error：", "中值滤波参数必须为奇数！", QMessageBox.Ok)
            self.lineEdit_2.setText(str(1))
        self.wirteYaml()

    def butterFilter(self):
        self.butter_mode = self.comboBox.currentText()
        self.butter_low = float(self.lineEdit.text())
        self.butter_high = float(self.lineEdit_3.text())
        self.butter_order = float(self.lineEdit_4.text())
        self.wirteYaml()

    def dwtFilter(self):
        self.dwt_mode = self.comboBox_2.currentText()
        self.dwt_order = int(self.lineEdit_5.text())
        self.dwt_threshold = self.comboBox_4.currentText()
        self.dwt_threshold_value = self.comboBox_5.currentText()
        self.wirteYaml()

    def devFilter(self):
        self.dev_mode = self.comboBox_3.currentText()
        self.dev_path = self.lineEdit_8.text()
        if self.dev_path == "":
            self.filter_num = 0
            QMessageBox.warning(self, "Warning", "请输入参考信号路径！", QMessageBox.Ok)

        elif os.path.exists(self.dev_path):
            if self.dev_mode == 'DGIF':
                self.DGIF_LF = float(self.lineEdit_6.text())
                self.DGIF_HF = float(self.lineEdit_7.text())
        else:
            self.filter_num = 0
            QMessageBox.warning(self, "Warning", "参考信号路径不正确！", QMessageBox.Ok)
            self.lineEdit_8.setText('')
        self.wirteYaml()

    def conFilter(self):
        self.wirteYaml()

    def chebFilter(self):
        self.cheb_fp = float(self.lineEdit_9.text())
        self.cheb_fs = float(self.lineEdit_10.text())
        self.cheb_rp = float(self.lineEdit_11.text())
        self.cheb_rs = float(self.lineEdit_12.text())
        self.cheb_Fs = float(self.lineEdit_13.text())
        self.butter_mode = self.comboBox_6.currentText()

        self.wirteYaml()

    def chebFilter2(self):
        self.cheb_fp = float(self.lineEdit_9.text())
        self.cheb_fs = float(self.lineEdit_10.text())
        self.cheb_rp = float(self.lineEdit_11.text())
        self.cheb_rs = float(self.lineEdit_12.text())
        self.cheb_Fs = float(self.lineEdit_13.text())
        self.butter_mode = self.comboBox_6.currentText()

        self.wirteYaml()

    def besselFilter(self):
        self.butter_mode = self.comboBox.currentText()
        self.butter_low = float(self.lineEdit.text())
        self.butter_high = float(self.lineEdit_3.text())
        self.butter_order = float(self.lineEdit_4.text())
        self.wirteYaml()

    def ellipticFilter(self):
        self.cheb_fp = float(self.lineEdit_9.text())
        self.cheb_fs = float(self.lineEdit_10.text())
        self.cheb_rp = float(self.lineEdit_11.text())
        self.cheb_rs = float(self.lineEdit_12.text())
        self.cheb_Fs = float(self.lineEdit_13.text())

        self.wirteYaml()


    def transParams(self):
        params = [self.filter_num,self.MF,self.butter_mode,self.butter_low,self.butter_high,self.butter_order,
                  self.dwt_mode,self.dwt_order,self.dwt_threshold,self.dwt_threshold_value,
                  self.dev_mode,self.dev_path,self.DGIF_LF,self.DGIF_HF]
        return params

    def wirteYaml(self):
        path = r'C:\thztools\thz\Algconfig.yaml'

        yaml = YAML()
        with open(path, "r", encoding='utf-8') as f:
            data = yaml.load(f)

        data['Signal_configs']['filter_type'] = self.filter_num

        if self.filter_num == 1:
            data['Signal_configs']['medianf'] = self.MF
        elif self.filter_num == 2:
            if self.butter_mode == 'Lowpass':
                data['Signal_configs']['filter_name'] = 'lowpass'
            elif self.butter_mode == 'Highpass':
                data['Signal_configs']['filter_name'] = 'highpass'
            elif self.butter_mode == 'Bandpass':
                data['Signal_configs']['filter_name'] = 'bandpass'
            elif self.butter_mode == 'Bandstop':
                data['Signal_configs']['filter_name'] = 'bandstop'
            data['Signal_configs']['butter_order'] = self.butter_order
            data['Signal_configs']['lowpass_value'] = self.butter_low
            data['Signal_configs']['highpass_value'] = self.butter_high
        elif self.filter_num == 3:
            data['Signal_configs']['dwt'] = 1
            data['Signal_configs']['wave_name'] = self.dwt_mode
            data['Signal_configs']['wave_level'] = self.dwt_order
            data['Signal_configs']['wave_threshold'] = self.dwt_threshold
        elif self.filter_num == 4:
            data['Signal_configs']['dev_mode'] = self.dev_mode
            data['Signal_configs']['dev_path'] = self.dev_path
            if self.dev_mode == 'DGIF':
                data['Signal_configs']['DGIF_LF'] = self.DGIF_LF
                data['Signal_configs']['DGIF_HF'] = self.DGIF_HF
        elif self.filter_num == 5:
            data['Signal_configs']['open_cov'] = 1
        elif self.filter_num == 6:
            data['Signal_configs']['fp'] = self.cheb_fp
            data['Signal_configs']['fs'] = self.cheb_fs
            data['Signal_configs']['rp'] = self.cheb_rp
            data['Signal_configs']['rs'] = self.cheb_rs
            data['Signal_configs']['Fs'] = self.cheb_Fs
            if self.butter_mode == 'Lowpass':
                data['Signal_configs']['filter_name'] = 'lowpass'
            elif self.butter_mode == 'Highpass':
                data['Signal_configs']['filter_name'] = 'highpass'
        elif self.filter_num == 7:
            if self.butter_mode == 'Lowpass':
                data['Signal_configs']['filter_name'] = 'lowpass'
            elif self.butter_mode == 'Highpass':
                data['Signal_configs']['filter_name'] = 'highpass'
            elif self.butter_mode == 'Bandpass':
                data['Signal_configs']['filter_name'] = 'bandpass'
            elif self.butter_mode == 'Bandstop':
                data['Signal_configs']['filter_name'] = 'bandstop'
            data['Signal_configs']['butter_order'] = self.butter_order
            data['Signal_configs']['lowpass_value'] = self.butter_low
            data['Signal_configs']['highpass_value'] = self.butter_high
        elif self.filter_num == 8:
            data['Signal_configs']['fp'] = self.cheb_fp
            data['Signal_configs']['fs'] = self.cheb_fs
            data['Signal_configs']['rp'] = self.cheb_rp
            data['Signal_configs']['rs'] = self.cheb_rs
            data['Signal_configs']['Fs'] = self.cheb_Fs
            if self.butter_mode == 'Lowpass':
                data['Signal_configs']['filter_name'] = 'lowpass'
            elif self.butter_mode == 'Highpass':
                data['Signal_configs']['filter_name'] = 'highpass'
        elif self.filter_num == 9:
            data['Signal_configs']['fp'] = self.cheb_fp
            data['Signal_configs']['fs'] = self.cheb_fs
            data['Signal_configs']['rp'] = self.cheb_rp
            data['Signal_configs']['rs'] = self.cheb_rs
            data['Signal_configs']['Fs'] = self.cheb_Fs
            if self.butter_mode == 'Lowpass':
                data['Signal_configs']['filter_name'] = 'lowpass'
            elif self.butter_mode == 'Highpass':
                data['Signal_configs']['filter_name'] = 'highpass'


        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(data, f)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    action = Action()
    action.show()
    sys.exit(app.exec_())
