# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'API_Window.ui'
#
# Created by: PyQt5 UI code generator 5.15.4
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets

#此文件是由ui文件转换而来
class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(563, 989)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.radioButton = QtWidgets.QRadioButton(self.centralwidget)
        self.radioButton.setGeometry(QtCore.QRect(80, 50, 115, 19))
        self.radioButton.setObjectName("radioButton")
        self.radioButton_2 = QtWidgets.QRadioButton(self.centralwidget)
        self.radioButton_2.setGeometry(QtCore.QRect(80, 140, 131, 19))
        self.radioButton_2.setObjectName("radioButton_2")
        self.radioButton_3 = QtWidgets.QRadioButton(self.centralwidget)
        self.radioButton_3.setGeometry(QtCore.QRect(80, 300, 115, 19))
        self.radioButton_3.setObjectName("radioButton_3")
        self.radioButton_4 = QtWidgets.QRadioButton(self.centralwidget)
        self.radioButton_4.setGeometry(QtCore.QRect(80, 470, 115, 19))
        self.radioButton_4.setObjectName("radioButton_4")
        self.label_9 = QtWidgets.QLabel(self.centralwidget)
        self.label_9.setGeometry(QtCore.QRect(100, 510, 111, 21))
        self.label_9.setAlignment(QtCore.Qt.AlignCenter)
        self.label_9.setObjectName("label_9")
        self.lineEdit_8 = QtWidgets.QLineEdit(self.centralwidget)
        self.lineEdit_8.setGeometry(QtCore.QRect(240, 510, 301, 21))
        self.lineEdit_8.setObjectName("lineEdit_8")
        self.pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton.setGeometry(QtCore.QRect(90, 910, 93, 28))
        self.pushButton.setObjectName("pushButton")
        self.radioButton_5 = QtWidgets.QRadioButton(self.centralwidget)
        self.radioButton_5.setGeometry(QtCore.QRect(80, 640, 115, 19))
        self.radioButton_5.setObjectName("radioButton_5")
        self.comboBox = QtWidgets.QComboBox(self.centralwidget)
        self.comboBox.setGeometry(QtCore.QRect(242, 169, 150, 21))
        self.comboBox.setObjectName("comboBox")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.layoutWidget = QtWidgets.QWidget(self.centralwidget)
        self.layoutWidget.setGeometry(QtCore.QRect(91, 171, 121, 101))
        self.layoutWidget.setObjectName("layoutWidget")
        self.gridLayout_2 = QtWidgets.QGridLayout(self.layoutWidget)
        self.gridLayout_2.setContentsMargins(0, 0, 0, 0)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.label_11 = QtWidgets.QLabel(self.layoutWidget)
        self.label_11.setAlignment(QtCore.Qt.AlignCenter)
        self.label_11.setObjectName("label_11")
        self.gridLayout_2.addWidget(self.label_11, 0, 0, 1, 1)
        self.label_2 = QtWidgets.QLabel(self.layoutWidget)
        self.label_2.setAlignment(QtCore.Qt.AlignCenter)
        self.label_2.setObjectName("label_2")
        self.gridLayout_2.addWidget(self.label_2, 1, 0, 1, 1)
        self.label_3 = QtWidgets.QLabel(self.layoutWidget)
        self.label_3.setAlignment(QtCore.Qt.AlignCenter)
        self.label_3.setObjectName("label_3")
        self.gridLayout_2.addWidget(self.label_3, 2, 0, 1, 1)
        self.label_4 = QtWidgets.QLabel(self.layoutWidget)
        self.label_4.setAlignment(QtCore.Qt.AlignCenter)
        self.label_4.setObjectName("label_4")
        self.gridLayout_2.addWidget(self.label_4, 3, 0, 1, 1)
        self.layoutWidget1 = QtWidgets.QWidget(self.centralwidget)
        self.layoutWidget1.setGeometry(QtCore.QRect(242, 189, 151, 88))
        self.layoutWidget1.setObjectName("layoutWidget1")
        self.gridLayout_5 = QtWidgets.QGridLayout(self.layoutWidget1)
        self.gridLayout_5.setContentsMargins(0, 0, 0, 0)
        self.gridLayout_5.setObjectName("gridLayout_5")
        self.lineEdit_3 = QtWidgets.QLineEdit(self.layoutWidget1)
        self.lineEdit_3.setObjectName("lineEdit_3")
        self.gridLayout_5.addWidget(self.lineEdit_3, 2, 0, 1, 1)
        self.lineEdit_4 = QtWidgets.QLineEdit(self.layoutWidget1)
        self.lineEdit_4.setObjectName("lineEdit_4")
        self.gridLayout_5.addWidget(self.lineEdit_4, 3, 0, 1, 1)
        self.lineEdit = QtWidgets.QLineEdit(self.layoutWidget1)
        self.lineEdit.setObjectName("lineEdit")
        self.gridLayout_5.addWidget(self.lineEdit, 1, 0, 1, 1)
        self.layoutWidget2 = QtWidgets.QWidget(self.centralwidget)
        self.layoutWidget2.setGeometry(QtCore.QRect(240, 330, 101, 110))
        self.layoutWidget2.setObjectName("layoutWidget2")
        self.gridLayout_3 = QtWidgets.QGridLayout(self.layoutWidget2)
        self.gridLayout_3.setContentsMargins(0, 0, 0, 0)
        self.gridLayout_3.setObjectName("gridLayout_3")
        self.comboBox_4 = QtWidgets.QComboBox(self.layoutWidget2)
        self.comboBox_4.setObjectName("comboBox_4")
        self.comboBox_4.addItem("")
        self.comboBox_4.addItem("")
        self.gridLayout_3.addWidget(self.comboBox_4, 4, 0, 1, 1)
        self.comboBox_5 = QtWidgets.QComboBox(self.layoutWidget2)
        self.comboBox_5.setObjectName("comboBox_5")
        self.comboBox_5.addItem("")
        self.comboBox_5.addItem("")
        self.comboBox_5.addItem("")
        self.gridLayout_3.addWidget(self.comboBox_5, 5, 0, 1, 1)
        self.comboBox_2 = QtWidgets.QComboBox(self.layoutWidget2)
        self.comboBox_2.setObjectName("comboBox_2")
        self.gridLayout_3.addWidget(self.comboBox_2, 1, 0, 1, 1)
        self.lineEdit_5 = QtWidgets.QLineEdit(self.layoutWidget2)
        self.lineEdit_5.setObjectName("lineEdit_5")
        self.gridLayout_3.addWidget(self.lineEdit_5, 3, 0, 1, 1)
        self.layoutWidget3 = QtWidgets.QWidget(self.centralwidget)
        self.layoutWidget3.setGeometry(QtCore.QRect(92, 332, 121, 101))
        self.layoutWidget3.setObjectName("layoutWidget3")
        self.gridLayout_6 = QtWidgets.QGridLayout(self.layoutWidget3)
        self.gridLayout_6.setContentsMargins(0, 0, 0, 0)
        self.gridLayout_6.setObjectName("gridLayout_6")
        self.label_5 = QtWidgets.QLabel(self.layoutWidget3)
        self.label_5.setAlignment(QtCore.Qt.AlignCenter)
        self.label_5.setObjectName("label_5")
        self.gridLayout_6.addWidget(self.label_5, 0, 0, 1, 1)
        self.label_6 = QtWidgets.QLabel(self.layoutWidget3)
        self.label_6.setAlignment(QtCore.Qt.AlignCenter)
        self.label_6.setObjectName("label_6")
        self.gridLayout_6.addWidget(self.label_6, 1, 0, 1, 1)
        self.label_7 = QtWidgets.QLabel(self.layoutWidget3)
        self.label_7.setAlignment(QtCore.Qt.AlignCenter)
        self.label_7.setObjectName("label_7")
        self.gridLayout_6.addWidget(self.label_7, 2, 0, 1, 1)
        self.label_8 = QtWidgets.QLabel(self.layoutWidget3)
        self.label_8.setAlignment(QtCore.Qt.AlignCenter)
        self.label_8.setObjectName("label_8")
        self.gridLayout_6.addWidget(self.label_8, 3, 0, 1, 1)
        self.layoutWidget4 = QtWidgets.QWidget(self.centralwidget)
        self.layoutWidget4.setGeometry(QtCore.QRect(242, 539, 101, 85))
        self.layoutWidget4.setObjectName("layoutWidget4")
        self.gridLayout_4 = QtWidgets.QGridLayout(self.layoutWidget4)
        self.gridLayout_4.setContentsMargins(0, 0, 0, 0)
        self.gridLayout_4.setObjectName("gridLayout_4")
        self.comboBox_3 = QtWidgets.QComboBox(self.layoutWidget4)
        self.comboBox_3.setObjectName("comboBox_3")
        self.comboBox_3.addItem("")
        self.comboBox_3.addItem("")
        self.gridLayout_4.addWidget(self.comboBox_3, 0, 0, 1, 1)
        self.lineEdit_6 = QtWidgets.QLineEdit(self.layoutWidget4)
        self.lineEdit_6.setFrame(True)
        self.lineEdit_6.setDragEnabled(False)
        self.lineEdit_6.setReadOnly(False)
        self.lineEdit_6.setObjectName("lineEdit_6")
        self.gridLayout_4.addWidget(self.lineEdit_6, 1, 0, 1, 1)
        self.lineEdit_7 = QtWidgets.QLineEdit(self.layoutWidget4)
        self.lineEdit_7.setObjectName("lineEdit_7")
        self.gridLayout_4.addWidget(self.lineEdit_7, 2, 0, 1, 1)
        self.layoutWidget5 = QtWidgets.QWidget(self.centralwidget)
        self.layoutWidget5.setGeometry(QtCore.QRect(92, 541, 121, 81))
        self.layoutWidget5.setObjectName("layoutWidget5")
        self.gridLayout_7 = QtWidgets.QGridLayout(self.layoutWidget5)
        self.gridLayout_7.setContentsMargins(0, 0, 0, 0)
        self.gridLayout_7.setObjectName("gridLayout_7")
        self.label_10 = QtWidgets.QLabel(self.layoutWidget5)
        self.label_10.setAlignment(QtCore.Qt.AlignCenter)
        self.label_10.setObjectName("label_10")
        self.gridLayout_7.addWidget(self.label_10, 0, 0, 1, 1)
        self.label_12 = QtWidgets.QLabel(self.layoutWidget5)
        self.label_12.setAlignment(QtCore.Qt.AlignCenter)
        self.label_12.setObjectName("label_12")
        self.gridLayout_7.addWidget(self.label_12, 1, 0, 1, 1)
        self.label_13 = QtWidgets.QLabel(self.layoutWidget5)
        self.label_13.setAlignment(QtCore.Qt.AlignCenter)
        self.label_13.setObjectName("label_13")
        self.gridLayout_7.addWidget(self.label_13, 2, 0, 1, 1)
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(91, 91, 121, 21))
        self.label.setObjectName("label")
        self.lineEdit_2 = QtWidgets.QLineEdit(self.centralwidget)
        self.lineEdit_2.setGeometry(QtCore.QRect(240, 90, 91, 21))
        self.lineEdit_2.setObjectName("lineEdit_2")
        self.radioButton_6 = QtWidgets.QRadioButton(self.centralwidget)
        self.radioButton_6.setGeometry(QtCore.QRect(80, 670, 141, 19))
        self.radioButton_6.setObjectName("radioButton_6")
        self.layoutWidget6 = QtWidgets.QWidget(self.centralwidget)
        self.layoutWidget6.setGeometry(QtCore.QRect(240, 730, 101, 161))
        self.layoutWidget6.setObjectName("layoutWidget6")
        self.gridLayout_8 = QtWidgets.QGridLayout(self.layoutWidget6)
        self.gridLayout_8.setContentsMargins(0, 0, 0, 0)
        self.gridLayout_8.setObjectName("gridLayout_8")
        self.lineEdit_9 = QtWidgets.QLineEdit(self.layoutWidget6)
        self.lineEdit_9.setObjectName("lineEdit_9")
        self.gridLayout_8.addWidget(self.lineEdit_9, 0, 0, 1, 1)
        self.lineEdit_10 = QtWidgets.QLineEdit(self.layoutWidget6)
        self.lineEdit_10.setObjectName("lineEdit_10")
        self.gridLayout_8.addWidget(self.lineEdit_10, 1, 0, 1, 1)
        self.lineEdit_11 = QtWidgets.QLineEdit(self.layoutWidget6)
        self.lineEdit_11.setObjectName("lineEdit_11")
        self.gridLayout_8.addWidget(self.lineEdit_11, 2, 0, 1, 1)
        self.lineEdit_12 = QtWidgets.QLineEdit(self.layoutWidget6)
        self.lineEdit_12.setObjectName("lineEdit_12")
        self.gridLayout_8.addWidget(self.lineEdit_12, 3, 0, 1, 1)
        self.lineEdit_13 = QtWidgets.QLineEdit(self.layoutWidget6)
        self.lineEdit_13.setObjectName("lineEdit_13")
        self.gridLayout_8.addWidget(self.lineEdit_13, 4, 0, 1, 1)
        self.pushButton_2 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_2.setGeometry(QtCore.QRect(200, 910, 93, 28))
        self.pushButton_2.setObjectName("pushButton_2")
        self.radioButton_7 = QtWidgets.QRadioButton(self.centralwidget)
        self.radioButton_7.setGeometry(QtCore.QRect(240, 140, 121, 19))
        self.radioButton_7.setObjectName("radioButton_7")
        self.radioButton_8 = QtWidgets.QRadioButton(self.centralwidget)
        self.radioButton_8.setGeometry(QtCore.QRect(390, 670, 115, 19))
        self.radioButton_8.setObjectName("radioButton_8")
        self.comboBox_6 = QtWidgets.QComboBox(self.centralwidget)
        self.comboBox_6.setGeometry(QtCore.QRect(240, 700, 101, 22))
        self.comboBox_6.setObjectName("comboBox_6")
        self.comboBox_6.addItem("")
        self.comboBox_6.addItem("")
        self.layoutWidget7 = QtWidgets.QWidget(self.centralwidget)
        self.layoutWidget7.setGeometry(QtCore.QRect(81, 700, 121, 191))
        self.layoutWidget7.setObjectName("layoutWidget7")
        self.gridLayout = QtWidgets.QGridLayout(self.layoutWidget7)
        self.gridLayout.setContentsMargins(0, 0, 0, 0)
        self.gridLayout.setObjectName("gridLayout")
        self.label_19 = QtWidgets.QLabel(self.layoutWidget7)
        self.label_19.setAlignment(QtCore.Qt.AlignCenter)
        self.label_19.setObjectName("label_19")
        self.gridLayout.addWidget(self.label_19, 0, 0, 1, 1)
        self.label_14 = QtWidgets.QLabel(self.layoutWidget7)
        self.label_14.setAlignment(QtCore.Qt.AlignCenter)
        self.label_14.setObjectName("label_14")
        self.gridLayout.addWidget(self.label_14, 1, 0, 1, 1)
        self.label_15 = QtWidgets.QLabel(self.layoutWidget7)
        self.label_15.setAlignment(QtCore.Qt.AlignCenter)
        self.label_15.setObjectName("label_15")
        self.gridLayout.addWidget(self.label_15, 2, 0, 1, 1)
        self.label_18 = QtWidgets.QLabel(self.layoutWidget7)
        self.label_18.setAlignment(QtCore.Qt.AlignCenter)
        self.label_18.setObjectName("label_18")
        self.gridLayout.addWidget(self.label_18, 3, 0, 1, 1)
        self.label_17 = QtWidgets.QLabel(self.layoutWidget7)
        self.label_17.setAlignment(QtCore.Qt.AlignCenter)
        self.label_17.setObjectName("label_17")
        self.gridLayout.addWidget(self.label_17, 4, 0, 1, 1)
        self.label_16 = QtWidgets.QLabel(self.layoutWidget7)
        self.label_16.setAlignment(QtCore.Qt.AlignCenter)
        self.label_16.setObjectName("label_16")
        self.gridLayout.addWidget(self.label_16, 5, 0, 1, 1)
        self.radioButton_9 = QtWidgets.QRadioButton(self.centralwidget)
        self.radioButton_9.setGeometry(QtCore.QRect(240, 670, 131, 21))
        self.radioButton_9.setObjectName("radioButton_9")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 563, 26))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.radioButton.setText(_translate("MainWindow", "中值滤波"))
        self.radioButton_2.setText(_translate("MainWindow", "巴特沃斯滤波"))
        self.radioButton_3.setText(_translate("MainWindow", "小波变换"))
        self.radioButton_4.setText(_translate("MainWindow", "反卷积滤波"))
        self.label_9.setText(_translate("MainWindow", "参考信号路径"))
        self.pushButton.setText(_translate("MainWindow", "OK"))
        self.radioButton_5.setText(_translate("MainWindow", "卷积滤波"))
        self.comboBox.setItemText(0, _translate("MainWindow", "Lowpass"))
        self.comboBox.setItemText(1, _translate("MainWindow", "Highpass"))
        self.comboBox.setItemText(2, _translate("MainWindow", "Bandpass"))
        self.comboBox.setItemText(3, _translate("MainWindow", "Bandstop"))
        self.label_11.setText(_translate("MainWindow", "滤波选择"))
        self.label_2.setText(_translate("MainWindow", "低通滤波阈值"))
        self.label_3.setText(_translate("MainWindow", "高通滤波阈值"))
        self.label_4.setText(_translate("MainWindow", "滤波阶数"))
        self.lineEdit_3.setText(_translate("MainWindow", "0.8"))
        self.lineEdit_4.setText(_translate("MainWindow", "6"))
        self.lineEdit.setText(_translate("MainWindow", "0.02"))
        self.comboBox_4.setItemText(0, _translate("MainWindow", "soft"))
        self.comboBox_4.setItemText(1, _translate("MainWindow", "hard"))
        self.comboBox_5.setItemText(0, _translate("MainWindow", "sqtwolog"))
        self.comboBox_5.setItemText(1, _translate("MainWindow", "stein"))
        self.comboBox_5.setItemText(2, _translate("MainWindow", "maxmin"))
        self.lineEdit_5.setText(_translate("MainWindow", "5"))
        self.label_5.setText(_translate("MainWindow", "小波函数"))
        self.label_6.setText(_translate("MainWindow", "小波分解层数"))
        self.label_7.setText(_translate("MainWindow", "小波阈值选择"))
        self.label_8.setText(_translate("MainWindow", "小波阈值方法"))
        self.comboBox_3.setItemText(0, _translate("MainWindow", "DGIF"))
        self.comboBox_3.setItemText(1, _translate("MainWindow", "SD"))
        self.lineEdit_6.setText(_translate("MainWindow", "0.24"))
        self.lineEdit_7.setText(_translate("MainWindow", "0.56"))
        self.label_10.setText(_translate("MainWindow", "反卷积的选择"))
        self.label_12.setText(_translate("MainWindow", "DGIF中LF参数"))
        self.label_13.setText(_translate("MainWindow", "DGIF中HF参数"))
        self.label.setText(_translate("MainWindow", "中值滤波参数"))
        self.lineEdit_2.setText(_translate("MainWindow", "7"))
        self.radioButton_6.setText(_translate("MainWindow", "ChebyⅠ型滤波"))
        self.lineEdit_9.setText(_translate("MainWindow", "1"))
        self.lineEdit_10.setText(_translate("MainWindow", "3"))
        self.lineEdit_11.setText(_translate("MainWindow", "3"))
        self.lineEdit_12.setText(_translate("MainWindow", "40"))
        self.lineEdit_13.setText(_translate("MainWindow", "100"))
        self.pushButton_2.setText(_translate("MainWindow", "取消滤波"))
        self.radioButton_7.setText(_translate("MainWindow", "贝塞尔滤波"))
        self.radioButton_8.setText(_translate("MainWindow", "椭圆滤波"))
        self.comboBox_6.setItemText(0, _translate("MainWindow", "Lowpass"))
        self.comboBox_6.setItemText(1, _translate("MainWindow", "Highpass"))
        self.label_19.setText(_translate("MainWindow", "滤波选择"))
        self.label_14.setText(_translate("MainWindow", "通带截止频率"))
        self.label_15.setText(_translate("MainWindow", "阻带截止频率"))
        self.label_18.setText(_translate("MainWindow", "边带区衰减DB"))
        self.label_17.setText(_translate("MainWindow", "截止区衰减DB"))
        self.label_16.setText(_translate("MainWindow", "采样率"))
        self.radioButton_9.setText(_translate("MainWindow", "ChebyⅡ型滤波"))
