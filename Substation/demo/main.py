import sys, os
from mainwindow import Ui_MainWindow
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QMessageBox, QLineEdit, QWidget, QApplication
from PyQt5.QtGui import QPalette, QBrush, QPixmap
import numpy as np
import cv2
from PyQt5.QtCore import QThread, pyqtSignal
import glob, time
# pyuic5 -o mainwindow.py mainwindow.ui

import matplotlib.pyplot as plt
sys.path.insert(0, '/home/songwenlong/qt_demo/meter_reco/detect_rectify_read')
sys.path.insert(0, "meter_reco/ransac")
sys.path.insert(0, "meter_reco/mmdetection")
sys.path.insert(0, "/home/songwenlong/qt_demo/defect_det")
from meter_reco import detect_rectify_read_an_image
from defect_detection_and_recognation0 import detect_and_reco_defect, tran_and_pad_img

# @colorful('blueGreen')
class DemoWindow(QWidget, Ui_MainWindow):
    def __init__(self, MainWindow):
        super().__init__()
        self.setupUi(MainWindow)
        self.setupbt()
        self.img_path = ('', '')
        self.folder_path = ''
        self.worker_meter = WorkThread_meter()
        self.worker_meter.trigger.connect(self.display)
        self.worker_defect = WorkThread_defect()
        self.worker_defect.trigger.connect(self.display)

    def setupbt(self):
        self.pushButton.clicked.connect(self.defect_detect)
        self.pushButton_2.clicked.connect(self.meter_reco)
        self.pushButton_3.clicked.connect(self.meter_reco_folder)
        self.pushButton_4.clicked.connect(self.defect_det_folder)
        self.pushButton_5.clicked.connect(self.clear_label)
        self.pushButton_6.clicked.connect(QtCore.QCoreApplication.instance().quit)

    def defect_detect(self):
        self.load_image()
        input_img_path = self.img_path
        QApplication.processEvents()
        if self.img_path[0] == '':
            QMessageBox.about(self, '提示', '请载入需要检测的图片！')
        else:
            output_img_path, parm = self.model_detect(input_img_path)
            QMessageBox.about(self, '提示', '检测完成！')
            pix = QPixmap(output_img_path)
            self.label.setPixmap(pix)
            self.label.setScaledContents(True)

    
    def meter_reco(self):
        self.load_image()
        input_img_path = self.img_path
        QApplication.processEvents()
        if self.img_path[0] == '':
            QMessageBox.about(self, '提示', '请载入需要识别的图片！')
        else:
            output_img_path, parm = self.model_reco(input_img_path)
            QMessageBox.about(self, '提示', '识别完成！')
            self.display(output_img_path)

    def load_image(self):
        self.img_path = QtWidgets.QFileDialog.getOpenFileName(self,
              "getOpenFileName","./","All Files (*)")
        if self.img_path[0] != '':
            tran_and_pad_img(self.img_path[0], './show_tmp.jpg')
            QMessageBox.about(self, '提示', '加载完成！')
            self.display('./show_tmp.jpg')

    def meter_reco_folder(self):
        self.folder_path = QtWidgets.QFileDialog.getExistingDirectory(self, "getExistingDirectory", "./")
        imgs_to_reco = glob.glob(os.path.join(self.folder_path, '*.*[png|jpg|jpeg]'))
        print(imgs_to_reco)
        if not imgs_to_reco or self.folder_path == '':
            QMessageBox.about(self, '提示', '文件夹中无文件！')
        else:
            self.worker_meter.run(imgs_to_reco)
            QMessageBox.about(self, '提示', '演示完成！')

    def defect_det_folder(self):
        self.folder_path = QtWidgets.QFileDialog.getExistingDirectory(self, "getExistingDirectory", "./")
        imgs_to_reco = glob.glob(os.path.join(self.folder_path, '*.*[png|jpg|jpeg]'))
        print(imgs_to_reco)
        if not imgs_to_reco or self.folder_path == '':
            QMessageBox.about(self, '提示', '文件夹中无文件！')
        else:
            self.worker_defect.run(imgs_to_reco)
            QMessageBox.about(self, '提示', '演示完成！')

    def clear_label(self):
        self.img_path = ('', '')
        self.label.setPixmap(QPixmap(""))

    def display(self, img_path):
        print(img_path)
        pix = QPixmap(img_path)
        self.label.setPixmap(pix)
        self.label.setScaledContents(True)
        QApplication.processEvents()

    def model_detect(self, input_img_path):
        # 输入为待检测图片的路径
        # 输出为检测完成后的图像路径以及检测过程中的参数
        output_img_path, res_str = detect_and_reco_defect(input_img_path[0])
        # parm = 'None'
        # output_img_path = input_img_path[0]
        #draw text
        np_img = cv2.imread(output_img_path)
        y0, dy = 50, 50
        for i, txt in enumerate(res_str.split('\n')):
            y = y0 + i * dy
            cv2.putText(np_img, txt,(50, y), cv2.FONT_HERSHEY_PLAIN, 3.0, (0, 255, 0), 3)
        cv2.imwrite(output_img_path, np_img)
        print(res_str)
        return output_img_path, res_str

    def model_reco(self, input_img_path):
        # 输入为待检测图片的路径
        # 输出为检测完成后的图像路径以及检测过程中的参数
        im = plt.imread(os.path.join(input_img_path[0]))[:,:,:3]
        im_anno, meter, meter_type, meter_rect, reading_pred = detect_rectify_read_an_image(im)
        cv2.putText(im_anno, 'meter_type:{}, result:{}'.format(meter_type, str(round(reading_pred, 3))),
         (50, 50), cv2.FONT_HERSHEY_PLAIN, 3.0, (0, 255, 0), 3)
        h1, w1 = meter_rect.shape[:2]
        meter_rect = meter_rect * 255
        im_anno[50:50+h1, 2200:2200+w1] = meter_rect.astype(np.uint8)
        plt.imsave('./temp_meter.jpg', im_anno.astype(np.uint8))
        output_img_path = './temp_meter.jpg'
        print(meter_type, reading_pred)
        parm = 'None'
        return output_img_path, parm



class WorkThread_meter(QThread):
    trigger = pyqtSignal(str)
    def __init__(self):
        super(WorkThread_meter, self).__init__()

    def run(self, imgs_to_reco):
        for img_path in imgs_to_reco:
            tran_and_pad_img(img_path, './show_tmp.jpg')
            self.trigger.emit('./show_tmp.jpg')
            im = plt.imread(os.path.join(img_path))[:,:,:3]
            im_anno, meter, meter_type, meter_rect, reading_pred = detect_rectify_read_an_image(im)
            cv2.putText(im_anno, 'meter_type:{}, result:{}'.format(meter_type, str(round(reading_pred, 3))),
             (50, 50), cv2.FONT_HERSHEY_PLAIN, 3.0, (0, 255, 0), 3)
            h1, w1 = meter_rect.shape[:2]
            meter_rect = meter_rect * 255
            im_anno[50:50+h1, 2200:2200+w1] = meter_rect
            plt.imsave('./temp_meter.jpg', im_anno.astype(np.uint8))
            output_img_path = './temp_meter.jpg'
            self.trigger.emit(output_img_path)
            time.sleep(2)

class WorkThread_defect(QThread):
    trigger = pyqtSignal(str)
    def __init__(self):
        super(WorkThread_defect, self).__init__()

    def run(self, imgs_to_reco):
        for img_path in imgs_to_reco:
            tran_and_pad_img(img_path, './show_tmp.jpg')
            self.trigger.emit('./show_tmp.jpg')
            output_img_path, res_str = detect_and_reco_defect(img_path)
            #draw text
            np_img = cv2.imread(output_img_path)
            y0, dy = 50, 50
            for i, txt in enumerate(res_str.split('\n')):
                y = y0 + i * dy
                cv2.putText(np_img, txt,(50, y), cv2.FONT_HERSHEY_PLAIN, 3.0, (0, 255, 0), 3)
            cv2.imwrite(output_img_path, np_img)
            self.trigger.emit(output_img_path)
            time.sleep(2)


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)  # 创建一个QApplication，也就是你要开发的软件app
    MainWindow = QtWidgets.QMainWindow()    # 创建一个QMainWindow，用来装载你需要的各种组件、控件
    Demo = DemoWindow(MainWindow)           # ui是Ui_MainWindow()类的实例化对象
    MainWindow.show()                       # 执行QMainWindow的show()方法，显示这个QMainWindow
    sys.exit(app.exec_())                   # 使用exit()或者点击关闭按钮退出QApplication