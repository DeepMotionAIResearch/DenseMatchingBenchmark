# -*- coding: utf-8 -*-
import sys
import os
import matplotlib
matplotlib.use('TkAgg')
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog
from PyQt5 import QtWidgets, QtGui, QtCore
from PyQt5.QtWidgets import QFileDialog
from PyQt5.QtCore import Qt, QPoint
from PyQt5.QtGui import QPainter, QColor, QFont

from UI import Ui_CostView as Ui_MainWindow
from dmb.visualization.stereo.vis import disp_err_to_color

from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import pickle

def softmax(x, dim=0):
    mx = np.max(x, axis=dim)
    e_x = np.exp(x - mx)
    return e_x / e_x.sum(axis=dim)

class windows(QtWidgets.QWidget, Ui_MainWindow):
    def __init__(self):
        super(windows, self).__init__()
        self.setupUi(self)
        self._translate = QtCore.QCoreApplication.translate
        self.height = 0
        self.width = 0
        self.pad_size = [600, 1500]
        self.result_path = None
        self.diag_path = '../'
        self.path = dict()

    def setCentralWidget(self, widget):
        pass

    def openDiag(self):
        FileName, FileType = QFileDialog.getOpenFileName(self,
                                                         "OpenFile",
                                                         self.diag_path,
                                                         " All Files (*);;*.pkl;;")
        sender_info = self.sender().text()
        print('{}: {}'.format(sender_info, FileName))
        self.path[sender_info] = FileName
        self.diag_path = os.path.join(FileName, '../')

        self.process()

    def openDiagDirectory(self):
        directory = QFileDialog.getExistingDirectory(self,
                                                     "OpenDirectory",
                                                     self.diag_path)
        sender_info = self.sender().text()
        print('{}: {}'.format(sender_info, directory))
        self.path[sender_info] = directory
        self.diag_path = directory

    def setXY_andRun(self):
        h = self.x_num.value()
        w = self.y_num.value()
        if self.isValid(h, w):
            self.showInfo.setText(self._translate("CostView", "It\'s a valid position:({}, {})".format(h, w)))
            self.plotCost(h, w)

        else:
            self.showInfo.setText(self._translate("CostView", "It\'s not a valid position, press again"))

    def process(self):
        print("Start Loading ... ")
        with open(file=self.path['LoadResult'], mode='rb') as fp:
            result = pickle.load(fp)
        ori_data = result['OriginalData']
        net_result = result['Result']
        self.leftImage = ori_data['leftImage']
        self.rightImage = ori_data['rightImage']
        self.gtDisp = ori_data['leftDisp']
        # [H, W]
        self.estDisp = net_result['disps'][0][0, 0,].cpu().numpy()
        # [D, H, W]
        self.costVolume = net_result['costs'][0][0].cpu().numpy()
        print("Loaded!")

        print("Start Computing Error Map ...")
        err_map = np.abs(self.gtDisp - self.estDisp)
        img = disp_err_to_color(self.estDisp, self.gtDisp)

        self.height = img.shape[0]
        self.width = img.shape[1]

        # pad to pad_size
        down_pad = self.pad_size[0] - img.shape[0]
        right_pad = self.pad_size[1] - img.shape[1]
        img = np.lib.pad(img, ((0, down_pad), (0, right_pad), (0, 0)), mode='constant', constant_values=255)

        pad_path = self.path['LoadResult'].replace('.pkl', '_pad_error.png')
        print("Error Map is saved to ", pad_path)
        Image.fromarray(img).save(pad_path)
        self.show_image(self.show_error_map, pad_path)
        print("Data Prepared, Please Click on Error Map to See Cost Distribution!")

    def show_image(self, label, path):
        if os.path.exists(path):
            png = QtGui.QPixmap(path).scaled(label.width(), label.height())
            label.setPixmap(png)
        else:
            label.setText(self._translate("CostView", "Disparity Error Map Display Window"))

    def isValid(self, h, w):
        if h < 0 or h > (self.height):
            return False
        if w < 0 or w > (self.width):
            return False
        return True

    def mousePressEvent(self, e):
        w = e.x()
        h = e.y()
        w0 = self.show_error_map.pos().x()
        h0 = self.show_error_map.pos().y()
        w = w - w0
        h = h - h0
        if self.isValid(h, w):
            self.showInfo.setText(self._translate("CostView", "It\'s a valid position:({}, {})".format(h, w)))
            self.plotCost(h, w)

        else:
            self.showInfo.setText(self._translate("CostView", "It\'s not a valid position, press again"))

    def plotCost(self, h, w):
        # [height, width]
        estDisp = self.estDisp

        # [height, width]
        gtDisp = self.gtDisp

        # [height, width, 3]
        leftImage = self.leftImage

        # [height, width, 3]
        rightImage = self.rightImage

        # [D, height, width]
        cost = self.costVolume

        # [192, height, width]
        probability = softmax(cost, dim=0)

        # calculate error
        gt = gtDisp[h, w]
        est = estDisp[h, w]
        print('h:{}, w:{}, gt:{:.4f} pred:{:.4f} abs error:{:.4f}'.format(h, w, gt, est, abs(gt - est)))

        x = np.arange(probability.shape[0])
        num_bin = 100
        # probability
        plt.figure(figsize=(80, 60))
        plt.subplot(2, 2, 1)
        gt_y = np.linspace(0, probability[:, h, w].max(), num_bin)
        gt_x = [gt] * num_bin
        est_y = np.linspace(0, probability[:, h, w].max(), num_bin)
        est_x = [est] * num_bin
        plt.plot(gt_x, gt_y, 'r-')
        plt.plot(x, probability[:, h, w], 'b')
        plt.plot(est_x, est_y, 'y-.')
        plt.ylabel('Probability')
        plt.title('Probability    Max:{:.2f}, Min:{:.2f} \n Red: GroundTruth, Yellow: Estimation'.format(probability[:, h, w].max(), probability[:, h, w].min()))

        # cost
        plt.subplot(2, 2, 3)
        plt.plot(x, cost[:, h, w], 'b')
        gt_y = np.linspace(cost[:, h, w].min(), cost[:, h, w].max(), num_bin)
        gt_x = [gt] * num_bin
        est_y = np.linspace(cost[:, h, w].min(), cost[:, h, w].max(), num_bin)
        est_x = [est] * num_bin
        plt.plot(gt_x, gt_y, 'r-')
        plt.plot(est_x, est_y, 'y-.')
        plt.ylabel('Cost')
        plt.title('Cost    Max:{:.2f}, Min:{:.2f}'.format(cost[:, h, w].max(), cost[:, h, w].min()))

        plt.subplot(2, 2, 2)
        plt.imshow(leftImage/255.0)
        plt.plot(w, h, 'w+')
        plt.title('Left Image, Position: ({}, {}) \n White: GroundTruth, Yellow: Estimation'.format(h, w ))
        plt.subplot(2, 2, 4)
        plt.imshow(rightImage/255.0)
        plt.title('Right Image, GT:{:.4f} Pred:{:.4f} Abs Error:{:.4f}'.format(gt, est, abs(gt - est)))
        if w-gt > 0:
            plt.plot(int(w - gt), h, 'w+')
            plt.plot(int(w - est), h, 'y+')
        else:
            plt.plot(int(w - gt), h, 'r+')
            plt.plot(int(w - est), h, 'y+')
        plt.show()


if __name__ == '__main__':
    # set UI
    app = QApplication(sys.argv)
    MainWindow = windows()
    MainWindow.show()
    sys.exit(app.exec_())
