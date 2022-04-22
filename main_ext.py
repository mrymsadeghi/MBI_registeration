#!/usr/bin/env python

from PyQt5.QtCore import QDir, Qt, QPoint, QSize, pyqtSignal, QEvent
from PyQt5.QtGui import QImage, QPainter, QPalette, QPixmap, QImageReader, QTextCursor, QCursor, QColor, QPen
from PyQt5.QtWidgets import (QAction, QApplication, QFileDialog, QLabel,
                             QMainWindow, QMenu, QMessageBox, QScrollArea, QSizePolicy, QVBoxLayout, QHBoxLayout,
                             QPushButton, QWidget, QTextEdit, QDialog, QLineEdit, QInputDialog)
import cv2 as cv
from functionMapping import *
from funcLandmarkDetection import *

gPath = "."
gFirstFileName = "image1.jpg"
gSecondFileName = "image2.jpg"

class MyLabel(QLabel):
    mouseMove = pyqtSignal(int, int)
    mouseClicked = pyqtSignal(int, int)

    def __init__(self, isClickable, imageFile, landmarksCount):
        super(MyLabel, self).__init__()

        # save parameters
        self.isClickable = isClickable
        self.imageFile = imageFile
        self.landmarksCount = landmarksCount

        # data
        self.points = []
        self.scaleFactor = 1

        # UI
        self.setMouseTracking(True)
        self.setCursor(QCursor(Qt.CrossCursor))
        self.whitePen = QPen(QColor('white'))
        self.mypixmap = None

        # Open image
        self.openImage()

    def openImage(self):
        self.mypixmap = QPixmap(self.imageFile)
        if self.mypixmap.isNull():
            return

        # redraw
        self.update()

    def setPixmap(self, pixmap):
        self.mypixmap = pixmap

    def mouseMoveEvent(self, QMouseEvent):
        super().mouseMoveEvent(QMouseEvent)

        if not self.mypixmap:
            return

        self.scaleFactor = max(self.mypixmap.width() / self.width(),
                               self.mypixmap.height() / self.height())

        self.mouseMove.emit(QMouseEvent.x() * self.scaleFactor, QMouseEvent.y() * self.scaleFactor)

    def mousePressEvent(self, QMouseEvent):
        super().mousePressEvent(QMouseEvent)

        if not self.isClickable:
            return

        if QMouseEvent.button() == Qt.LeftButton:
            pos = QMouseEvent.pos() * self.scaleFactor
            self.mouseClicked.emit(pos.x(), pos.y())

            # Save the point and label
            self.points.append(pos)

            # Draw
            self.update()

    def contextMenuEvent(self, event):
        contextMenu = QMenu(self)
        openAction = contextMenu.addAction("Open...")
        landmarksAction = contextMenu.addAction("Landmarks...")
        action = contextMenu.exec_(self.mapToGlobal(event.pos()))

        if action == openAction:
            options = QFileDialog.Options()
            options |= QFileDialog.DontUseNativeDialog
            fileName, _ = QFileDialog.getOpenFileName(self, "Open Image", "",
                                                      "Image files (*.jpg *.png);;All Files (*)", options=options)
            if fileName:
                self.imageFile = fileName
                self.openImage()

        elif action == landmarksAction:
            count, okPressed = QInputDialog.getInt(self, "Input", "Landmarks Count:", self.landmarksCount, 1, 1000, 1)
            if okPressed:
                coord = funcLandmarkDetection(self.imageFile, count)
                print(coord)

                for pos in coord:
                    pos = QPoint(pos[1], pos[0])
                    self.points.append(pos)
                    self.mouseClicked.emit(pos.x(), pos.y())

                self.update()

    def paintEvent(self, QPaintEvent):
        super().paintEvent(QPaintEvent)

        if not self.mypixmap:
            return

        self.scaleFactor = max(self.mypixmap.width() / self.width(),
                               self.mypixmap.height() / self.height())

        # draw image
        painter = QPainter(self)
        scaledPix = self.mypixmap.scaled(self.size(), Qt.KeepAspectRatio, transformMode=Qt.SmoothTransformation)
        self.setMaximumSize(scaledPix.size())
        painter.drawPixmap(QPoint(0, 0), scaledPix)
        self.setMaximumSize(QSize(4000, 5000))

        # draw points
        painter.setBrush(Qt.red)
        for i in range(len(self.points)):
            point = self.points[i]
            x = point.x() / self.scaleFactor
            y = point.y() / self.scaleFactor

            painter.drawEllipse(x - 4, y - 4, 8, 8)

        # draw point text
        painter.setPen(self.whitePen)

        midWidth = self.width() / 2
        midHeight = self.height() / 2
        for i in range(len(self.points)):
            point = self.points[i]
            x = point.x() / self.scaleFactor
            y = point.y() / self.scaleFactor

            # Biase the x, y
            if x < midWidth:
                x -= 15
            else:
                x += 5

            if y < midHeight:
                y -= 7
            else:
                y += 15

            # Clamp x, y
            if x < 5:
                x = 5
            elif x > (self.width() - 15):
                x = (self.width() - 15)

            if y < 15:
                y = 15
            elif y > (self.height() - 10):
                y = (self.height() - 10)

            painter.drawText(x, y, "%d" % i)


    def clearPoints(self):
        self.points.clear()
        self.update()

class ResultDialog(QDialog):
    def __init__(self, parent=None):
        super(ResultDialog, self).__init__(parent)

        mainLayout = QVBoxLayout()

        # image layout
        layout1 = QHBoxLayout()
        self.imageLabel = MyLabel(False, None, None)
        self.imageLabel.setScaledContents(False)
        self.imageLabel.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)

        mainLayout.addWidget(self.imageLabel)

        # Button
        layout2 = QHBoxLayout()
        self.saveButton = QPushButton("Save")
        self.saveButton.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        layout2.addWidget(self.saveButton)
        mainLayout.addLayout(layout2)

        self.restartButton = QPushButton("Restart")
        self.restartButton.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        layout2.addWidget(self.restartButton)
        mainLayout.addLayout(layout2)

        self.setLayout(mainLayout)
        self.setWindowTitle("Result")

        # resize
        self.resize(600, 500)
        mainLayout.activate()

        # signal connect
        self.saveButton.clicked.connect(self.accept)
        self.restartButton.clicked.connect(self.close)

    def setPixmap(self, pixmap):
        self.imageLabel.setPixmap(pixmap)

class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()

        mainLayout = QVBoxLayout()

        # image layout
        layout1 = QHBoxLayout()
        self.imageLabel1 = MyLabel(True, gFirstFileName, 200)
        self.imageLabel1.setScaledContents(False)
        self.imageLabel1.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)

        self.imageLabel2 = MyLabel(True, gSecondFileName, 400)
        self.imageLabel2.setScaledContents(False)
        self.imageLabel2.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)

        layout1.addWidget(self.imageLabel1)
        layout1.addWidget(self.imageLabel2)
        mainLayout.addLayout(layout1, 3)

        # information label layout
        layout2 = QHBoxLayout()

        self.sizeLablel1 = QLabel()
        layout2.addWidget(self.sizeLablel1)
        self.positionLabel1 = QLabel("(0, 0)")
        layout2.addWidget(self.positionLabel1)

        self.sizeLablel2 = QLabel()
        layout2.addWidget(self.sizeLablel2)
        self.positionLabel2 = QLabel("(0, 0)")
        layout2.addWidget(self.positionLabel2)

        mainLayout.addLayout(layout2)

        # textedit layout
        layout3 = QHBoxLayout()
        self.textEdit1 = QTextEdit()
        self.textEdit1.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)

        self.textEdit2 = QTextEdit()
        self.textEdit2.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)

        layout3.addWidget(self.textEdit1)
        layout3.addWidget(self.textEdit2)
        mainLayout.addLayout(layout3, 1)

        # Button
        layout4 = QHBoxLayout()
        self.doneButton = QPushButton("Done")
        self.doneButton.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        layout4.addWidget(self.doneButton)
        mainLayout.addLayout(layout4)

        # main window
        self.widget = QWidget()
        self.widget.setLayout(mainLayout)

        self.setCentralWidget(self.widget)
        self.setWindowTitle("Two Images")

        # resize
        self.resize(1000, 500)
        mainLayout.activate()

        # data structure
        self.img1coords = []
        self.img2coords = []

        # signal connect
        self.imageLabel1.mouseMove.connect(self.image1MouseMove)
        self.imageLabel2.mouseMove.connect(self.image2MouseMove)
        self.imageLabel1.mouseClicked.connect(self.image1MouseClicked)
        self.imageLabel2.mouseClicked.connect(self.image2MouseClicked)

        self.doneButton.clicked.connect(self.on_doneButton_clicked)

    def resizeEvent(self, *args, **kwargs):
        super().resizeEvent(*args, **kwargs)

        self.imageLabel1.update()
        self.imageLabel2.update()

        if self.imageLabel1.mypixmap:
            self.sizeLablel1.setText("size: (%d, %d) - (%d, %d)" % (self.imageLabel1.mypixmap.width(), self.imageLabel1.mypixmap.height(),
                                                                self.imageLabel1.width(), self.imageLabel1.height()))

        if self.imageLabel2.mypixmap:
            self.sizeLablel2.setText("size: (%d, %d) - (%d, %d)" % (self.imageLabel2.mypixmap.width(), self.imageLabel2.mypixmap.height(),
                                                                self.imageLabel2.width(), self.imageLabel2.height()))


    def image1MouseMove(self, x, y):
        self.positionLabel1.setText("Position: (%d, %d)" % (x, y))

    def image2MouseMove(self, x, y):
        self.positionLabel2.setText("Position: (%d, %d)" % (x, y))


    def image1MouseClicked(self, x, y):
        self.textEdit1.append("(%d, %d)" % (x, y))
        self.textEdit1.moveCursor(QTextCursor.End)

        self.img1coords.append((x, y))

    def image2MouseClicked(self, x, y):
        self.textEdit2.append("(%d, %d)" % (x, y))
        self.textEdit2.moveCursor(QTextCursor.End)

        self.img2coords.append((x, y))

    def on_doneButton_clicked(self):
        resultDialog = ResultDialog(self)
        resultDialog.setWindowFlag(Qt.WindowMaximizeButtonHint)

        mergedPixmap = self.mergeImage()

        resultDialog.setPixmap(mergedPixmap)

        if resultDialog.exec()==QDialog.Accepted:
            self.save()
        else:
            self.restart()

    def mergeImage(self):
        # test
        # self.img1coords = [(20, 646), (0, 725), (22, 804), (38, 883), (69, 962), (124, 1041), (185, 1098), (247, 1124), (349, 1107), (449, 1067), (481, 1008), (600, 949), (645, 890), (648, 831), (710, 772), (726, 713), (704, 654), (597, 590), (724, 531), (732, 472), (725, 413), (710, 354), (673, 295), (642, 236), (587, 177), (568, 118), (455, 7), (342, 0), (255, 64), (168, 81), (115, 162), (75, 243), (62, 324), (40, 405), (66, 486), (190, 567), (271, 571), (352, 575), (433, 579), (514, 583)]
        # self.img2coords = [(184, 775), (176, 835), (191, 895), (216, 955), (249, 1015), (307, 1075), (375, 1119), (443, 1136), (519, 1122), (596, 1089), (684, 1042), (724, 995), (751, 948), (774, 901), (800, 854), (819, 807), (812, 760), (760, 710), (813, 663), (818, 616), (800, 569), (774, 522), (751, 475), (723, 428), (682, 381), (593, 334), (510, 301), (427, 290), (367, 311), (307, 350), (249, 410), (215, 470), (190, 530), (175, 590), (184, 650), (253, 715), (354, 714), (455, 713), (556, 712), (657, 711)]

        # merge
        self.resultImage = funcMapping(gPath, self.imageLabel1.imageFile, self.imageLabel2.imageFile,
                                  self.img1coords, self.img2coords)

        # display
        mergedImage = QImage(self.resultImage.data, self.resultImage.shape[1], self.resultImage.shape[0],
                                  QImage.Format_RGB888).rgbSwapped()
        mergedPixmap = QPixmap.fromImage(mergedImage)

        return mergedPixmap

    def save(self):
        # Save fil dialog
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getSaveFileName(self, "Save Image", "",
                                                  "Image files (*.jpg *.png);;All Files (*)", options=options)
        if not fileName:
            alert = QMessageBox()
            alert.setText('The result image is not saved.')
            alert.setWindowTitle('Save')
            alert.exec_()
            return

        # Save Image
        cv.imwrite(fileName, self.resultImage)

        alert = QMessageBox()
        alert.setText('The result image is saved successfully!')
        alert.setWindowTitle('Save')
        alert.exec_()

    def restart(self):
        alert = QMessageBox()
        alert.setText('Restarting now...')
        alert.setWindowTitle('Restart')
        alert.exec_()

        self.textEdit1.setText("")
        self.textEdit2.setText("")

        self.imageLabel1.clearPoints()
        self.imageLabel2.clearPoints()

        self.img1coords.clear()
        self.img2coords.clear()

if __name__ == '__main__':

    import sys

    app = QApplication(sys.argv)
    mainWindow = MainWindow()
    mainWindow.show()
    sys.exit(app.exec_())
