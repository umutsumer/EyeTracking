import sys
#import main as m
#import uitest as m
import cemalui as m
import cv2
from PyQt5.QtWidgets import QApplication, QDialog


class EyeTracking(QDialog):
    def __init__(self):
        super(EyeTracking, self).__init__()

if __name__ == "__main__":
    App = QApplication(sys.argv)
    Root = m.MainWindow()
    Root.show()
    
    sys.exit(App.exec())


