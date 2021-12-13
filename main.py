# -*- coding: utf-8 -*-
"""
Created on Tue May  7 10:27:46 2019

@author: lochu lin
"""

from MainWidget import MainWidget
from PyQt5.QtWidgets import QApplication
import sys

def main():
    app = QApplication(sys.argv)
    
    mainWidget = MainWidget()
    mainWidget.show()
    
    exit(app.exec_())
    
    
if __name__ == '__main__':
    main()
