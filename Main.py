import gui as gui
from PyQt5.QtWidgets import QWidget, QPushButton, QFileDialog, QDialog
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap, QImage
from Classification import NaiveBayes, KNN

class Main(QWidget):
    __csvUrl = ""
    __posUrl = ""
    __negUrl = ""
    __trainUrl = ""
    
    def __init__(self):
        super().__init__()
        self.ui = gui.Ui_Form()
        self.ui.setupUi(self)
        
        self.ui.csv_button.toggled.connect(self.radioCsvSelected)
        self.ui.csv_edit.setDisabled(True)
        self.ui.csv_browse.setDisabled(True)
        self.ui.csv_browse.clicked.connect(self.csvBrowserClicked)
        
        self.ui.str_button.toggled.connect(self.radioStrSelected)
        self.ui.str_edit.setDisabled(True)
        
        self.ui.pos_browse.clicked.connect(self.posClicked)
        self.ui.neg_browse.clicked.connect(self.negClicked)
        self.ui.train_browse.clicked.connect(self.trainClicked)
        
        self.ui.pushButton.clicked.connect(self.pushClicked)
        self.ui.submitButton.clicked.connect(self.submitClicked)
        self.show()
        
    def radioCsvSelected(self, enable):
      if enable:
         self.ui.csv_edit.setDisabled(False)
         self.ui.csv_browse.setDisabled(False)
         self.ui.str_edit.setDisabled(True)
         
    def csvBrowserClicked(self):
      url = str(QFileDialog.getOpenFileName(self, "Select File")[0])
      if url != "":
         self.__csvUrl = url
         self.ui.csv_edit.setText(self.__csvUrl)
      print(self.__csvUrl)
      
    def radioTxtSelected(self, enable):
      if enable:
         self.ui.csv_edit.setDisabled(True)
         self.ui.csv_browse.setDisabled(True)
         self.ui.str_edit.setDisabled(True)
         
    def txtBrowserClicked(self):
      url = str(QFileDialog.getOpenFileName(self, "Select File")[0])
      if url != "":
         self.__txtUrl = url
         self.ui.selectCSV_edit.setText(self.__txtUrl)
      print(self.__txtUrl)
      
    def radioStrSelected(self, enable):
      if enable:
         self.ui.csv_edit.setDisabled(True)
         self.ui.csv_browse.setDisabled(True)
         self.ui.str_edit.setDisabled(False)
         
    def posClicked(self):
        url = str(QFileDialog.getOpenFileName(self, "Select File")[0])
        if url != "":
            self.__posUrl = url
            self.ui.pos_edit.setText(self.__posUrl)
        print(self.__posUrl)
    
    def negClicked(self):
        url = str(QFileDialog.getOpenFileName(self, "Select File")[0])
        if url != "":
            self.__negUrl = url
            self.ui.neg_edit.setText(self.__negUrl)
        print(self.__negUrl)
    
    def trainClicked(self):
        url = str(QFileDialog.getOpenFileName(self, "Select File")[0])
        if url != "":
            self.__trainUrl = url
            self.ui.train_edit.setText(self.__trainUrl)
        print(self.__trainUrl)
         
    def pushClicked(self):
        self.learningSelection = str(self.ui.metode_combo.currentText())
        self.k = int(self.ui.k_combo.currentText())
        self.__posUrl = str(self.ui.pos_edit.text())
        self.__negUrl = str(self.ui.neg_edit.text())
        self.__trainUrl = str(self.ui.train_edit.text())
        if (self.learningSelection == "Naive Bayes"):
            nb = NaiveBayes.NaiveBayes(self.__negUrl, self.__posUrl, self.__trainUrl, "")
            accuracy = nb.classifyAll(self.__trainUrl)
        else:
            knn = KNN.KNN(self.__negUrl, self.__posUrl, self.__trainUrl, "", self.k)
            accuracy = knn.classifyAll(self.__trainUrl, self.k)
            print(accuracy)
        self.ui.akurasi.setText("%.2f" % (accuracy * 100) + "%")
        
    def submitClicked(self):
        self.sentence = self.ui.str_edit.toPlainText()
        self.learningSelection = str(self.ui.metode_combo.currentText())
        self.k = int(self.ui.k_combo.currentText())
        self.__posUrl = str(self.ui.pos_edit.text())
        self.__negUrl = str(self.ui.neg_edit.text())
        self.__trainUrl = str(self.ui.csv_edit.text())
        if (self.ui.csv_button.isChecked()):
            if (self.learningSelection == "Naive Bayes"):
                nb = NaiveBayes.NaiveBayes(self.__negUrl, self.__posUrl, self.__trainUrl, "")
                result = nb.classifyTest(self.__trainUrl)
                self.ui.positif.setText("%.2f" % result[0] + "%\nPOSITIVE")
                self.ui.negatif.setText("%.2f" % result[1] + "%\nNEGATIVE")
                self.ui.netral.setText("%.2f" % result[2] + "%\nNETRAL")
            else:
                knn = KNN.KNN(self.__negUrl, self.__posUrl, self.__trainUrl, "", self.k)
                result = knn.classifyTest(self.__trainUrl, self.k)
                self.ui.positif.setText("%.2f" % result[0] + "%\nPOSITIVE")
                self.ui.negatif.setText("%.2f" % result[1] + "%\nNEGATIVE")
                self.ui.netral.setText("%.2f" % result[2] + "%\nNETRAL")
        elif (self.ui.str_button.isChecked()):
            if (self.learningSelection == "Naive Bayes"):
                nb = NaiveBayes.NaiveBayes(self.__negUrl, self.__posUrl, "", self.sentence)
                kelas = str(nb.classify(self.sentence))
            else:
                knn = KNN.KNN(self.__negUrl, self.__posUrl, "", self.sentence, self.k)
                kelas = str(knn.classify(self.sentence, self.k))
            self.ui.kelas.setText("Kelas: " + kelas)

mainWindow = Main()