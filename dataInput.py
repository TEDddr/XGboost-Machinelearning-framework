from PyQt5 import QtWidgets, QtGui
import GUI
import sys

class DataInputButton(GUI.Ui_MainWindow, QtWidgets.QMainWindow):
    def __init__(self):
        super(DataInputButton, self).__init__()
        self.setupUi(self)
        self.pushButton.clicked.connect(self.loadFile)
        self.pushButton_6.clicked.connect(self.previewData)

    def loadFile(self):
        filePath, _ = QtWidgets.QFileDialog.getOpenFileName(self, "选取文件", "./", "All Files (*)")
        if filePath:
            self.check_preview.set_content(filePath)
            print("导入的文件:", filePath)

    def previewData(self):
        self.check_preview.preview()



class CheckAndPreview:
    def __init__(self):
        self.content = ""  # 初始化为空

    def check(self):
        if self.content:
            with open(self.content, 'r') as file:
                content = file.read()
            print("预览数据如窗口所示:")
            print(content)
            preview_window = PreviewWindow(content)
            preview_window.show()
        else:
            QtWidgets.QMessageBox.warning(None, "文件未导入", "请导入数据")

    def preview(self):
        if self.content:
            with open(self.content, 'r') as file:
                content = file.read()
            preview_window = PreviewWindow(content)
            preview_window.exec_()
        else:
            QtWidgets.QMessageBox.warning(None, "文件未导入", "请导入数据")

    def set_content(self, content):
        self.content = content

class PreviewWindow(QtWidgets.QDialog):
    def __init__(self, content):
        super().__init__()
        self.setWindowTitle("文件预览")
        self.setGeometry(100, 100, 800, 600)

        self.text_edit = QtWidgets.QTextEdit(self)
        self.text_edit.setPlainText(content)

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.text_edit)

        self.setLayout(layout)

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)

    check_preview = CheckAndPreview()

    mywindow = DataInputButton()
    mywindow.check_preview = check_preview  # 传递实例以供调用
    mywindow.show()

    sys.exit(app.exec_())
