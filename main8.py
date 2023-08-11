import webbrowser
import GSXGBoost1
import RAXGBoost1
import XGBoost1
import pandas as pd
import 贝叶斯XGBoost1
from PyQt5 import QtWidgets, QtGui
import GUI
import sys
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QSizePolicy, QTableWidgetItem, QFileDialog
from config import content
from matplotlib import pyplot as plt
from PyQt5.QtCore import Qt
from future.moves import pickle

# 按钮功能
class PushButton(GUI.Ui_MainWindow, QtWidgets.QMainWindow):
    def __init__(self):
        super(PushButton, self).__init__()
        self.setupUi(self)
        self.pushButton.clicked.connect(self.loadFile)
        self.pushButton_6.clicked.connect(self.previewData)
        self.pushButton_5.clicked.connect(self.loadPklFile)
        self.pushButton_3.clicked.connect(self.runAlgorithm)
        self.pushButton_2.clicked.connect(self.saveData)
        self.pushButton_4.clicked.connect(self.saveTrainedModel)

        self.df = None
        self.correlation_matrix = None  # Add this attribute

    def saveTrainedModel(self):
        if hasattr(self, 'trained_model'):
            options = QFileDialog.Options()
            options |= QFileDialog.DontUseNativeDialog

            file_path, _ = QFileDialog.getSaveFileName(
                self, "保存模型", "", "Pickle Files (*.pkl)", options=options)
            if file_path:
                if not file_path.lower().endswith('.pkl'):
                    file_path += '.pkl'
                try:
                    with open(file_path, 'wb') as f:
                        pickle.dump(self.trained_model, f)
                    QtWidgets.QMessageBox.information(None, "保存成功", f"模型已保存为：{file_path}")
                except Exception as e:
                    QtWidgets.QMessageBox.warning(None, "保存失败", f"无法保存模型：{str(e)}")
        else:
            QtWidgets.QMessageBox.warning(None, "模型未训练", "请先运行算法并训练模型或导入模型")

    def saveData(self):
        # 检查是否存在 y_pred_array，意味着已经完成预测
        if hasattr(self, 'y_pred_array'):
            options = QFileDialog.Options()
            options |= QFileDialog.DontUseNativeDialog  # 使用 PyQt 对话框

            # 打开文件对话框以选择保存位置和文件名
            excel_filename, _ = QFileDialog.getSaveFileName(
                self, "保存预测数据", "", "Excel 文件 (*.xlsx)", options=options)

            if excel_filename:
                # 如果用户没有添加 .xlsx 后缀名，则自动添加
                if not excel_filename.lower().endswith('.xlsx'):
                    excel_filename += '.xlsx'

                try:
                    df = pd.DataFrame({'Predicted Values': self.y_pred_array})
                    df.to_excel(excel_filename, index=False, engine='xlsxwriter')
                    QtWidgets.QMessageBox.information(None, "保存成功", f"数据已保存为：{excel_filename}")
                except Exception as e:
                    QtWidgets.QMessageBox.warning(None, "保存失败", f"无法保存数据：{str(e)}")
        else:
            QtWidgets.QMessageBox.warning(None, "预测未完成", "请先运行算法以完成预测")

    def calculate_approximate_formula(self):
        input_column_names = self.df.columns[5:11]
        output_column_name = self.df.columns[11]

        inputs = self.df[input_column_names]
        output = self.df[output_column_name]

        from sklearn.linear_model import LinearRegression

        # Create a linear regression model
        model = LinearRegression()
        model.fit(inputs, output)

        # Get the coefficients and intercept of the linear model
        coefficients = model.coef_
        intercept = model.intercept_

        # Create the approximate formula string
        formula = "{} = {:.2f} + ".format(output_column_name, intercept)
        for col_name, coef in zip(input_column_names, coefficients):
            formula += "{:.2f} * {} + ".format(coef, col_name)
        formula = formula[:-3]  # Remove the last "+ "

        return formula

    def calculate_and_display_correlation(self):
        selected_columns = self.df.iloc[:, 5:13]
        correlation_matrix = selected_columns.corr()

        self.correlation_matrix = correlation_matrix
        self.display_correlation_matrix()

    def display_correlation_matrix(self):
        if self.correlation_matrix is not None:
            self.tableWidget.setRowCount(len(self.correlation_matrix))
            self.tableWidget.setColumnCount(len(self.correlation_matrix))

            for row_idx, (col_name, col_data) in enumerate(self.correlation_matrix.iterrows()):
                for col_idx, correlation_value in enumerate(col_data):
                    item = QTableWidgetItem(f"{correlation_value:.2f}")
                    self.tableWidget.setItem(row_idx, col_idx, item)

            self.tableWidget.setHorizontalHeaderLabels(self.correlation_matrix.columns)
            self.tableWidget.setVerticalHeaderLabels(self.correlation_matrix.index)

    def show_scatter_plot(self, y_true_array, y_pred_array):
        plt.figure(figsize=(10, 6))
        plt.scatter(y_true_array, y_pred_array, color='blue', alpha=0.05, label='Predicted vs True')
        plt.plot([min(y_true_array), max(y_true_array)], [min(y_true_array), max(y_true_array)], color='red',
                 linestyle='--', linewidth=2, label='Perfect Prediction Line')
        plt.xlabel('True Values')
        plt.ylabel('Predicted Values')
        plt.title('Predicted vs True Values')
        plt.legend()
        plt.grid(True)
        plt.savefig('散点图.jpg')

        pixmap = QPixmap('散点图.jpg')
        label = QtWidgets.QLabel(self.frame_6)
        label.setPixmap(pixmap)
        label.setScaledContents(True)
        label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        label.setAlignment(Qt.AlignCenter)
        label.setGeometry(0, 0, self.frame_6.width(), self.frame_6.height())
        label.show()

    def show_line_plot(self, y_true_array, y_pred_array):
        plt.figure(figsize=(10, 6))
        plt.plot(y_true_array[:4000], color='black', label='True Values')
        plt.plot(y_pred_array[:4000], color='blue', alpha=0.5, label='Predicted Values')
        plt.xlabel('Data Index')
        plt.ylabel('Values')
        plt.title('True vs Predicted Values')
        plt.legend()
        plt.grid(True)
        plt.savefig('曲线图.jpg')

        pixmap = QPixmap('曲线图.jpg')
        label = QtWidgets.QLabel(self.frame_5)
        label.setPixmap(pixmap)
        label.setScaledContents(True)
        label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        label.setAlignment(Qt.AlignCenter)
        label.setGeometry(0, 0, self.frame_5.width(), self.frame_5.height())
        label.show()

    def loadFile(self):
        try:
            filePath, _ = QtWidgets.QFileDialog.getOpenFileName(self, "选取文件", "./", "Excel Files (*.xlsx)")
            if filePath:
                content = filePath
                self.check_preview.set_content(filePath)
                print("导入的文件:", filePath)
                self.df = pd.read_excel(filePath)
                print("成功导入数据",filePath)
        except Exception as e:
            error_message = f"加载文件时出现错误：{str(e)}"
            QtWidgets.QMessageBox.warning(self, "错误", error_message)

    def previewData(self):
        self.check_preview.open_file()

    def loadPklFile(self):
        filePath, _ = QtWidgets.QFileDialog.getOpenFileName(self, "选取 .pkl 文件", "./", "Pickle Files (*.pkl)")
        if filePath:
            print("加载的 .pkl 文件路径和名称:", filePath)
            with open(filePath, 'rb') as f:
                self.trained_model = pickle.load(f)

    def runAlgorithm(self):
        if self.df is None:
            QtWidgets.QMessageBox.warning(None, "数据未导入", "请先导入数据")
            return

        radioButton_checked = self.radioButton.isChecked()
        radioButton_2_checked = self.radioButton_2.isChecked()
        radioButton_4_checked = self.radioButton_4.isChecked()
        radioButton_9_checked = self.radioButton_9.isChecked()

        radioButton_5_checked = self.radioButton_5.isChecked()
        radioButton_6_checked = self.radioButton_6.isChecked()
        radioButton_7_checked = self.radioButton_7.isChecked()
        radioButton_8_checked = self.radioButton_8.isChecked()

        run_a = radioButton_2_checked and radioButton_8_checked  # XGB
        run_b = radioButton_2_checked and radioButton_6_checked  # XGB-网格搜搜
        run_c = radioButton_2_checked and radioButton_7_checked  # XGB-贝叶斯
        run_d = radioButton_2_checked and radioButton_5_checked  # XGB-随机森林
        run_rest = radioButton_4_checked or radioButton_9_checked or radioButton_checked  # 提示作用

        no_algorithm_selected = not any(
            [radioButton_checked, radioButton_2_checked, radioButton_4_checked, radioButton_9_checked,
             radioButton_5_checked, radioButton_6_checked, radioButton_7_checked, radioButton_8_checked])

        if run_a:
            print("XGBoost运行中")
            try:
                result_dict = XGBoost1.run(self.df)
                r2_xgb_value = result_dict["r2_xgb"]
                mae_xgb_true_value = result_dict["mae_xgb_true"]
                mse_xgb_true_value = result_dict["mse_xgb_true"]
                y_true_array = result_dict["y_true"]
                self.y_pred_array = result_dict["y_pred"]
                data_true_dataframe = result_dict["data_true"]
                self.trained_model = result_dict["model"]

                # 显示指标
                self.textBrowser.clear()
                self.textBrowser.append(f"{r2_xgb_value:.5f}")
                self.textBrowser_3.clear()
                self.textBrowser_3.append(f"{mae_xgb_true_value:.5f}")
                self.textBrowser_2.clear()
                self.textBrowser_2.append(f"{mse_xgb_true_value:.5f}")
                # 画图
                self.show_scatter_plot(y_true_array, self.y_pred_array)
                self.show_line_plot(y_true_array, self.y_pred_array)
                # 计算相关性
                self.calculate_and_display_correlation()
                # 近似公式计算
                formula = self.calculate_approximate_formula()
                self.textBrowser_4.clear()
                self.textBrowser_4.append(formula)
            except Exception as e:
                print("算法XGBoost运行时出现错误:", str(e))
            print("算法XGBoost完成")
        elif run_b:
            print("GS-XGBoost运行中")
            try:
                result_dict = GSXGBoost1.run(self.df)
                r2_xgb_value = result_dict["r2_xgb"]
                mae_xgb_true_value = result_dict["mae_xgb_true"]
                mse_xgb_true_value = result_dict["mse_xgb_true"]
                y_true_array = result_dict["y_true"]
                self.y_pred_array = result_dict["y_pred"]
                data_true_dataframe = result_dict["data_true"]
                self.trained_model = result_dict["model"]

                # 显示指标
                self.textBrowser.clear()
                self.textBrowser.append(f"{r2_xgb_value:.5f}")
                self.textBrowser_3.clear()
                self.textBrowser_3.append(f"{mae_xgb_true_value:.5f}")
                self.textBrowser_2.clear()
                self.textBrowser_2.append(f"{mse_xgb_true_value:.5f}")
                # 画图
                self.show_scatter_plot(y_true_array, self.y_pred_array)
                self.show_line_plot(y_true_array, self.y_pred_array)
                # 计算相关性
                self.calculate_and_display_correlation()
                # 近似公式计算
                formula = self.calculate_approximate_formula()
                self.textBrowser_4.clear()
                self.textBrowser_4.append(formula)
            except Exception as e:
                print("算法GSXGBoost运行时出现错误:", str(e))
            print("GSXGBoost完成")
        elif run_c:
            print("贝叶斯-XGBoost运行中")
            try:
                result_dict = 贝叶斯XGBoost1.run(self.df)
                r2_xgb_value = result_dict["r2_xgb"]
                mae_xgb_true_value = result_dict["mae_xgb_true"]
                mse_xgb_true_value = result_dict["mse_xgb_true"]
                y_true_array = result_dict["y_true"]
                self.y_pred_array = result_dict["y_pred"]
                data_true_dataframe = result_dict["data_true"]
                self.trained_model = result_dict["model"]

                # 显示指标
                self.textBrowser.clear()
                self.textBrowser.append(f"{r2_xgb_value:.5f}")
                self.textBrowser_3.clear()
                self.textBrowser_3.append(f"{mae_xgb_true_value:.5f}")
                self.textBrowser_2.clear()
                self.textBrowser_2.append(f"{mse_xgb_true_value:.5f}")
                # 画图
                self.show_scatter_plot(y_true_array, self.y_pred_array)
                self.show_line_plot(y_true_array, self.y_pred_array)
                # 计算相关性
                self.calculate_and_display_correlation()
                # 近似公式计算
                formula = self.calculate_approximate_formula()
                self.textBrowser_4.clear()
                self.textBrowser_4.append(formula)
            except Exception as e:
                print("贝叶斯XGBoost运行时出现错误:", str(e))
            print("贝叶斯XGBoost完成")
        elif run_d:
            print("RA-XGBoost运行中")
            try:
                result_dict = RAXGBoost1.run(self.df)
                r2_xgb_value = result_dict["r2_xgb"]
                mae_xgb_true_value = result_dict["mae_xgb_true"]
                mse_xgb_true_value = result_dict["mse_xgb_true"]
                y_true_array = result_dict["y_true"]
                self.y_pred_array = result_dict["y_pred"]
                data_true_dataframe = result_dict["data_true"]
                self.trained_model = result_dict["model"]

                # 显示指标
                self.textBrowser.clear()
                self.textBrowser.append(f"{r2_xgb_value:.5f}")
                self.textBrowser_3.clear()
                self.textBrowser_3.append(f"{mae_xgb_true_value:.5f}")
                self.textBrowser_2.clear()
                self.textBrowser_2.append(f"{mse_xgb_true_value:.5f}")
                # 画图
                self.show_scatter_plot(y_true_array, self.y_pred_array)
                self.show_line_plot(y_true_array, self.y_pred_array)
                # 计算相关性
                self.calculate_and_display_correlation()
                # 近似公式计算
                formula = self.calculate_approximate_formula()
                self.textBrowser_4.clear()
                self.textBrowser_4.append(formula)
            except Exception as e:
                print("RAXGBoost运行时出现错误:", str(e))
            print("RAXGBoost完成")
        elif run_rest:
            QtWidgets.QMessageBox.information(None, "提示", "当前预测算法仅提供XGBoost试用，其他算法未放入项目包，可联系开发者或自行根据框架载入算法包（鼓励）")
        elif no_algorithm_selected:
            QtWidgets.QMessageBox.warning(None, "未选择算法", "请选择合适的算法")
        else:
            QtWidgets.QMessageBox.warning(None, "未匹配的组合", "请选择正确的算法组合")

#数据预览功能模块
class CheckAndPreview:
    def __init__(self):
        self.content = content  # 初始化为空
        self.data = ""
    def open_file(self):
        if self.content:
            try:
                webbrowser.open(self.content)  # 使用系统默认程序打开文件
            except Exception as e:
                QtWidgets.QMessageBox.warning(None, "打开文件失败", f"无法打开文件：{str(e)}")
        else:
            QtWidgets.QMessageBox.warning(None, "文件未导入", "请导入数据")

    def set_content(self, content):
        self.content = content

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)

    check_preview = CheckAndPreview()

    mywindow = PushButton()
    mywindow.check_preview = check_preview  # 传递实例以供调用
    mywindow.show()

    sys.exit(app.exec_())
