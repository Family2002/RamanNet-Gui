import sys
import numpy as np
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                             QPushButton, QFileDialog, QLabel, QLineEdit, QTabWidget, 
                             QSplitter, QTextEdit, QProgressBar,QMessageBox, QFrame)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QPalette, QColor, QFont
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from sklearn.metrics import confusion_matrix
import seaborn as sns

from codes.data_processing import segment_spectrum_batch
from codes.train_model import train_model
from codes.RamanNet_model import RamanNet
from keras.models import load_model

# 设置全局字体
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("RamanNet 训练与预测")
        self.setGeometry(100, 100, 1200, 800)
        self.setStyleSheet("""
            QMainWindow {
                background-color: #1e1e1e;
            }
            QTabWidget::pane {
                border: none;
                background-color: #252526;
            }
            QTabBar::tab {
                background-color: #2d2d30;
                color: #d4d4d4;
                padding: 10px 25px;
                border-top-left-radius: 5px;
                border-top-right-radius: 5px;
                margin-right: 2px;
            }
            QTabBar::tab:selected {
                background-color: #3c3c3c;
                color: #ffffff;
            }
            QPushButton {
                background-color: #0e639c;
                color: #ffffff;
                border: none;
                padding: 8px 20px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #1177bb;
            }
            QPushButton:pressed {
                background-color: #0d5689;
            }
            QLineEdit {
                background-color: #3c3c3c;
                color: #d4d4d4;
                border: 1px solid #555555;
                padding: 5px;
                border-radius: 4px;
            }
            QLabel {
                color: #d4d4d4;
                font-size: 14px;
            }
            QTextEdit {
                background-color: #1e1e1e;
                color: #d4d4d4;
                border: 1px solid #3c3c3c;
                border-radius: 4px;
            }
            QProgressBar {
                border: none;
                background-color: #3c3c3c;
                color: #ffffff;
                text-align: center;
                height: 10px;
            }
            QProgressBar::chunk {
                background-color: #0e639c;
                border-radius: 5px;
            }
            QSplitter::handle {
                background-color: #2d2d30;
            }
            QFrame {
                background-color: #252526;
                border-radius: 8px;
            }
        """)

        # 创建标签页
        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)

        # 创建训练和预测标签页
        self.train_tab = TrainTab()
        self.predict_tab = PredictTab()

        self.tabs.addTab(self.train_tab, "训练")
        self.tabs.addTab(self.predict_tab, "预测")

class TrainTab(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout()
        layout.setSpacing(15)

        # 创建可拉伸的分割器
        splitter = QSplitter(Qt.Vertical)

        # 数据载入和参数调整模块
        data_params_widget = QFrame()
        data_params_layout = QVBoxLayout(data_params_widget)
        data_params_layout.setContentsMargins(20, 20, 20, 20)
        
        # 添加数据路径框和加载按钮
        data_path_layout = QHBoxLayout()
        self.data_path_input = QLineEdit()
        self.data_path_input.setPlaceholderText("数据文件路径")
        data_path_layout.addWidget(self.data_path_input)
        self.load_data_btn = QPushButton("加载数据")
        self.load_data_btn.clicked.connect(self.load_data)
        data_path_layout.addWidget(self.load_data_btn)
        data_params_layout.addLayout(data_path_layout)

        # 添加模型保存路径框
        self.model_path_input = QLineEdit()
        self.model_path_input.setPlaceholderText("模型保存路径")
        self.model_path_input.setText("trained_model.h5")  # 默认值
        data_params_layout.addWidget(self.model_path_input)

        # 其他参数输入框
        params_layout = QHBoxLayout()
        params_layout.addWidget(QLabel("窗口长度:"))
        self.w_len_input = QLineEdit("50")
        params_layout.addWidget(self.w_len_input)
        params_layout.addWidget(QLabel("步长:"))
        self.dw_input = QLineEdit("25")
        params_layout.addWidget(self.dw_input)
        params_layout.addWidget(QLabel("训练轮数:"))
        self.epochs_input = QLineEdit("100")
        params_layout.addWidget(self.epochs_input)
        params_layout.addWidget(QLabel("验证集比例:"))
        self.val_split_input = QLineEdit("0.2")
        params_layout.addWidget(self.val_split_input)
        data_params_layout.addLayout(params_layout)

        # 添加数据可视化图表
        self.data_fig = Figure(figsize=(5, 4), dpi=100)
        self.data_canvas = FigureCanvas(self.data_fig)
        data_params_layout.addWidget(self.data_canvas)

        data_params_widget.setLayout(data_params_layout)
        splitter.addWidget(data_params_widget)

        # 训练结果模块
        train_results_widget = QWidget()
        train_results_layout = QVBoxLayout()
        self.train_btn = QPushButton("开始训练")
        self.train_btn.clicked.connect(self.start_training)
        train_results_layout.addWidget(self.train_btn)

        # 添加进度条，但初始时隐藏它
        self.progress_bar = QProgressBar()
        self.progress_bar.hide()  # 初始时隐藏进度条
        train_results_layout.addWidget(self.progress_bar)

        self.results_fig = Figure(figsize=(5, 4), dpi=100)
        self.results_canvas = FigureCanvas(self.results_fig)
        train_results_layout.addWidget(self.results_canvas)

        train_results_widget.setLayout(train_results_layout)
        splitter.addWidget(train_results_widget)

        # 数据评估模块
        eval_widget = QWidget()
        eval_layout = QVBoxLayout()
        self.eval_text = QTextEdit()
        eval_layout.addWidget(self.eval_text)
        eval_widget.setLayout(eval_layout)
        splitter.addWidget(eval_widget)

        layout.addWidget(splitter)
        self.setLayout(layout)

    def load_data(self):
        filename, _ = QFileDialog.getOpenFileName(self, "选择数据文件", "", "CSV Files (*.csv)")
        if filename:
            self.data_path_input.setText(filename)
            data = np.loadtxt(filename, delimiter=',')
            self.y = data[:, 0].astype(int)
            self.X = data[:, 1:]
            self.update_data_plot()

    def update_data_plot(self):
        self.data_fig.clear()
        ax = self.data_fig.add_subplot(111)
        
        # 对类别标签进行四舍五入，以解决浮点数精度问题
        rounded_y = np.round(self.y).astype(int)
        unique_classes = np.unique(rounded_y)
        
        colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_classes)))
        
        for i, class_label in enumerate(unique_classes):
            class_data = self.X[rounded_y == class_label]
            mean_spectrum = np.mean(class_data, axis=0)
            ax.plot(mean_spectrum, color=colors[i], label=f'类别 {class_label}', alpha=0.8)
        
        ax.set_title(f"加载的数据 (共 {len(unique_classes)} 个类别, {len(self.y)} 条数据)", fontsize=12)
        ax.set_xlabel("波长", fontsize=10)
        ax.set_ylabel("强度", fontsize=10)
        ax.legend(fontsize=8, loc='best')
        self.data_canvas.draw()

        # 打印一些数据统计信息
        print(f"数据形状: {self.X.shape}")
        print(f"类别数量: {len(unique_classes)}")
        print(f"每个类别的数据量:")
        for class_label in unique_classes:
            print(f"  类别 {class_label}: {np.sum(rounded_y == class_label)}")

    def start_training(self):
        if not hasattr(self, 'X') or not hasattr(self, 'y'):
            QMessageBox.warning(self, "警告", "请先加载数据")
            return

        w_len = int(self.w_len_input.text())
        dw = int(self.dw_input.text())
        epochs = int(self.epochs_input.text())
        val_split = float(self.val_split_input.text())
        model_path = self.model_path_input.text()

        self.progress_bar.setValue(0)  # 重置进度条
        self.progress_bar.show()  # 显示进度条

        self.training_thread = TrainingThread(self.X, self.y, w_len, dw, epochs, val_split, model_path)
        self.training_thread.progress_signal.connect(self.update_progress_bar)
        self.training_thread.finished_signal.connect(self.training_finished)
        self.training_thread.start()

        self.train_btn.setEnabled(False)

    def update_progress_bar(self, value):
        self.progress_bar.setValue(value)

    def training_finished(self, mdl, history):
        self.update_results_plot(history)
        self.update_eval_text(history)
        self.train_btn.setEnabled(True)
        self.progress_bar.hide()  # 训练完成后隐藏进度条

    def update_results_plot(self, history):
        # 更新损失图
        self.results_fig.clear()
        ax1 = self.results_fig.add_subplot(111)
        ax1.plot(history.history['loss'], 'b', label='训练损失')
        ax1.plot(history.history['val_loss'], 'r', label='验证损失')
        ax1.set_title('损失', fontsize=12)
        ax1.set_xlabel('轮次', fontsize=10)
        ax1.set_ylabel('损失值', fontsize=10)
        ax1.legend(fontsize=8)
        self.results_fig.tight_layout()
        self.results_canvas.draw()

    def update_eval_text(self, history):
        final_train_loss = history.history['loss'][-1]
        final_val_loss = history.history['val_loss'][-1]
        final_train_acc = history.history['classification_categorical_accuracy'][-1]
        final_val_acc = history.history['val_classification_categorical_accuracy'][-1]

        eval_text = f"训练结果评估：\n"
        eval_text += f"最终训练损失：{final_train_loss:.4f}\n"
        eval_text += f"最终验证损失：{final_val_loss:.4f}\n"
        eval_text += f"最终训练准确率：{final_train_acc:.4f}\n"
        eval_text += f"最终验证准确率：{final_val_acc:.4f}\n"

        self.eval_text.setText(eval_text)

class PredictTab(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout()

        # 模型加载部分
        model_layout = QHBoxLayout()
        self.model_path_input = QLineEdit()
        self.model_path_input.setPlaceholderText("模型文件路径")
        model_layout.addWidget(self.model_path_input)
        self.load_model_btn = QPushButton("加载模型")
        self.load_model_btn.clicked.connect(self.load_model)
        model_layout.addWidget(self.load_model_btn)
        layout.addLayout(model_layout)

        # 预测数据加载部分
        data_layout = QHBoxLayout()
        self.data_path_input = QLineEdit()
        self.data_path_input.setPlaceholderText("预测数据文件路径")
        data_layout.addWidget(self.data_path_input)
        self.load_data_btn = QPushButton("加载预测数据")
        self.load_data_btn.clicked.connect(self.load_predict_data)
        data_layout.addWidget(self.load_data_btn)
        layout.addLayout(data_layout)

        # 参数调整部分
        params_layout = QHBoxLayout()
        params_layout.addWidget(QLabel("窗口长度:"))
        self.w_len_input = QLineEdit("50")
        params_layout.addWidget(self.w_len_input)
        params_layout.addWidget(QLabel("步长:"))
        self.dw_input = QLineEdit("25")
        params_layout.addWidget(self.dw_input)
        layout.addLayout(params_layout)

        # 预测按钮
        self.predict_btn = QPushButton("开始预测")
        self.predict_btn.clicked.connect(self.start_prediction)
        layout.addWidget(self.predict_btn)

        # 预测结果显示
        self.predict_fig = Figure(figsize=(5, 4), dpi=100)
        self.predict_canvas = FigureCanvas(self.predict_fig)
        layout.addWidget(self.predict_canvas)

        self.predict_result_text = QTextEdit()
        layout.addWidget(self.predict_result_text)

        # 混淆矩阵显示
        self.confusion_fig = Figure(figsize=(5, 4), dpi=100)
        self.confusion_canvas = FigureCanvas(self.confusion_fig)
        layout.addWidget(self.confusion_canvas)

        self.setLayout(layout)

        # 初始化变量
        self.model = None
        self.predict_data = None

    def load_model(self):
        filename, _ = QFileDialog.getOpenFileName(self, "选择模型文件", "", "H5 Files (*.h5)")
        if filename:
            self.model_path_input.setText(filename)
            try:
                self.model = load_model(filename)
                QMessageBox.information(self, "成功", "模型加载成功")
            except Exception as e:
                QMessageBox.warning(self, "错误", f"模型加载失败: {str(e)}")

    def load_predict_data(self):
        filename, _ = QFileDialog.getOpenFileName(self, "选择预测数据文件", "", "CSV Files (*.csv)")
        if filename:
            self.data_path_input.setText(filename)
            try:
                self.predict_data = np.loadtxt(filename, delimiter=',')
                QMessageBox.information(self, "成功", "预测数据加载成功")
            except Exception as e:
                QMessageBox.warning(self, "错误", f"预测数据加载失败: {str(e)}")

    def start_prediction(self):
        if self.model is None:
            QMessageBox.warning(self, "警告", "请先加载模型")
            return
        if self.predict_data is None:
            QMessageBox.warning(self, "警告", "请先加载预测数据")
            return

        try:
            # 获取用户输入的参数
            w_len = int(self.w_len_input.text())
            dw = int(self.dw_input.text())

            # 假设预测数据的格式与训练数据相同，第一列为真实标签
            X_predict = self.predict_data[:, 1:]
            y_true = self.predict_data[:, 0].astype(int)

            # 数据预处理（使用用户输入的参数）
            X_predict_segmented = segment_spectrum_batch(X_predict, w_len, dw)

            # 进行预测
            predictions = self.model.predict(X_predict_segmented)
            
            # 获取分类结果（假设模型输出包含嵌入和分类两部分）
            class_predictions = np.argmax(predictions[1], axis=1)

            # 更新预测结果图表
            self.update_predict_plot(X_predict, class_predictions)

            # 更新预测结果文本
            self.update_predict_text(class_predictions)

            # 生成并显示混淆矩阵
            self.show_confusion_matrix(y_true, class_predictions)

        except Exception as e:
            QMessageBox.warning(self, "错误", f"预测过程中出错: {str(e)}")

    def update_predict_plot(self, X_predict, class_predictions):
        self.predict_fig.clear()
        ax = self.predict_fig.add_subplot(111)
        
        unique_classes = np.unique(class_predictions)
        colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_classes)))
        
        for i, class_label in enumerate(unique_classes):
            class_data = X_predict[class_predictions == class_label]
            mean_spectrum = np.mean(class_data, axis=0)
            ax.plot(mean_spectrum, color=colors[i], label=f'类别 {class_label}', alpha=0.8)
        
        ax.set_title("预测结果", fontsize=12)
        ax.set_xlabel("波长", fontsize=10)
        ax.set_ylabel("强度", fontsize=10)
        ax.legend(fontsize=8, loc='best')
        self.predict_canvas.draw()

    def update_predict_text(self, class_predictions):
        unique, counts = np.unique(class_predictions, return_counts=True)
        result_text = "预测结果统计：\n"
        for class_label, count in zip(unique, counts):
            result_text += f"类别 {class_label}: {count} 个样本\n"
        self.predict_result_text.setText(result_text)

    def show_confusion_matrix(self, y_true, y_pred):
        # 计算混淆矩阵
        cm = confusion_matrix(y_true, y_pred)
        
        # 绘制混淆矩阵
        self.confusion_fig.clear()
        ax = self.confusion_fig.add_subplot(111)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_title('混淆矩阵')
        ax.set_xlabel('预测标签')
        ax.set_ylabel('真实标签')
        self.confusion_canvas.draw()

class TrainingThread(QThread):
    progress_signal = pyqtSignal(int)
    finished_signal = pyqtSignal(object, object)

    def __init__(self, X, y, w_len, dw, epochs, val_split, model_path):
        super().__init__()
        self.X = X
        self.y = y
        self.w_len = w_len
        self.dw = dw
        self.epochs = epochs
        self.val_split = val_split
        self.model_path = model_path
        self.mdl = None

    def run(self):
        self.mdl, history = train_model(self.X, self.y, self.w_len, self.dw, self.epochs, 
                                        self.val_split, self.model_path, 
                                        plot=False, progress_callback=self.update_progress)
        self.finished_signal.emit(self.mdl, history)

    def update_progress(self, epoch, total_epochs):
        progress = int((epoch + 1) / total_epochs * 100)
        self.progress_signal.emit(progress)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    palette = QPalette()
    palette.setColor(QPalette.Window, QColor(30, 30, 30))
    palette.setColor(QPalette.WindowText, QColor(212, 212, 212))
    palette.setColor(QPalette.Base, QColor(30, 30, 30))
    palette.setColor(QPalette.AlternateBase, QColor(45, 45, 45))
    palette.setColor(QPalette.ToolTipBase, QColor(30, 30, 30))
    palette.setColor(QPalette.ToolTipText, QColor(212, 212, 212))
    palette.setColor(QPalette.Text, QColor(212, 212, 212))
    palette.setColor(QPalette.Button, QColor(45, 45, 45))
    palette.setColor(QPalette.ButtonText, QColor(212, 212, 212))
    palette.setColor(QPalette.BrightText, Qt.red)
    palette.setColor(QPalette.Link, QColor(14, 99, 156))
    palette.setColor(QPalette.Highlight, QColor(14, 99, 156))
    palette.setColor(QPalette.HighlightedText, Qt.white)
    app.setPalette(palette)

    # 设置全局字体
    font = QFont("Segoe UI", 10)
    app.setFont(font)

    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec_())