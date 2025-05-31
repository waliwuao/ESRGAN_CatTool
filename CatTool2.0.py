import cv2
import numpy as np
from PIL import Image
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QLineEdit, QFileDialog, QProgressBar, QTextEdit, QMessageBox
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal

TILE_SIZE = 128
UPSCALE_FACTOR = 4
OVERLAP = 10
MAX_THREADS = 4  # 最大线程数

def get_resource_path(relative_path):
    """获取资源文件的绝对路径"""
    try:
        # PyInstaller创建的临时目录
        base_path = sys._MEIPASS
    except Exception:
        # 普通Python脚本运行时
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)

class RealESRGANProcessor(QThread):
    progress_updated = pyqtSignal(int, int)
    status_message = pyqtSignal(str)
    processing_finished = pyqtSignal(str)
    error_occurred = pyqtSignal(str)

    def __init__(self, input_image_path, output_image_path, onnx_model_path, parent=None):
        super().__init__(parent)
        self.input_image_path = input_image_path
        self.output_image_path = output_image_path
        self.onnx_model_path = onnx_model_path

    def run(self):
        self.process_image_with_realesrgan()

    def process_image_with_realesrgan(self):
        self.status_message.emit("正在初始化模型...")
        try:
            # 使用OpenCV的dnn模块加载ONNX模型
            net = cv2.dnn.readNetFromONNX(self.onnx_model_path)
            # 设置为CPU模式
            net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
            self.status_message.emit("ONNX模型初始化成功，使用CPU进行推理")
        except Exception as e:
            self.error_occurred.emit(f"ONNX模型初始化失败: {e}")
            return

        self.status_message.emit(f"正在读取图片：{self.input_image_path}...")
        try:
            img = Image.open(self.input_image_path).convert("RGB")  # 确保是RGB模式
        except FileNotFoundError:
            self.error_occurred.emit(f"错误：找不到输入图片文件：{self.input_image_path}")
            return
        except Exception as e:
            self.error_occurred.emit(f"读取图片失败：{e}")
            return

        img_width, img_height = img.size
        self.status_message.emit(f"原始图片尺寸: {img_width}x{img_height}")

        output_width = img_width * UPSCALE_FACTOR
        output_height = img_height * UPSCALE_FACTOR
        self.status_message.emit(f"预期输出图片尺寸: {output_width}x{output_height}")

        output_img_np_canvas = np.zeros((output_height, output_width, 3), dtype=np.float32)
        output_weights_canvas = np.zeros((output_height, output_width, 3), dtype=np.float32)

        # 计算瓦片坐标
        y_coords = []
        current_y = 0
        while current_y < img_height:
            y_coords.append(current_y)
            current_y += (TILE_SIZE - OVERLAP)
        if img_height > 0 and (img_height % (TILE_SIZE - OVERLAP) != 0 or img_height < TILE_SIZE):
            last_y = img_height - TILE_SIZE
            if last_y < 0: last_y = 0
            if last_y not in y_coords:
                y_coords.append(last_y)
        y_coords.sort()

        x_coords = []
        current_x = 0
        while current_x < img_width:
            x_coords.append(current_x)
            current_x += (TILE_SIZE - OVERLAP)
        if img_width > 0 and (img_width % (TILE_SIZE - OVERLAP) != 0 or img_width < TILE_SIZE):
            last_x = img_width - TILE_SIZE
            if last_x < 0: last_x = 0
            if last_x not in x_coords:
                x_coords.append(last_x)
        x_coords.sort()

        total_tiles = len(y_coords) * len(x_coords)
        processed_tiles = 0

        self.status_message.emit(f"开始处理 {total_tiles} 个瓦片...")

        # 使用线程池并行处理瓦片
        with ThreadPoolExecutor(max_workers=MAX_THREADS) as executor:
            futures = []
            for i, y in enumerate(y_coords):
                for j, x in enumerate(x_coords):
                    # 为每个线程创建独立的模型实例
                    net_copy = cv2.dnn.readNetFromONNX(self.onnx_model_path)
                    net_copy.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
                    net_copy.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
                    futures.append(executor.submit(self.process_tile, net_copy, img, x, y, output_img_np_canvas, output_weights_canvas))
            for future in futures:
                try:
                    future.result()
                    processed_tiles += 1
                    self.progress_updated.emit(processed_tiles, total_tiles)
                except Exception as e:
                    self.error_occurred.emit(f"处理瓦片失败: {e}")
                    return

        self.status_message.emit("所有瓦片处理完毕，正在合成图像...")

        output_weights_canvas[output_weights_canvas == 0] = 1.0  # 避免除零
        final_output_img_np = output_img_np_canvas / output_weights_canvas
        final_output_img_np = np.clip(final_output_img_np, 0, 255).astype(np.uint8)

        output_img = Image.fromarray(final_output_img_np)

        self.status_message.emit(f"正在保存超分辨率图片到：{self.output_image_path}...")
        try:
            output_img.save(self.output_image_path)
            self.status_message.emit("图片保存成功。")
            self.processing_finished.emit(self.output_image_path)
        except Exception as e:
            self.error_occurred.emit(f"保存图片失败：{e}")

    def process_tile(self, net, img, x, y, output_img_np_canvas, output_weights_canvas):
        x_start = x
        y_start = y
        x_end = min(x + TILE_SIZE, img.size[0])
        y_end = min(y + TILE_SIZE, img.size[1])

        input_tile = img.crop((x_start, y_start, x_end, y_end))

        actual_tile_width = x_end - x_start
        actual_tile_height = y_end - y_start

        padded_input_tile = Image.new('RGB', (TILE_SIZE, TILE_SIZE), (0, 0, 0))
        padded_input_tile.paste(input_tile, (0, 0))

        tile_np = np.array(padded_input_tile).astype(np.float32) / 255.0
        tile_np = np.transpose(tile_np, (2, 0, 1))
        input_tensor = np.expand_dims(tile_np, axis=0)

        try:
            # 使用OpenCV的dnn模块进行推理
            net.setInput(input_tensor)
            output_tensor = net.forward()
        except Exception as e:
            raise Exception(f"模型推理失败于瓦片 ({x},{y}): {e}")

        output_tile_np = output_tensor[0]
        output_tile_np = np.transpose(output_tile_np, (1, 2, 0))  # CHW to HWC
        output_tile_np = output_tile_np * 255.0
        output_tile_np = np.clip(output_tile_np, 0, 255).astype(np.uint8)

        effective_output_tile_width = actual_tile_width * UPSCALE_FACTOR
        effective_output_tile_height = actual_tile_height * UPSCALE_FACTOR

        effective_output_tile = output_tile_np[:effective_output_tile_height, :effective_output_tile_width, :]

        out_x_start = x_start * UPSCALE_FACTOR
        out_y_start = y_start * UPSCALE_FACTOR
        out_x_end = out_x_start + effective_output_tile_width
        out_y_end = out_y_start + effective_output_tile_height

        # Ensure slices are within bounds
        out_y_end = min(out_y_end, output_img_np_canvas.shape[0])
        out_x_end = min(out_x_end, output_img_np_canvas.shape[1])

        effective_output_tile = effective_output_tile[:(out_y_end - out_y_start), :(out_x_end - out_x_start), :]

        output_img_np_canvas[out_y_start:out_y_end, out_x_start:out_x_end, :] += effective_output_tile
        output_weights_canvas[out_y_start:out_y_end, out_x_start:out_x_end, :] += 1.0

class ImageUpscalerApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("哇哩唔嗷的 Real-ESRGAN 图像超分辨率工具")
        self.setAcceptDrops(True)  # 启用拖放功能
        self.input_image_path = ""
        self.output_image_path = ""
        self.onnx_model_path = get_resource_path("real_esrgan_x4plus-real-esrgan-x4plus-float.onnx")

        self.init_ui()
        self.check_model_exists()

    def init_ui(self):
        main_layout = QVBoxLayout()

        # 1. 拖放区域
        self.drop_label = QLabel("将图片文件拖放到此处 或 点击 '选择图片'")
        self.drop_label.setAlignment(Qt.AlignCenter)
        self.drop_label.setFixedSize(400, 100)
        self.reset_drop_label_style()  # 使用函数重置样式
        main_layout.addWidget(self.drop_label, alignment=Qt.AlignCenter)

        select_button = QPushButton("选择图片文件")
        select_button.clicked.connect(self.select_image_file)
        main_layout.addWidget(select_button)

        input_layout = QHBoxLayout()
        input_layout.addWidget(QLabel("输入路径:"))
        self.input_path_line = QLineEdit()
        self.input_path_line.setReadOnly(True)
        input_layout.addWidget(self.input_path_line)
        main_layout.addLayout(input_layout)

        output_layout = QHBoxLayout()
        output_layout.addWidget(QLabel("输出路径:"))
        self.output_path_line = QLineEdit()
        self.output_path_line.textChanged.connect(self.update_output_path)
        output_layout.addWidget(self.output_path_line)
        main_layout.addLayout(output_layout)

        model_layout = QHBoxLayout()
        model_layout.addWidget(QLabel("模型路径:"))
        self.model_path_line = QLineEdit(self.onnx_model_path)
        self.model_path_line.textChanged.connect(self.update_model_path)
        model_layout.addWidget(self.model_path_line)
        model_browse_button = QPushButton("浏览模型")
        model_browse_button.clicked.connect(self.select_model_file)
        model_layout.addWidget(model_browse_button)
        main_layout.addLayout(model_layout)

        self.process_button = QPushButton("开始超分辨率处理")
        self.process_button.clicked.connect(self.start_processing)
        self.process_button.setEnabled(False)
        main_layout.addWidget(self.process_button)

        self.progress_bar = QProgressBar()
        self.progress_bar.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(self.progress_bar)

        self.status_text = QTextEdit()
        self.status_text.setReadOnly(True)
        self.status_text.setFixedHeight(100)
        main_layout.addWidget(self.status_text)

        self.setLayout(main_layout)
        self.setFixedSize(500, 500)

    def reset_drop_label_style(self):
        """重置拖放区域的默认样式"""
        self.drop_label.setStyleSheet(
            "QLabel { "
            "   border: 2px dashed #aaa;"
            "   border-radius: 5px;"
            "   background-color: #f8f8f8;"
            "   font-size: 14px;"
            "   color: #555;"
            "}"
            "QLabel:hover { "
            "   border: 2px dashed #888;"
            "   background-color: #f0f0f0;"
            "}"
        )

    def check_model_exists(self):
        if not os.path.exists(self.onnx_model_path):
            QMessageBox.warning(self, "模型文件缺失",
                                f"找不到ONNX模型文件：\n{self.onnx_model_path}\n"
                                "请确保模型文件与程序在同一目录下，或手动指定模型路径。",
                                QMessageBox.Ok)
            self.model_path_line.setStyleSheet("background-color: #ffe0e0;")  # 红色背景提示
            self.process_button.setEnabled(False)
        else:
            self.model_path_line.setStyleSheet("")  # 恢复正常背景
            self.enable_process_button_if_ready()

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
            self.drop_label.setStyleSheet(
                "QLabel { "
                "   border: 2px dashed #888;"
                "   border-radius: 5px;"
                "   background-color: #e0e0ff;"  # 拖入时改变背景色
                "   font-size: 14px;"
                "   color: #555;"
                "}"
            )
        else:
            event.ignore()

    def dragLeaveEvent(self, event):
        self.reset_drop_label_style()  # 离开时重置样式
        event.accept()

    def dropEvent(self, event):
        self.reset_drop_label_style()  # 拖放结束后重置样式
        for url in event.mimeData().urls():
            file_path = url.toLocalFile()
            if os.path.isfile(file_path) and file_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                self.set_input_image_path(file_path)
                break
        event.acceptProposedAction()

    def select_image_file(self):
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择输入图片文件", "",
            "图片文件 (*.png *.jpg *.jpeg *.bmp *.tiff);;所有文件 (*)", options=options
        )
        if file_path:
            self.set_input_image_path(file_path)

    def select_model_file(self):
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择ONNX模型文件", "",
            "ONNX模型文件 (*.onnx);;所有文件 (*)", options=options
        )
        if file_path:
            self.onnx_model_path = file_path
            self.model_path_line.setText(file_path)
            self.check_model_exists()  # 重新检查模型路径

    def update_model_path(self, path):
        self.onnx_model_path = path
        self.check_model_exists()  # 用户手动修改后也检查一下

    def set_input_image_path(self, path):
        self.input_image_path = path
        self.input_path_line.setText(path)
        self.generate_output_path(path)
        self.enable_process_button_if_ready()

    def generate_output_path(self, input_path):
        if input_path:
            dir_name = os.path.dirname(input_path)
            file_name_without_ext = os.path.splitext(os.path.basename(input_path))[0]
            output_file_name = f"{file_name_without_ext}_upscaled.png"
            self.output_image_path = os.path.join(dir_name, output_file_name)
            self.output_path_line.setText(self.output_image_path)
        else:
            self.output_image_path = ""
            self.output_path_line.setText("")

    def update_output_path(self, text):
        self.output_image_path = text
        self.enable_process_button_if_ready()

    def enable_process_button_if_ready(self):
        self.process_button.setEnabled(
            bool(self.input_image_path) and
            bool(self.output_image_path) and
            os.path.exists(self.onnx_model_path)  # 确保模型存在
        )

    def start_processing(self):
        if not self.input_image_path:
            QMessageBox.warning(self, "警告", "请先选择一个输入图片文件！")
            return
        if not self.output_image_path:
            QMessageBox.warning(self, "警告", "输出图片路径不能为空！")
            return
        if not os.path.exists(self.onnx_model_path):
            QMessageBox.critical(self, "错误", "ONNX模型文件不存在，请检查路径！")
            return

        # 禁用按钮防止重复点击
        self.process_button.setEnabled(False)
        self.input_path_line.setEnabled(False)
        self.output_path_line.setEnabled(False)
        self.model_path_line.setEnabled(False)
        self.progress_bar.setValue(0)
        self.status_text.clear()
        self.status_text.append("开始处理...")

        self.processor_thread = RealESRGANProcessor(
            self.input_image_path, self.output_image_path, self.onnx_model_path
        )
        self.processor_thread.progress_updated.connect(self.update_progress)
        self.processor_thread.status_message.connect(self.update_status)
        self.processor_thread.processing_finished.connect(self.on_processing_finished)
        self.processor_thread.error_occurred.connect(self.on_error)
        self.processor_thread.start()

    def update_progress(self, current, total):
        if total > 0:
            percentage = int((current / total) * 100)
            self.progress_bar.setValue(percentage)
            self.status_text.verticalScrollBar().setValue(self.status_text.verticalScrollBar().maximum())
        self.status_text.append(f"处理瓦片: {current}/{total}")

    def update_status(self, message):
        self.status_text.append(message)
        self.status_text.verticalScrollBar().setValue(self.status_text.verticalScrollBar().maximum())

    def on_processing_finished(self, output_path):
        self.status_text.append(f"处理完成！输出图片已保存到：{output_path}")
        QMessageBox.information(self, "完成", f"图片超分辨率处理成功！\n输出文件：{output_path}", QMessageBox.Ok)
        if hasattr(self, 'processor_thread') and self.processor_thread.isRunning():
            self.processor_thread.quit()
            self.processor_thread.wait()
        self.reset_ui()

    def on_error(self, message):
        self.status_text.append(f"错误：{message}")
        QMessageBox.critical(self, "错误", f"处理过程中发生错误：\n{message}", QMessageBox.Ok)
        if hasattr(self, 'processor_thread') and self.processor_thread.isRunning():
            self.processor_thread.quit()
            self.processor_thread.wait()
        self.reset_ui()

    def reset_ui(self):
        self.process_button.setEnabled(True)
        self.input_path_line.setEnabled(True)
        self.output_path_line.setEnabled(True)
        self.model_path_line.setEnabled(True)

        self.input_image_path = ""
        self.output_image_path = ""
        self.input_path_line.clear()
        self.output_path_line.clear()
        self.progress_bar.setValue(0)

        self.reset_drop_label_style()

        self.enable_process_button_if_ready()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ImageUpscalerApp()
    window.show()
    sys.exit(app.exec_())

