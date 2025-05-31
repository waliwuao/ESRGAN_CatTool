import cv2
import numpy as np
from PIL import Image
import os
import sys

from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QLineEdit, QFileDialog, QProgressBar, QTextEdit, QMessageBox, QCheckBox
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal

TILE_SIZE = 128
UPSCALE_FACTOR = 4
OVERLAP = 10


def get_resource_path(relative_path):
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)


class RealESRGANProcessor(QThread):
    progress_updated = pyqtSignal(int, int)
    status_message = pyqtSignal(str)
    processing_finished = pyqtSignal(str)
    error_occurred = pyqtSignal(str)

    def __init__(self, input_image_path, output_image_path, onnx_model_path, resize_to_1024=False, parent=None):
        super().__init__(parent)
        self.input_image_path = input_image_path
        self.output_image_path = output_image_path
        self.onnx_model_path = onnx_model_path
        self.resize_to_1024 = resize_to_1024  # 新增参数

    def run(self):
        self.process_image_with_realesrgan()

    def process_image_with_realesrgan(self):
        self.status_message.emit("正在初始化模型...")
        try:
            # 使用OpenCV的dnn模块加载ONNX模型
            net = cv2.dnn.readNetFromONNX(self.onnx_model_path)

            # 检测GPU是否可用
            if cv2.cuda.getCudaEnabledDeviceCount() > 0:
                net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
                net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
                self.status_message.emit("ONNX模型初始化成功，使用GPU进行推理")
            else:
                net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
                net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
                self.status_message.emit("未检测到可用GPU，使用CPU进行推理")
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

        if self.resize_to_1024:
            self.status_message.emit("正在调整图片大小以接近1024x1024像素...")
            img = self.resize_image_to_1024(img)

        img_width, img_height = img.size
        self.status_message.emit(f"处理后的图片尺寸: {img_width}x{img_height}")

        output_width = img_width * UPSCALE_FACTOR
        output_height = img_height * UPSCALE_FACTOR
        self.status_message.emit(f"预期输出图片尺寸: {output_width}x{output_height}")

        output_img_np_canvas = np.zeros((output_height, output_width, 3), dtype=np.float32)
        output_weights_canvas = np.zeros((output_height, output_width, 3), dtype=np.float32)

        # 确保至少有一个瓦片，即使图片尺寸非常小
        # 这里的瓦片计算逻辑是正确的，它会生成所有起始点，并在裁剪时处理边界
        y_coords = []
        current_y = 0
        while current_y < img_height:
            y_coords.append(current_y)
            current_y += (TILE_SIZE - OVERLAP)
        # 确保如果最后一个瓦片没有覆盖到图片底部，则添加一个从底部开始的瓦片
        if img_height > 0 and (img_height % (TILE_SIZE - OVERLAP) != 0 or img_height < TILE_SIZE):
            last_y = img_height - TILE_SIZE
            if last_y < 0: last_y = 0  # For images smaller than TILE_SIZE
            if last_y not in y_coords:
                y_coords.append(last_y)
        y_coords.sort()  # Ensure order and remove duplicates if any

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
        self.status_message.emit(f"大约需要{int(total_tiles/100*260)}秒")


        for i, y in enumerate(y_coords):
            for j, x in enumerate(x_coords):
                processed_tiles += 1
                self.progress_updated.emit(processed_tiles, total_tiles)
                self.status_message.emit(f"正在处理第{processed_tiles}个瓦片")

                x_start = x
                y_start = y
                x_end = min(x + TILE_SIZE, img_width)
                y_end = min(y + TILE_SIZE, img_height)

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
                    self.error_occurred.emit(f"模型推理失败于瓦片 ({x},{y}): {e}")
                    return

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
                out_y_end = min(out_y_end, output_height)
                out_x_end = min(out_x_end, output_width)

                effective_output_tile = effective_output_tile[:(out_y_end - out_y_start), :(out_x_end - out_x_start), :]

                output_img_np_canvas[out_y_start:out_y_end, out_x_start:out_x_end, :] += effective_output_tile
                output_weights_canvas[out_y_start:out_y_end, out_x_start:out_x_end, :] += 1.0

        self.status_message.emit("所有瓦片处理完毕，正在合成图像...")

        output_weights_canvas[output_weights_canvas == 0] = 1.0  # Avoid division by zero
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

    def resize_image_to_1024(self, img):
        original_width, original_height = img.size
        count = (1024 * 1024 / (original_width * original_height)) ** 0.5
        new_width = int(original_width * count)
        new_height = int(original_height * count)

        if original_height * original_width > 1024 * 1024:
            img = img.resize((new_width, new_height), Image.BILINEAR)
        else:
            img = img.resize((new_width, new_height), Image.BICUBIC)

        return img


class ImageUpscalerApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("哇哩唔嗷的 Real-ESRGAN 图像超分辨率工具")
        self.setAcceptDrops(True)  # 启用拖放功能
        self.input_image_path = ""
        self.output_image_path = ""
        self.onnx_model_path = get_resource_path("real_esrgan_x4plus-real-esrgan-x4plus-float.onnx")
        self.resize_to_1024 = False  # 新增变量

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

        # 新增复选框
        self.resize_checkbox = QCheckBox("调整图片大小至接近1024x1024像素")
        self.resize_checkbox.stateChanged.connect(self.update_resize_option)
        main_layout.addWidget(self.resize_checkbox)

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
        """重置拖放区域的样式"""
        self.drop_label.setStyleSheet(
            "QLabel { background-color: #f0f0f0; color: #333; font-size: 14px; border: 2px dashed #ccc; }")

    def update_resize_option(self, state):
        """更新是否调整图片大小的选项"""
        self.resize_to_1024 = state == Qt.Checked

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
        self.resize_checkbox.setEnabled(False)  # 禁用复选框
        self.progress_bar.setValue(0)
        self.status_text.clear()
        self.status_text.append("开始处理...")

        self.processor_thread = RealESRGANProcessor(
            self.input_image_path, self.output_image_path, self.onnx_model_path,
            resize_to_1024=self.resize_to_1024  # 传递是否调整图片大小的选项
        )
        self.processor_thread.progress_updated.connect(self.update_progress)
        self.processor_thread.status_message.connect(self.update_status)
        self.processor_thread.processing_finished.connect(self.on_processing_finished)
        self.processor_thread.error_occurred.connect(self.on_error)
        self.processor_thread.start()

    def update_progress(self, processed, total):
        self.progress_bar.setValue(int((processed / total) * 100))

    def update_status(self, message):
        self.status_text.append(message)

    def on_processing_finished(self, output_path):
        QMessageBox.information(self, "完成", f"超分辨率处理完成，输出文件已保存到：{output_path}")
        self.process_button.setEnabled(True)
        self.input_path_line.setEnabled(True)
        self.output_path_line.setEnabled(True)
        self.model_path_line.setEnabled(True)
        self.resize_checkbox.setEnabled(True)

    def on_error(self, message):
        QMessageBox.critical(self, "错误", message)
        self.process_button.setEnabled(True)
        self.input_path_line.setEnabled(True)
        self.output_path_line.setEnabled(True)
        self.model_path_line.setEnabled(True)
        self.resize_checkbox.setEnabled(True)

    def select_image_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "选择图片文件", "", "Image Files (*.png *.jpg *.jpeg *.bmp)")
        if file_path:
            self.input_image_path = file_path
            self.input_path_line.setText(file_path)
            self.output_image_path = os.path.splitext(file_path)[0] + "_upscaled.png"
            self.output_path_line.setText(self.output_image_path)
            self.process_button.setEnabled(True)

    def update_output_path(self, text):
        self.output_image_path = text

    def update_model_path(self, text):
        self.onnx_model_path = text

    def select_model_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "选择ONNX模型文件", "", "ONNX Files (*.onnx)")
        if file_path:
            self.onnx_model_path = file_path
            self.model_path_line.setText(file_path)

    def check_model_exists(self):
        if not os.path.exists(self.onnx_model_path):
            QMessageBox.warning(self, "警告", f"默认模型文件未找到，请选择一个有效的ONNX模型文件！")
            self.model_path_line.clear()
            self.process_button.setEnabled(False)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.accept()
        else:
            event.ignore()

    def dropEvent(self, event):
        file_path = event.mimeData().urls()[0].toLocalFile()
        if os.path.isfile(file_path):
            self.input_image_path = file_path
            self.input_path_line.setText(file_path)
            self.output_image_path = os.path.splitext(file_path)[0] + "_upscaled.png"
            self.output_path_line.setText(self.output_image_path)
            self.process_button.setEnabled(True)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ImageUpscalerApp()
    window.show()
    sys.exit(app.exec_())

