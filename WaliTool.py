import cv2
import numpy as np
from PIL import Image
import os
import sys

from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QLineEdit, QFileDialog, QProgressBar, QTextEdit, QMessageBox, QDoubleSpinBox,
    QGridLayout, QComboBox, QStackedWidget
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal

# 保持这些常量
TILE_SIZE = 128
OVERLAP = 10


def get_resource_path(relative_path):
    """获取资源文件的绝对路径"""
    try:
        # PyInstaller创建的临时目录
        base_path = sys._MEIPASS
    except Exception:
        # 普通Python脚本运行时
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)


# --- RealESRGANProcessor 线程 (超分辨率) ---
class RealESRGANProcessor(QThread):
    progress_updated = pyqtSignal(int, int)
    status_message = pyqtSignal(str)
    processing_finished = pyqtSignal(str)
    error_occurred = pyqtSignal(str)

    def __init__(self, input_image_path, output_image_path, onnx_model_path, upscale_factor, parent=None):
        super().__init__(parent)
        self.input_image_path = input_image_path
        self.output_image_path = output_image_path
        self.onnx_model_path = onnx_model_path
        self.upscale_factor = upscale_factor

    def run(self):
        self.process_image_with_realesrgan()

    def process_image_with_realesrgan(self):
        self.status_message.emit("正在初始化 Real-ESRGAN 模型...")
        try:
            net = cv2.dnn.readNetFromONNX(self.onnx_model_path)
            net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
            self.status_message.emit("ONNX模型初始化成功，使用CPU进行推理")
        except Exception as e:
            self.error_occurred.emit(f"ONNX模型初始化失败: {e}")
            return

        self.status_message.emit(f"正在读取图片：{self.input_image_path}...")
        try:
            img = Image.open(self.input_image_path).convert("RGB")
        except FileNotFoundError:
            self.error_occurred.emit(f"错误：找不到输入图片文件：{self.input_image_path}")
            return
        except Exception as e:
            self.error_occurred.emit(f"读取图片失败：{e}")
            return

        img_width, img_height = img.size
        self.status_message.emit(f"原始图片尺寸: {img_width}x{img_height}")

        output_width = int(img_width * self.upscale_factor)
        output_height = int(img_height * self.upscale_factor)
        self.status_message.emit(
            f"预期输出图片尺寸: {output_width}x{output_height} (超分辨率放缩比例: {self.upscale_factor:.2f})")

        output_img_np_canvas = np.zeros((output_height, output_width, 3), dtype=np.float32)
        output_weights_canvas = np.zeros((output_height, output_width, 3), dtype=np.float32)

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

        self.status_message.emit(f"开始处理 Real-ESRGAN 瓦片 {total_tiles} 个...")

        # 模型本身输出的放大倍数 (RealESRGAN通常是4x)
        model_base_upscale = 4

        for i, y in enumerate(y_coords):
            for j, x in enumerate(x_coords):
                processed_tiles += 1
                self.progress_updated.emit(processed_tiles, total_tiles)

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
                    net.setInput(input_tensor)
                    output_tensor = net.forward()
                except Exception as e:
                    self.error_occurred.emit(f"模型推理失败于瓦片 ({x},{y}): {e}")
                    return

                output_tile_np = output_tensor[0]
                output_tile_np = np.transpose(output_tile_np, (1, 2, 0))  # CHW to HWC
                output_tile_np = output_tile_np * 255.0
                output_tile_np = np.clip(output_tile_np, 0, 255).astype(np.uint8)

                # 计算输出瓦片需要被缩放的比例 (相对于模型基础输出)
                internal_scale_factor = self.upscale_factor / model_base_upscale

                full_model_output_tile = output_tile_np

                if internal_scale_factor != 1.0:
                    # 使用 CV2 resize 进行二次缩放，保持图像通道顺序
                    full_model_output_tile = cv2.resize(
                        full_model_output_tile,
                        (int(full_model_output_tile.shape[1] * internal_scale_factor),
                         int(full_model_output_tile.shape[0] * internal_scale_factor)),
                        interpolation=cv2.INTER_LANCZOS4
                    )

                final_effective_output_tile_width = int(actual_tile_width * self.upscale_factor)
                final_effective_output_tile_height = int(actual_tile_height * self.upscale_factor)

                effective_output_tile = full_model_output_tile[:final_effective_output_tile_height,
                                        :final_effective_output_tile_width, :]

                out_x_start = int(x_start * self.upscale_factor)
                out_y_start = int(y_start * self.upscale_factor)
                out_x_end = out_x_start + effective_output_tile.shape[1]
                out_y_end = out_y_start + effective_output_tile.shape[0]

                out_y_end = min(out_y_end, output_height)
                out_x_end = min(out_x_end, output_width)

                effective_output_tile = effective_output_tile[:(out_y_end - out_y_start), :(out_x_end - out_x_start), :]

                output_img_np_canvas[out_y_start:out_y_end, out_x_start:out_x_end, :] += effective_output_tile
                output_weights_canvas[out_y_start:out_y_end, out_x_start:out_x_end, :] += 1.0

        self.status_message.emit("所有 Real-ESRGAN 瓦片处理完毕，正在合成图像...")

        output_weights_canvas[output_weights_canvas == 0] = 1.0
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


# --- BasicImageScaler 线程 (纯粹图片放缩) ---
class BasicImageScaler(QThread):
    progress_updated = pyqtSignal(int, int)  # 这里 progress_updated 主要用于显示状态，而非进度条
    status_message = pyqtSignal(str)
    processing_finished = pyqtSignal(str)
    error_occurred = pyqtSignal(str)

    def __init__(self, input_image_path, output_image_path, scale_factor, interpolation_method, parent=None):
        super().__init__(parent)
        self.input_image_path = input_image_path
        self.output_image_path = output_image_path
        self.scale_factor = scale_factor
        self.interpolation_method = interpolation_method

    def run(self):
        self.scale_image()

    def scale_image(self):
        self.status_message.emit(f"正在读取图片：{self.input_image_path}...")
        try:
            # OpenCV 默认使用 BGR，PIL 使用 RGB，这里转换一下
            img_pil = Image.open(self.input_image_path).convert("RGB")
            img_np = np.array(img_pil)
            img_cv = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        except FileNotFoundError:
            self.error_occurred.emit(f"错误：找不到输入图片文件：{self.input_image_path}")
            return
        except Exception as e:
            self.error_occurred.emit(f"读取图片失败：{e}")
            return

        original_height, original_width = img_cv.shape[:2]
        new_width = int(original_width * self.scale_factor)
        new_height = int(original_height * self.scale_factor)

        self.status_message.emit(f"原始尺寸: {original_width}x{original_height}, 目标尺寸: {new_width}x{new_height}")

        interpolation_map = {
            "最近邻插值 (Nearest)": cv2.INTER_NEAREST,
            "双线性插值 (Linear)": cv2.INTER_LINEAR,
            "双三次插值 (Cubic)": cv2.INTER_CUBIC,
            "Lanczos插值 (Lanczos4)": cv2.INTER_LANCZOS4,
            "区域插值 (Area)": cv2.INTER_AREA  # 缩小图片时推荐
        }
        inter_method = interpolation_map.get(self.interpolation_method, cv2.INTER_LANCZOS4)

        self.status_message.emit(f"正在使用 {self.interpolation_method} 进行图片放缩...")
        try:
            # OpenCV resize 接受 (width, height)
            scaled_img_cv = cv2.resize(img_cv, (new_width, new_height), interpolation=inter_method)
            scaled_img_pil = Image.fromarray(cv2.cvtColor(scaled_img_cv, cv2.COLOR_BGR2RGB))
        except Exception as e:
            self.error_occurred.emit(f"图片放缩失败: {e}")
            return

        self.status_message.emit(f"正在保存放缩后的图片到：{self.output_image_path}...")
        try:
            scaled_img_pil.save(self.output_image_path)
            self.status_message.emit("图片保存成功。")
            self.processing_finished.emit(self.output_image_path)
        except Exception as e:
            self.error_occurred.emit(f"保存图片失败：{e}")


class ImageUpscalerApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("哇哩唔嗷的工具箱")
        self.setAcceptDrops(True)
        self.input_image_path = ""
        self.output_image_path = ""
        self.onnx_model_path = get_resource_path("real_esrgan_x4plus-real-esrgan-x4plus-float.onnx")
        self.upscale_factor = 4.0  # 默认 Real-ESRGAN 比例
        self.scale_factor = 2.0  # 默认纯放缩比例
        self.current_mode = "Real-ESRGAN 超分辨率"  # 默认模式
        self.interpolation_method = "Lanczos插值 (Lanczos4)"  # 默认插值方法

        self.init_ui()
        self.check_model_exists()

    def init_ui(self):
        main_layout = QVBoxLayout()
        control_layout = QGridLayout()

        # 功能选择
        mode_select_layout = QHBoxLayout()
        mode_select_layout.addWidget(QLabel("选择功能:"))
        self.mode_combo = QComboBox()
        self.mode_combo.addItem("Real-ESRGAN 超分辨率")
        self.mode_combo.addItem("图片放缩")
        self.mode_combo.currentIndexChanged.connect(self.change_mode)
        mode_select_layout.addWidget(self.mode_combo)
        mode_select_layout.addStretch(1)
        main_layout.addLayout(mode_select_layout)

        # 1. 拖放区域
        self.drop_label = QLabel("将图片文件拖放到此处 或 点击 '选择图片'")
        self.drop_label.setAlignment(Qt.AlignCenter)
        self.drop_label.setFixedSize(400, 100)
        self.reset_drop_label_style()
        main_layout.addWidget(self.drop_label, alignment=Qt.AlignCenter)

        select_button = QPushButton("选择图片文件")
        select_button.clicked.connect(self.select_image_file)
        main_layout.addWidget(select_button)

        # 输入路径
        control_layout.addWidget(QLabel("输入路径:"), 0, 0)
        self.input_path_line = QLineEdit()
        self.input_path_line.setReadOnly(True)
        control_layout.addWidget(self.input_path_line, 0, 1, 1, 2)

        # 输出路径
        control_layout.addWidget(QLabel("输出路径:"), 1, 0)
        self.output_path_line = QLineEdit()
        self.output_path_line.textChanged.connect(self.update_output_path)
        control_layout.addWidget(self.output_path_line, 1, 1, 1, 2)

        # ----------------------------------------------------
        # 使用 QStackedWidget 来切换特定模式的控件
        self.stacked_widget = QStackedWidget()

        # Real-ESRGAN 模式的专属控件
        self.esrgan_page = QWidget()
        esrgan_layout = QGridLayout(self.esrgan_page)
        esrgan_layout.setContentsMargins(0, 0, 0, 0)  # 移除内部边距
        esrgan_layout.addWidget(QLabel("模型路径:"), 0, 0)
        self.model_path_line = QLineEdit(self.onnx_model_path)
        self.model_path_line.textChanged.connect(self.update_model_path)
        esrgan_layout.addWidget(self.model_path_line, 0, 1)
        model_browse_button = QPushButton("浏览模型")
        model_browse_button.clicked.connect(self.select_model_file)
        esrgan_layout.addWidget(model_browse_button, 0, 2)
        esrgan_layout.addWidget(QLabel("Real-ESRGAN 放缩比例:"), 1, 0)
        self.esrgan_upscale_factor_spinbox = QDoubleSpinBox()
        self.esrgan_upscale_factor_spinbox.setMinimum(0.5)
        self.esrgan_upscale_factor_spinbox.setMaximum(8.0)
        self.esrgan_upscale_factor_spinbox.setSingleStep(0.5)
        self.esrgan_upscale_factor_spinbox.setValue(self.upscale_factor)
        self.esrgan_upscale_factor_spinbox.valueChanged.connect(self.update_esrgan_upscale_factor)
        esrgan_layout.addWidget(self.esrgan_upscale_factor_spinbox, 1, 1)
        esrgan_layout.addWidget(QLabel("（原图尺寸 x 比例）"), 1, 2)
        self.stacked_widget.addWidget(self.esrgan_page)  # 索引 0

        # 纯粹图片放缩模式的专属控件
        self.basic_scaler_page = QWidget()
        basic_scaler_layout = QGridLayout(self.basic_scaler_page)
        basic_scaler_layout.setContentsMargins(0, 0, 0, 0)
        basic_scaler_layout.addWidget(QLabel("放缩比例:"), 0, 0)
        self.basic_scale_factor_spinbox = QDoubleSpinBox()
        self.basic_scale_factor_spinbox.setMinimum(0.1)  # 可以缩小到很小
        self.basic_scale_factor_spinbox.setMaximum(10.0)  # 可以放大到很大
        self.basic_scale_factor_spinbox.setSingleStep(0.1)
        self.basic_scale_factor_spinbox.setValue(self.scale_factor)
        self.basic_scale_factor_spinbox.valueChanged.connect(self.update_basic_scale_factor)
        basic_scaler_layout.addWidget(self.basic_scale_factor_spinbox, 0, 1)
        basic_scaler_layout.addWidget(QLabel("（原图尺寸 x 比例）"), 0, 2)
        basic_scaler_layout.addWidget(QLabel("插值方法:"), 1, 0)
        self.interpolation_combo = QComboBox()
        self.interpolation_combo.addItem("Lanczos插值 (Lanczos4)")
        self.interpolation_combo.addItem("双三次插值 (Cubic)")
        self.interpolation_combo.addItem("双线性插值 (Linear)")
        self.interpolation_combo.addItem("最近邻插值 (Nearest)")
        self.interpolation_combo.addItem("区域插值 (Area)")
        self.interpolation_combo.setCurrentText(self.interpolation_method)
        self.interpolation_combo.currentTextChanged.connect(self.update_interpolation_method)
        basic_scaler_layout.addWidget(self.interpolation_combo, 1, 1, 1, 2)
        self.stacked_widget.addWidget(self.basic_scaler_page)  # 索引 1
        # ----------------------------------------------------

        # 将 QStackedWidget 添加到主控制布局中
        control_layout.addWidget(self.stacked_widget, 2, 0, 2, 3)  # 跨行跨列

        main_layout.addLayout(control_layout)

        self.process_button = QPushButton("开始处理")
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
        self.setFixedSize(500, 600)  # 稍微增加高度以适应新的控件

        # 初始显示 Real-ESRGAN 页面
        self.stacked_widget.setCurrentIndex(0)
        self.change_mode(0)  # 确保初始状态正确

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

    def change_mode(self, index):
        """根据选择切换显示不同的控件"""
        self.current_mode = self.mode_combo.currentText()
        self.stacked_widget.setCurrentIndex(index)
        self.generate_output_path(self.input_image_path)  # 切换模式后更新输出文件名

        # 根据模式启用/禁用相关控件
        if self.current_mode == "Real-ESRGAN 超分辨率":
            self.model_path_line.setEnabled(True)
            self.esrgan_upscale_factor_spinbox.setEnabled(True)
            self.basic_scale_factor_spinbox.setEnabled(False)
            self.interpolation_combo.setEnabled(False)
        else:  # 纯粹图片放缩
            self.model_path_line.setEnabled(False)
            self.esrgan_upscale_factor_spinbox.setEnabled(False)
            self.basic_scale_factor_spinbox.setEnabled(True)
            self.interpolation_combo.setEnabled(True)
        self.check_model_exists()  # 重新检查模型是否存在 (因为Real-ESRGAN模式依赖它)
        self.enable_process_button_if_ready()

    def check_model_exists(self):
        if self.current_mode == "Real-ESRGAN 超分辨率":
            if not os.path.exists(self.onnx_model_path):
                self.model_path_line.setStyleSheet("background-color: #ffe0e0;")
                # 不再弹框，只通过背景色提示，并在 process_button 禁用
                self.status_text.append(
                    f"<span style='color: red;'>警告: Real-ESRGAN模型文件缺失: {self.onnx_model_path}</span>")
            else:
                self.model_path_line.setStyleSheet("")
        else:  # 纯粹图片放缩模式下，模型路径无所谓
            self.model_path_line.setStyleSheet("")
        self.enable_process_button_if_ready()

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
            self.drop_label.setStyleSheet(
                "QLabel { "
                "   border: 2px dashed #888;"
                "   border-radius: 5px;"
                "   background-color: #e0e0ff;"
                "   font-size: 14px;"
                "   color: #555;"
                "}"
            )
        else:
            event.ignore()

    def dragLeaveEvent(self, event):
        self.reset_drop_label_style()
        event.accept()

    def dropEvent(self, event):
        self.reset_drop_label_style()
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
            self.check_model_exists()

    def update_model_path(self, path):
        self.onnx_model_path = path
        self.check_model_exists()

    def update_esrgan_upscale_factor(self, value):
        self.upscale_factor = value
        self.generate_output_path(self.input_image_path)

    def update_basic_scale_factor(self, value):
        self.scale_factor = value
        self.generate_output_path(self.input_image_path)

    def update_interpolation_method(self, text):
        self.interpolation_method = text
        self.generate_output_path(self.input_image_path)

    def set_input_image_path(self, path):
        self.input_image_path = path
        self.input_path_line.setText(path)
        self.generate_output_path(path)
        self.enable_process_button_if_ready()

    def generate_output_path(self, input_path):
        if input_path:
            dir_name = os.path.dirname(input_path)
            file_name_without_ext = os.path.splitext(os.path.basename(input_path))[0]

            if self.current_mode == "Real-ESRGAN 超分辨率":
                output_file_name = f"{file_name_without_ext}_ESRGAN_x{self.upscale_factor:.1f}.png"
            else:  # 纯粹图片放缩
                # 根据插值方法在文件名中添加标识
                interp_abbr = self.interpolation_method.split(' ')[0].replace('插值', '')
                output_file_name = f"{file_name_without_ext}_scaled_x{self.scale_factor:.1f}_{interp_abbr}.png"

            self.output_image_path = os.path.join(dir_name, output_file_name)
            self.output_path_line.setText(self.output_image_path)
        else:
            self.output_image_path = ""
            self.output_path_line.setText("")

    def update_output_path(self, text):
        self.output_image_path = text
        self.enable_process_button_if_ready()

    def enable_process_button_if_ready(self):
        is_ready = bool(self.input_image_path) and bool(self.output_image_path)

        if self.current_mode == "Real-ESRGAN 超分辨率":
            is_ready = is_ready and os.path.exists(self.onnx_model_path) and self.upscale_factor > 0
        else:  # 纯粹图片放缩
            is_ready = is_ready and self.scale_factor > 0

        self.process_button.setEnabled(is_ready)

    def start_processing(self):
        if not self.input_image_path:
            QMessageBox.warning(self, "警告", "请先选择一个输入图片文件！")
            return
        if not self.output_image_path:
            QMessageBox.warning(self, "警告", "输出图片路径不能为空！")
            return

        # 禁用所有输入控件
        self.process_button.setEnabled(False)
        self.input_path_line.setEnabled(False)
        self.output_path_line.setEnabled(False)
        self.mode_combo.setEnabled(False)
        self.esrgan_upscale_factor_spinbox.setEnabled(False)
        self.basic_scale_factor_spinbox.setEnabled(False)
        self.interpolation_combo.setEnabled(False)
        self.model_path_line.setEnabled(False)  # 即使不显示也禁用

        self.progress_bar.setValue(0)
        self.status_text.clear()
        self.status_text.append("开始处理...")

        if self.current_mode == "Real-ESRGAN 超分辨率":
            if not os.path.exists(self.onnx_model_path):
                QMessageBox.critical(self, "错误", "Real-ESRGAN ONNX模型文件不存在，请检查路径！")
                self.reset_ui()
                return
            self.processor_thread = RealESRGANProcessor(
                self.input_image_path, self.output_image_path, self.onnx_model_path, self.upscale_factor
            )
        else:  # 纯粹图片放缩
            self.processor_thread = BasicImageScaler(
                self.input_image_path, self.output_image_path, self.scale_factor, self.interpolation_method
            )

        self.processor_thread.progress_updated.connect(self.update_progress)
        self.processor_thread.status_message.connect(self.update_status)
        self.processor_thread.processing_finished.connect(self.on_processing_finished)
        self.processor_thread.error_occurred.connect(self.on_error)
        self.processor_thread.start()

    def update_progress(self, current, total):
        # 对于 BasicImageScaler，进度条可能只有 0 和 100
        if total > 0:
            percentage = int((current / total) * 100)
            self.progress_bar.setValue(percentage)
        else:  # For BasicImageScaler, just set to 0 or 100, no intermediate steps
            self.progress_bar.setValue(0 if current == 0 else 100)

        # 滚动到底部
        self.status_text.verticalScrollBar().setValue(self.status_text.verticalScrollBar().maximum())
        # 对于 BasicImageScaler，不需要每次都更新瓦片信息，只在 RealESRGAN模式下才需要
        if self.current_mode == "Real-ESRGAN 超分辨率":
            self.status_text.append(f"处理瓦片: {current}/{total}")

    def update_status(self, message):
        self.status_text.append(message)
        self.status_text.verticalScrollBar().setValue(self.status_text.verticalScrollBar().maximum())

    def on_processing_finished(self, output_path):
        self.status_text.append(f"处理完成！输出图片已保存到：{output_path}")
        QMessageBox.information(self, "完成", f"图片处理成功！\n输出文件：{output_path}", QMessageBox.Ok)
        self.cleanup_thread()
        self.reset_ui()

    def on_error(self, message):
        self.status_text.append(f"错误：{message}")
        QMessageBox.critical(self, "错误", f"处理过程中发生错误：\n{message}", QMessageBox.Ok)
        self.cleanup_thread()
        self.reset_ui()

    def cleanup_thread(self):
        if hasattr(self, 'processor_thread') and self.processor_thread.isRunning():
            self.processor_thread.quit()
            self.processor_thread.wait()  # 等待线程终止

    def reset_ui(self):
        # 重新启用所有输入控件
        self.input_path_line.setEnabled(True)
        self.output_path_line.setEnabled(True)
        self.mode_combo.setEnabled(True)

        # 根据当前模式重新启用对应控件
        self.change_mode(self.mode_combo.currentIndex())

        self.input_image_path = ""
        self.output_image_path = ""
        self.input_path_line.clear()
        self.output_path_line.clear()
        self.progress_bar.setValue(0)
        # self.status_text.clear() # 避免清空历史日志，让用户查看上次处理信息

        self.reset_drop_label_style()
        self.enable_process_button_if_ready()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ImageUpscalerApp()
    window.show()
    sys.exit(app.exec_())
