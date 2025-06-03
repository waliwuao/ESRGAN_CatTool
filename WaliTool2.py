import cv2
import numpy as np
from PIL import Image
import os
import sys
import time

from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QLineEdit, QFileDialog, QProgressBar, QTextEdit, QMessageBox, QDoubleSpinBox,
    QGridLayout, QComboBox, QStackedWidget, QSpinBox, QGroupBox, QScrollArea
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QMutex, QSize


# 保持这些常量
# TILE_SIZE 和 OVERLAP 现在会根据选择的模型动态调整


def get_resource_path(relative_path):
    """获取资源文件的绝对路径"""
    try:
        # PyInstaller创建的临时目录
        base_path = sys._MEIPASS
    except Exception:
        # 普通Python脚本运行时
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)


# 定义不同模型的参数
MODEL_SETTINGS = {
    "real_esrgan_x4plus-real-esrgan-x4plus-float.onnx": {
        "path": "real_esrgan_x4plus-real-esrgan-x4plus-float.onnx",
        "tile_size": 128,
        "overlap": 10,
        "default_upscale_factor": 4.0,
        "display_name": "Real-ESRGAN (通用)",
        "description": "适用于各种图片，平衡性能与效果"
    },
    "RealESRGAN_ANIME_6B_512x512.onnx": {
        "path": "RealESRGAN_ANIME_6B_512x512.onnx",
        "tile_size": 512,  # Anime模型通常需要更大的瓦片
        "overlap": 32,  # 瓦片增大，重叠也适当增大
        "default_upscale_factor": 4.0,
        "display_name": "Real-ESRGAN (动漫专用)",
        "description": "专为动漫和插画优化，效果更佳"
    }
}


# --- RealESRGANProcessor 线程 (图片超分辨率) ---
class RealESRGANProcessor(QThread):
    progress_updated = pyqtSignal(int, int)
    status_message = pyqtSignal(str)
    processing_finished = pyqtSignal(str)
    error_occurred = pyqtSignal(str)

    def __init__(self, input_image_path, output_image_path, onnx_model_path, upscale_factor, tile_size, overlap,
                 parent=None):
        super().__init__(parent)
        self.input_image_path = input_image_path
        self.output_image_path = output_image_path
        self.onnx_model_path = onnx_model_path
        self.upscale_factor = upscale_factor
        self.tile_size = tile_size
        self.overlap = overlap
        self._is_interrupted = False  # 用于更明确地控制中断

    def requestInterruption(self):
        super().requestInterruption()
        self._is_interrupted = True

    def run(self):
        self.process_image_with_realesrgan()

    def process_image_with_realesrgan(self):
        self.status_message.emit("正在初始化 Real-ESRGAN 模型...")
        try:
            # 模型初始化放在 run 方法内部，确保每个线程拥有独立的模型实例
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

        # 改进瓦片坐标计算
        y_coords = []
        current_y = 0
        while current_y < img_height:
            y_coords.append(current_y)
            current_y += (self.tile_size - self.overlap)
        if y_coords and y_coords[-1] < img_height - self.tile_size:  # 确保最后一个瓦片能覆盖到图像底部
            y_coords.append(img_height - self.tile_size)
        y_coords = sorted(list(set([max(0, coord) for coord in y_coords])))  # 避免负值，并去重排序

        x_coords = []
        current_x = 0
        while current_x < img_width:
            x_coords.append(current_x)
            current_x += (self.tile_size - self.overlap)
        if x_coords and x_coords[-1] < img_width - self.tile_size:  # 确保最后一个瓦片能覆盖到图像右侧
            x_coords.append(img_width - self.tile_size)
        x_coords = sorted(list(set([max(0, coord) for coord in x_coords])))  # 避免负值，并去重排序

        total_tiles = len(y_coords) * len(x_coords)
        processed_tiles = 0

        self.status_message.emit(
            f"开始处理 Real-ESRGAN 瓦片 {total_tiles} 个 (瓦片大小: {self.tile_size}x{self.tile_size}, 重叠: {self.overlap} 像素)...")

        model_base_upscale = 4

        for i, y in enumerate(y_coords):
            for j, x in enumerate(x_coords):
                if self._is_interrupted:  # 检查是否请求中断
                    self.status_message.emit("处理已中断。")
                    return

                processed_tiles += 1
                self.progress_updated.emit(processed_tiles, total_tiles)

                x_start = x
                y_start = y
                x_end = min(x + self.tile_size, img_width)
                y_end = min(y + self.tile_size, img_height)

                input_tile = img.crop((x_start, y_start, x_end, y_end))

                actual_tile_width = x_end - x_start
                actual_tile_height = y_end - y_start

                # 填充到模型期望的瓦片大小
                padded_input_tile = Image.new('RGB', (self.tile_size, self.tile_size), (0, 0, 0))
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

                # 处理 Real-ESRGAN 模型自带的4倍放大后，如果目标比例不是4倍，则进行二次放缩
                internal_scale_factor = self.upscale_factor / model_base_upscale

                full_model_output_tile = output_tile_np

                if internal_scale_factor != 1.0:
                    full_model_output_tile = cv2.resize(
                        full_model_output_tile,
                        (int(full_model_output_tile.shape[1] * internal_scale_factor),
                         int(full_model_output_tile.shape[0] * internal_scale_factor)),
                        interpolation=cv2.INTER_LANCZOS4
                    )

                # 裁剪出实际有效的部分
                final_effective_output_tile_width = int(actual_tile_width * self.upscale_factor)
                final_effective_output_tile_height = int(actual_tile_height * self.upscale_factor)

                effective_output_tile = full_model_output_tile[:final_effective_output_tile_height,
                                        :final_effective_output_tile_width, :]

                # 计算在最终输出画布上的位置
                out_x_start = int(x_start * self.upscale_factor)
                out_y_start = int(y_start * self.upscale_factor)
                out_x_end = out_x_start + effective_output_tile.shape[1]
                out_y_end = out_y_start + effective_output_tile.shape[0]

                # 确保不超过输出图像边界
                out_y_end = min(out_y_end, output_height)
                out_x_end = min(out_x_end, output_width)

                effective_output_tile = effective_output_tile[:(out_y_end - out_y_start), :(out_x_end - out_x_start), :]

                # 叠加到画布上
                output_img_np_canvas[out_y_start:out_y_end, out_x_start:out_x_end, :] += effective_output_tile
                output_weights_canvas[out_y_start:out_y_end, out_x_start:out_x_end, :] += 1.0

        self.status_message.emit("所有 Real-ESRGAN 瓦片处理完毕，正在合成图像...")

        # 防止除以零
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


# --- RealESRGANVideoProcessor 线程 (视频超分辨率) ---
class RealESRGANVideoProcessor(QThread):
    progress_updated = pyqtSignal(int, int)
    status_message = pyqtSignal(str)
    processing_finished = pyqtSignal(str)
    error_occurred = pyqtSignal(str)

    def __init__(self, input_video_path, output_video_path, onnx_model_path, upscale_factor, tile_size, overlap,
                 output_codec, output_fps, parent=None):
        super().__init__(parent)
        self.input_video_path = input_video_path
        self.output_video_path = output_video_path
        self.onnx_model_path = onnx_model_path
        self.upscale_factor = upscale_factor
        self.tile_size = tile_size
        self.overlap = overlap
        self.output_codec = output_codec
        self.output_fps = output_fps
        self._is_interrupted = False

    def requestInterruption(self):
        super().requestInterruption()
        self._is_interrupted = True

    def run(self):
        self.process_video_with_realesrgan()

    def process_video_with_realesrgan(self):
        self.status_message.emit("正在初始化 Real-ESRGAN 模型...")
        try:
            net = cv2.dnn.readNetFromONNX(self.onnx_model_path)
            net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
            self.status_message.emit("ONNX模型初始化成功，使用CPU进行推理")
        except Exception as e:
            self.error_occurred.emit(f"ONNX模型初始化失败: {e}")
            return

        self.status_message.emit(f"正在读取视频：{self.input_video_path}...")
        cap = cv2.VideoCapture(self.input_video_path)
        if not cap.isOpened():
            self.error_occurred.emit(f"错误：无法打开视频文件：{self.input_video_path}")
            return

        original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        original_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        self.status_message.emit(
            f"原始视频尺寸: {original_width}x{original_height}, 帧率: {original_fps:.2f}, 总帧数: {total_frames}")

        # 计算输出视频尺寸
        output_width = int(original_width * self.upscale_factor)
        output_height = int(original_height * self.upscale_factor)
        self.status_message.emit(
            f"预期输出视频尺寸: {output_width}x{output_height} (超分辨率放缩比例: {self.upscale_factor:.2f})")

        # 确定输出帧率
        final_fps = self.output_fps if self.output_fps > 0 else original_fps
        if final_fps == 0:  # 如果原视频也没有帧率信息，给个默认值
            final_fps = 30.0
            self.status_message.emit("警告: 无法获取原始视频帧率，将使用默认帧率 30 FPS。")

        # 设置视频编码器
        fourcc = cv2.VideoWriter_fourcc(*self.output_codec)
        out = cv2.VideoWriter(self.output_video_path, fourcc, final_fps, (output_width, output_height))

        if not out.isOpened():
            self.error_occurred.emit(f"错误：无法创建输出视频文件：{self.output_video_path}。请检查编码器和路径。")
            cap.release()
            return

        processed_frames = 0
        model_base_upscale = 4

        self.status_message.emit(
            f"开始处理视频帧 (瓦片大小: {self.tile_size}x{self.tile_size}, 重叠: {self.overlap} 像素)...")

        while True:
            if self._is_interrupted:
                self.status_message.emit("视频处理已中断。")
                break

            ret, frame = cap.read()
            if not ret:
                break

            processed_frames += 1
            self.progress_updated.emit(processed_frames, total_frames)
            self.status_message.emit(f"正在处理帧 {processed_frames}/{total_frames}...")

            # 将OpenCV BGR帧转换为PIL RGB图像
            img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            img_width, img_height = img_pil.size

            output_img_np_canvas = np.zeros((output_height, output_width, 3), dtype=np.float32)
            output_weights_canvas = np.zeros((output_height, output_width, 3), dtype=np.float32)

            # 瓦片处理逻辑，与图片处理相同
            y_coords = []
            current_y = 0
            while current_y < img_height:
                y_coords.append(current_y)
                current_y += (self.tile_size - self.overlap)
            if y_coords and y_coords[-1] < img_height - self.tile_size:
                y_coords.append(img_height - self.tile_size)
            y_coords = sorted(list(set([max(0, coord) for coord in y_coords])))

            x_coords = []
            current_x = 0
            while current_x < img_width:
                x_coords.append(current_x)
                current_x += (self.tile_size - self.overlap)
            if x_coords and x_coords[-1] < img_width - self.tile_size:
                x_coords.append(img_width - self.tile_size)
            x_coords = sorted(list(set([max(0, coord) for coord in x_coords])))

            for y in y_coords:
                for x in x_coords:
                    x_start = x
                    y_start = y
                    x_end = min(x + self.tile_size, img_width)
                    y_end = min(y + self.tile_size, img_height)

                    input_tile = img_pil.crop((x_start, y_start, x_end, y_end))
                    actual_tile_width = x_end - x_start
                    actual_tile_height = y_end - y_start

                    padded_input_tile = Image.new('RGB', (self.tile_size, self.tile_size), (0, 0, 0))
                    padded_input_tile.paste(input_tile, (0, 0))

                    tile_np = np.array(padded_input_tile).astype(np.float32) / 255.0
                    tile_np = np.transpose(tile_np, (2, 0, 1))
                    input_tensor = np.expand_dims(tile_np, axis=0)

                    try:
                        net.setInput(input_tensor)
                        output_tensor = net.forward()
                    except Exception as e:
                        self.error_occurred.emit(f"模型推理失败于帧 {processed_frames}, 瓦片 ({x},{y}): {e}")
                        cap.release()
                        out.release()
                        return

                    output_tile_np = output_tensor[0]
                    output_tile_np = np.transpose(output_tile_np, (1, 2, 0))  # CHW to HWC
                    output_tile_np = output_tile_np * 255.0
                    output_tile_np = np.clip(output_tile_np, 0, 255).astype(np.uint8)

                    internal_scale_factor = self.upscale_factor / model_base_upscale

                    full_model_output_tile = output_tile_np

                    if internal_scale_factor != 1.0:
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

                    effective_output_tile = effective_output_tile[:(out_y_end - out_y_start),
                                            :(out_x_end - out_x_start), :]

                    output_img_np_canvas[out_y_start:out_y_end, out_x_start:out_x_end, :] += effective_output_tile
                    output_weights_canvas[out_y_start:out_y_end, out_x_start:out_x_end, :] += 1.0

            output_weights_canvas[output_weights_canvas == 0] = 1.0
            final_output_frame_np = output_img_np_canvas / output_weights_canvas
            final_output_frame_np = np.clip(final_output_frame_np, 0, 255).astype(np.uint8)

            # 将处理后的RGB帧转换回BGR，并写入视频
            out.write(cv2.cvtColor(final_output_frame_np, cv2.COLOR_RGB2BGR))

        self.status_message.emit("所有视频帧处理完毕。")

        cap.release()
        out.release()
        self.status_message.emit(f"超分辨率视频已保存到：{self.output_video_path}。请注意，此视频不包含音频。")
        self.processing_finished.emit(self.output_video_path)


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
        self._is_interrupted = False

    def requestInterruption(self):
        super().requestInterruption()
        self._is_interrupted = True

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

        if self._is_interrupted:  # 检查中断
            self.status_message.emit("处理已中断。")
            return

        self.status_message.emit(f"正在保存放缩后的图片到：{self.output_image_path}...")
        try:
            scaled_img_pil.save(self.output_image_path)
            self.status_message.emit("图片保存成功。")  # 使用emit发送信号
            self.processing_finished.emit(self.output_image_path)
        except Exception as e:
            self.error_occurred.emit(f"保存图片失败：{e}")


class ImageUpscalerApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("哇哩唔嗷的工具箱 - 图像/视频增强")
        self.setAcceptDrops(True)
        self.input_file_path = ""  # 通用输入文件路径，可以是图片或视频
        self.output_file_path = ""  # 通用输出文件路径

        # 将模型路径、瓦片大小、重叠等参数改为动态获取
        self.current_model_key = list(MODEL_SETTINGS.keys())[0]  # 默认选择第一个模型
        self.current_model_settings = MODEL_SETTINGS[self.current_model_key]
        self.onnx_model_path = get_resource_path(self.current_model_settings["path"])
        self.tile_size = self.current_model_settings["tile_size"]
        self.overlap = self.current_model_settings["overlap"]
        self.upscale_factor = self.current_model_settings["default_upscale_factor"]

        self.scale_factor = 2.0  # 默认纯放缩比例
        self.current_mode = "Real-ESRGAN 图片超分辨率"  # 默认模式
        self.interpolation_method = "Lanczos插值 (Lanczos4)"  # 默认插值方法

        self.output_video_codec = "mp4v"  # 默认视频编码器
        self.output_video_fps = 0  # 默认继承原视频帧率，0表示自动

        self.processor_thread = None  # 声明线程变量

        self.init_ui()
        self.set_styles()  # 应用样式
        self.check_model_exists()  # 初始检查默认模型是否存在

    def init_ui(self):
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(25, 25, 25, 25)  # 增加主布局的边距
        main_layout.setSpacing(20)  # 增加控件之间的间距

        # --- 1. 功能选择 ---
        mode_select_layout = QHBoxLayout()
        mode_select_layout.addWidget(QLabel("<b>选择功能:</b>"))
        self.mode_combo = QComboBox()
        self.mode_combo.addItem("Real-ESRGAN 图片超分辨率")
        self.mode_combo.addItem("Real-ESRGAN 视频超分辨率")
        self.mode_combo.addItem("图片放缩")
        self.mode_combo.currentIndexChanged.connect(self.change_mode)
        mode_select_layout.addWidget(self.mode_combo)
        mode_select_layout.addStretch(1)
        main_layout.addLayout(mode_select_layout)

        # --- 2. 拖放区域 ---
        self.drop_label = QLabel("将图片/视频文件拖放到此处 或 点击 '选择文件'")
        self.drop_label.setAlignment(Qt.AlignCenter)
        self.drop_label.setFixedSize(QSize(600, 120))  # 调整拖放区域尺寸
        self.reset_drop_label_style()
        main_layout.addWidget(self.drop_label, alignment=Qt.AlignCenter)

        select_button = QPushButton("选择文件")
        select_button.clicked.connect(self.select_input_file)
        main_layout.addWidget(select_button)

        # --- 3. 输入输出路径 ---
        path_grid_layout = QGridLayout()
        path_grid_layout.setSpacing(12)  # 路径区域的间距

        path_grid_layout.addWidget(QLabel("输入路径:"), 0, 0)
        self.input_path_line = QLineEdit()
        self.input_path_line.setReadOnly(True)
        path_grid_layout.addWidget(self.input_path_line, 0, 1)
        path_grid_layout.setColumnStretch(1, 1)  # 让路径输入框所在的列占据更多空间

        path_grid_layout.addWidget(QLabel("输出路径:"), 1, 0)
        self.output_path_line = QLineEdit()
        self.output_path_line.textChanged.connect(self.update_output_path)
        path_grid_layout.addWidget(self.output_path_line, 1, 1)

        main_layout.addLayout(path_grid_layout)

        # --- 4. 模式特定设置 (QStackedWidget) ---
        self.stacked_widget = QStackedWidget()

        # Real-ESRGAN 图片/视频模式的专属控件 (共用 Real-ESRGAN 模型设置)
        # 将 Real-ESRGAN 设置内容放入一个 QWidget，然后用 QScrollArea 包含它
        esrgan_settings_content_widget = QWidget()
        esrgan_common_layout = QGridLayout(esrgan_settings_content_widget)
        esrgan_common_layout.setContentsMargins(20, 35, 20, 20)
        esrgan_common_layout.setSpacing(25)
        esrgan_common_layout.setColumnStretch(0, 1)
        esrgan_common_layout.setColumnStretch(1, 5)
        esrgan_common_layout.setColumnStretch(2, 2)

        # Re-apply the GroupBox visual style to the content widget
        esrgan_settings_content_widget.setObjectName("esrgan_settings_groupbox")  # Add object name for QSS

        esrgan_common_layout.addWidget(QLabel("选择 Real-ESRGAN 模型:"), 0, 0)
        self.model_selection_combo = QComboBox()
        for key, settings in MODEL_SETTINGS.items():
            self.model_selection_combo.addItem(settings["display_name"], userData=key)
        self.model_selection_combo.currentIndexChanged.connect(self.change_esrgan_model)
        self.model_selection_combo.setToolTip("选择预设的Real-ESRGAN模型。不同模型适用于不同图像类型。")
        esrgan_common_layout.addWidget(self.model_selection_combo, 0, 1, 1, 2)

        esrgan_common_layout.addWidget(QLabel("模型文件路径:"), 1, 0)
        self.model_path_line = QLineEdit(self.onnx_model_path)
        self.model_path_line.setReadOnly(True)
        self.model_path_line.setToolTip("当前使用的ONNX模型文件路径。")
        esrgan_common_layout.addWidget(self.model_path_line, 1, 1)
        model_browse_button = QPushButton("浏览模型")
        model_browse_button.clicked.connect(self.select_model_file)
        esrgan_common_layout.addWidget(model_browse_button, 1, 2)

        esrgan_common_layout.addWidget(QLabel("Real-ESRGAN 放缩比例:"), 2, 0)
        self.esrgan_upscale_factor_spinbox = QDoubleSpinBox()
        self.esrgan_upscale_factor_spinbox.setMinimum(0.5)
        self.esrgan_upscale_factor_spinbox.setMaximum(8.0)
        self.esrgan_upscale_factor_spinbox.setSingleStep(0.5)
        self.esrgan_upscale_factor_spinbox.setValue(self.upscale_factor)
        self.esrgan_upscale_factor_spinbox.valueChanged.connect(self.update_esrgan_upscale_factor)
        self.esrgan_upscale_factor_spinbox.setToolTip("图像将按此比例放大。Real-ESRGAN模型内置4倍放大，此项为额外放缩。")
        esrgan_common_layout.addWidget(self.esrgan_upscale_factor_spinbox, 2, 1)
        esrgan_common_layout.addWidget(QLabel("（原图尺寸 x 比例）"), 2, 2)

        esrgan_common_layout.addWidget(QLabel("瓦片大小:"), 3, 0)
        self.tile_size_label = QLabel(f"{self.tile_size}x{self.tile_size}")
        self.tile_size_label.setToolTip("用于处理大图像的瓦片尺寸，由所选模型决定。")
        esrgan_common_layout.addWidget(self.tile_size_label, 3, 1)
        esrgan_common_layout.addWidget(QLabel("重叠:"), 3, 2)
        self.overlap_label = QLabel(f"{self.overlap} 像素")
        self.overlap_label.setToolTip("瓦片之间重叠的像素数，用于消除拼接痕迹，由所选模型决定。")
        esrgan_common_layout.addWidget(self.overlap_label, 3, 3)

        # 视频专属设置 (作为 Real-ESRGAN common page 的一部分)
        self.video_settings_group = QGroupBox("视频输出设置")
        video_settings_layout = QGridLayout(self.video_settings_group)
        video_settings_layout.setContentsMargins(15, 25, 15, 15)
        video_settings_layout.setSpacing(15)

        video_settings_layout.addWidget(QLabel("输出视频编码器:"), 0, 0)
        self.video_codec_combo = QComboBox()
        self.video_codec_combo.addItem("mp4v (MP4)", userData="mp4v")
        self.video_codec_combo.addItem("XVID (AVI)", userData="XVID")
        self.video_codec_combo.addItem("MJPG (AVI)", userData="MJPG")
        self.video_codec_combo.currentTextChanged.connect(self.update_video_codec)
        self.video_codec_combo.setToolTip("选择输出视频的编码器。不同编码器兼容性、文件大小和质量有所不同。")
        video_settings_layout.addWidget(self.video_codec_combo, 0, 1)

        video_settings_layout.addWidget(QLabel("输出帧率 (FPS):"), 1, 0)
        self.video_fps_spinbox = QSpinBox()
        self.video_fps_spinbox.setMinimum(0)  # 0 表示使用原始视频帧率
        self.video_fps_spinbox.setMaximum(120)
        self.video_fps_spinbox.setValue(self.output_video_fps)
        self.video_fps_spinbox.setToolTip("设为0则使用原始视频帧率。")
        video_settings_layout.addWidget(self.video_fps_spinbox, 1, 1)
        video_settings_layout.addWidget(QLabel("(0=原视频帧率)"), 1, 2)

        esrgan_common_layout.addWidget(self.video_settings_group, 4, 0, 2, 4)

        # Create QScrollArea for Real-ESRGAN settings
        self.esrgan_scroll_area = QScrollArea()
        self.esrgan_scroll_area.setWidgetResizable(True)
        self.esrgan_scroll_area.setWidget(esrgan_settings_content_widget)
        self.esrgan_scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.esrgan_scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.esrgan_scroll_area.setStyleSheet("""
            QScrollArea {
                border: 1px solid #ced4da;
                border-radius: 10px;
                background-color: #ffffff;
            }
            QScrollArea QWidget#esrgan_settings_groupbox {
                font-weight: bold;
                border: none; /* No border for the inner widget, border is on QScrollArea */
                margin-top: 0; /* Remove top margin */
                padding: 0; /* Remove inner padding, layout will handle it */
                background-color: #ffffff;
            }
            QScrollArea QWidget#esrgan_settings_groupbox::title { /* QGroupBox::title is not applicable to QWidget */
                display: none; /* Hide if it somehow appears */
            }
        """)
        # Add a custom QLabel to act as the title for the Real-ESRGAN section above the scroll area
        self.esrgan_title_label = QLabel("Real-ESRGAN 模型设置")
        self.esrgan_title_label.setStyleSheet("""
            QLabel {
                font-weight: bold;
                background-color: #e9ecef;
                border: 1px solid #ced4da;
                border-radius: 7px;
                padding: 5px 15px;
                margin: 0;
                min-width: 150px;
                max-width: 200px;
                qproperty-alignment: AlignCenter;
            }
        """)
        esrgan_layout_with_title = QVBoxLayout()
        esrgan_layout_with_title.addWidget(self.esrgan_title_label, alignment=Qt.AlignCenter)
        esrgan_layout_with_title.addSpacing(10)  # Space between title and scroll area
        esrgan_layout_with_title.addWidget(self.esrgan_scroll_area)

        esrgan_page_container = QWidget()
        esrgan_page_container.setLayout(esrgan_layout_with_title)
        self.stacked_widget.addWidget(esrgan_page_container)  # 索引 0

        # 纯粹图片放缩模式的专属控件
        self.basic_scaler_page = QGroupBox("图片放缩设置")  # 使用 QGroupBox
        basic_scaler_layout = QGridLayout(self.basic_scaler_page)
        basic_scaler_layout.setContentsMargins(20, 35, 20, 20)  # 增加边距
        basic_scaler_layout.setSpacing(25)  # 增加间距
        basic_scaler_layout.setColumnStretch(0, 1)
        basic_scaler_layout.setColumnStretch(1, 3)  # 给输入框更多空间
        basic_scaler_layout.setColumnStretch(2, 1)

        basic_scaler_layout.addWidget(QLabel("放缩比例:"), 0, 0)
        self.basic_scale_factor_spinbox = QDoubleSpinBox()
        self.basic_scale_factor_spinbox.setMinimum(0.1)
        self.basic_scale_factor_spinbox.setMaximum(10.0)
        self.basic_scale_factor_spinbox.setSingleStep(0.1)
        self.basic_scale_factor_spinbox.setValue(self.scale_factor)
        self.basic_scale_factor_spinbox.valueChanged.connect(self.update_basic_scale_factor)
        self.basic_scale_factor_spinbox.setToolTip("图像将按此比例放大或缩小。")
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
        self.interpolation_combo.setToolTip("选择图片放缩的插值算法。Lanczos4通常提供最佳效果，Area适合缩小。")
        basic_scaler_layout.addWidget(self.interpolation_combo, 1, 1, 1, 2)  # 跨越两列

        self.stacked_widget.addWidget(self.basic_scaler_page)  # 索引 1

        main_layout.addWidget(self.stacked_widget)  # 堆叠控件放在主布局中

        # --- 5. 进度条和状态信息 ---
        self.process_button = QPushButton("开始处理")
        self.process_button.clicked.connect(self.start_processing)
        self.process_button.setEnabled(False)
        main_layout.addWidget(self.process_button)

        self.progress_bar = QProgressBar()
        self.progress_bar.setAlignment(Qt.AlignCenter)
        self.progress_bar.setMinimumHeight(20)  # 进度条高度
        main_layout.addWidget(self.progress_bar)

        self.status_text = QTextEdit()
        self.status_text.setReadOnly(True)
        self.status_text.setFixedHeight(140)  # 状态文本框高度
        self.status_text.setToolTip("显示处理过程中的状态和日志信息。")
        main_layout.addWidget(self.status_text)

        self.setLayout(main_layout)
        # 调整窗口大小，显著增加宽度，以确保所有长文本能完全显示，高度适度
        # 考虑到滚动区域，高度可以稍微放宽，但我们让滚动区域自己处理
        self.setFixedSize(QSize(980, 1000))  # 进一步增加宽度和高度，以容纳更多空间

        # 初始显示 Real-ESRGAN 页面
        self.stacked_widget.setCurrentIndex(0)
        self.change_mode(0)  # 确保初始状态正确

    def set_styles(self):
        """设置应用程序的整体样式"""
        self.setStyleSheet("""
            QWidget {
                background-color: #f8f9fa; /* 浅灰色背景 */
                font-family: 'Segoe UI', 'Microsoft YaHei', sans-serif;
                font-size: 11pt; /* 默认字体大小 */
                color: #343a40; /* 深灰色文本 */
            }

            QGroupBox {
                font-weight: bold;
                border: 1px solid #ced4da; /* 浅边框 */
                border-radius: 10px;
                margin-top: 2ex; /* 标题与边框的距离 */
                padding: 15px; /* 内部填充 */
                background-color: #ffffff; /* 白色背景 */
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top center; /* 标题居中 */
                padding: 0 10px;
                background-color: #e9ecef; /* 标题背景色 */
                border-radius: 7px;
            }

            /* Style for the QWidget inside QScrollArea to mimic GroupBox */
            QWidget#esrgan_settings_groupbox {
                font-weight: bold;
                border: none; /* ScrollArea will have the border */
                border-radius: 10px;
                margin: 0;
                padding: 0; /* Layout inside will handle padding */
                background-color: #ffffff;
            }


            /* --- 针对模型设置区域的控件（QGroupBox内的元素）特别放大 --- */
            QGroupBox QLabel, 
            QGroupBox QLineEdit, 
            QGroupBox QComboBox, 
            QGroupBox QDoubleSpinBox, 
            QGroupBox QSpinBox,
            QWidget#esrgan_settings_groupbox QLabel,
            QWidget#esrgan_settings_groupbox QLineEdit,
            QWidget#esrgan_settings_groupbox QComboBox,
            QWidget#esrgan_settings_groupbox QDoubleSpinBox,
            QWidget#esrgan_settings_groupbox QSpinBox
            {
                font-size: 12.5pt; /* 进一步加大模型设置区域的字体 */
            }
            QGroupBox QLineEdit, 
            QGroupBox QComboBox, 
            QGroupBox QDoubleSpinBox, 
            QGroupBox QSpinBox,
            QWidget#esrgan_settings_groupbox QLineEdit,
            QWidget#esrgan_settings_groupbox QComboBox,
            QWidget#esrgan_settings_groupbox QDoubleSpinBox,
            QWidget#esrgan_settings_groupbox QSpinBox
            {
                padding: 8px; /* 增加内边距，使输入框更高 */
                min-height: 32px; /* 确保高度足以容纳大字体 */
            }


            QLabel {
                color: #495057; /* 标签文本颜色 */
            }

            /* --- 非QGroupBox内的控件，保持适中大小 --- */
            QLineEdit, QComboBox, QDoubleSpinBox, QSpinBox {
                border: 1px solid #ced4da;
                border-radius: 6px;
                padding: 5px; /* 适中内边距 */
                background-color: #ffffff;
                selection-background-color: #007bff;
                selection-color: white;
            }
            /* 确保输入输出路径框有足够的最小宽度 */
            QLineEdit#input_path_line, QLineEdit#output_path_line {
                min-width: 650px; /* 再次增加最小宽度，确保长路径显示 */
            }


            QComboBox::drop-down {
                subcontrol-origin: padding;
                subcontrol-position: top right;
                width: 25px;
                border-left-width: 1px;
                border-left-color: #ced4da;
                border-left-style: solid;
                border-top-right-radius: 5px;
                border-bottom-right-radius: 5px;
            }
            QComboBox::down-arrow {
                image: url(data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAYAAAAf8/9hAAAABmJLR0QA/wD/AP+gvaeTAAAAVElEQVQ4jWNgGAWjYBSMGgAIAAD//wEZgYkBgO0C/P8fGKgY/P+DgaYn0OQZAPGgYJADvNMgqQEYjC+B4o/jE/9iB4T/D/2/gfB/g/+/gXQYAEQ4AH0H3GjJAAAAAElFTkSuQmCC);
                width: 16px;
                height: 16px;
            }

            QPushButton {
                background-color: #007bff;
                color: white;
                border-radius: 8px; /* 按钮圆角适中 */
                padding: 8px 18px; /* 按钮内边距适中 */
                font-weight: bold;
                border: none;
            }
            QPushButton:hover {
                background-color: #0056b3;
            }
            QPushButton:pressed {
                background-color: #004085;
            }
            QPushButton:disabled {
                background-color: #a0c4ff;
                color: #e0e0e0;
            }

            QProgressBar {
                border: 1px solid #ced4da;
                border-radius: 8px;
                text-align: center;
                color: #343a40;
                background-color: #e9ecef;
                min-height: 20px; /* 明确设置最小高度 */
            }
            QProgressBar::chunk {
                background-color: #28a745;
                border-radius: 8px;
            }

            QTextEdit {
                border: 1px solid #ced4da;
                border-radius: 8px;
                padding: 8px; /* 适中内边距 */
                background-color: #fdfdfe;
                color: #343a40;
            }

            /* ScrollArea styles */
            QScrollArea {
                border: 1px solid #ced4da;
                border-radius: 10px;
                background-color: #ffffff;
            }
            QScrollBar:vertical {
                border: none;
                background: #e9ecef;
                width: 10px;
                margin: 0px 0px 0px 0px;
                border-radius: 5px;
            }
            QScrollBar::handle:vertical {
                background: #a8dadc;
                min-height: 20px;
                border-radius: 5px;
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                border: none;
                background: none;
            }
            QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {
                background: none;
            }
            QScrollBar:horizontal {
                border: none;
                background: #e9ecef;
                height: 10px;
                margin: 0px 0px 0px 0px;
                border-radius: 5px;
            }
            QScrollBar::handle:horizontal {
                background: #a8dadc;
                min-width: 20px;
                border-radius: 5px;
            }
            QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {
                border: none;
                background: none;
            }
            QScrollBar::add-page:horizontal, QScrollBar::sub-page:horizontal {
                background: none;
            }
        """)
        # 为 input_path_line 和 output_path_line 分配 objectName，以便在QSS中精确控制
        self.input_path_line.setObjectName("input_path_line")
        self.output_path_line.setObjectName("output_path_line")

    def reset_drop_label_style(self):
        """重置拖放区域的默认样式"""
        self.drop_label.setStyleSheet(
            "QLabel { "
            "   border: 2px dashed #a8dadc;"
            "   border-radius: 10px;"
            "   background-color: #f1faee;"
            "   font-size: 13pt;"  # 拖放区域字体保持适中
            "   color: #457b9d;"
            "   font-weight: bold;"
            "}"
            "QLabel:hover { "
            "   border: 2px dashed #457b9d;"
            "   background-color: #e0f2f7;"
            "}"
        )

    def change_mode(self, index):
        """根据选择切换显示不同的控件"""
        self.current_mode = self.mode_combo.currentText()

        # ESRGAN 图片和视频模式共用第一个堆叠页面 (索引0)
        if self.current_mode == "Real-ESRGAN 图片超分辨率" or self.current_mode == "Real-ESRGAN 视频超分辨率":
            self.stacked_widget.setCurrentIndex(0)  # ESRGAN page
            self.esrgan_title_label.setVisible(True)  # Show the custom title
            self.model_selection_combo.setEnabled(True)
            self.esrgan_upscale_factor_spinbox.setEnabled(True)

            self.basic_scale_factor_spinbox.setEnabled(False)
            self.interpolation_combo.setEnabled(False)

            self.tile_size_label.setText(f"{self.tile_size}x{self.tile_size}")
            self.overlap_label.setText(f"{self.overlap} 像素")

            # 视频专属设置的启用/禁用
            if self.current_mode == "Real-ESRGAN 视频超分辨率":
                self.video_settings_group.setVisible(True)
                self.video_codec_combo.setEnabled(True)
                self.video_fps_spinbox.setEnabled(True)
                self.status_text.append(
                    "<span style='color: #d6801b;'>提示: 视频超分辨率功能不处理音频，输出视频将是静音的。</span>")
            else:
                self.video_settings_group.setVisible(False)
                self.video_codec_combo.setEnabled(False)
                self.video_fps_spinbox.setEnabled(False)


        elif self.current_mode == "图片放缩":
            self.stacked_widget.setCurrentIndex(1)  # Basic Scaler page
            self.esrgan_title_label.setVisible(False)  # Hide the custom title
            self.model_selection_combo.setEnabled(False)
            self.model_path_line.setEnabled(False)  # 模型路径在图片放缩模式下始终禁用
            self.esrgan_upscale_factor_spinbox.setEnabled(False)

            self.basic_scale_factor_spinbox.setEnabled(True)
            self.interpolation_combo.setEnabled(True)

            self.tile_size_label.setText("")
            self.overlap_label.setText("")
            self.video_settings_group.setVisible(False)
            self.video_codec_combo.setEnabled(False)
            self.video_fps_spinbox.setEnabled(False)

        self.generate_output_path(self.input_file_path)  # 切换模式后更新输出文件名
        self.check_model_exists()  # 重新检查模型是否存在 (因为Real-ESRGAN模式依赖它)
        self.enable_process_button_if_ready()

    def change_esrgan_model(self, index):
        """当 Real-ESRGAN 模型选择改变时"""
        self.current_model_key = self.model_selection_combo.itemData(index)
        self.current_model_settings = MODEL_SETTINGS[self.current_model_key]
        self.onnx_model_path = get_resource_path(self.current_model_settings["path"])
        self.tile_size = self.current_model_settings["tile_size"]
        self.overlap = self.current_model_settings["overlap"]
        self.upscale_factor = self.current_model_settings["default_upscale_factor"]  # 自动调整放缩比例

        self.model_path_line.setText(self.onnx_model_path)
        self.esrgan_upscale_factor_spinbox.setValue(self.upscale_factor)
        self.tile_size_label.setText(f"{self.tile_size}x{self.tile_size}")
        self.overlap_label.setText(f"{self.overlap} 像素")

        self.check_model_exists()
        self.generate_output_path(self.input_file_path)

    def check_model_exists(self):
        if self.current_mode.startswith("Real-ESRGAN"):  # 无论是图片还是视频模式都依赖模型
            if not os.path.exists(self.onnx_model_path):
                # 调整样式以匹配QSS，直接使用QSS的背景色和边框色
                self.model_path_line.setStyleSheet("background-color: #ffe0e0; border: 1px solid #dc3545;")
                self.status_text.append(
                    f"<span style='color: #dc3545;'>警告: Real-ESRGAN模型文件缺失: {self.onnx_model_path}</span>")
                self.model_selection_combo.setEnabled(True)
                self.esrgan_upscale_factor_spinbox.setEnabled(False)
                self.model_path_line.setReadOnly(False)
                self.model_path_line.setToolTip("模型文件不存在，请点击浏览选择或检查路径。")
            else:
                # 恢复默认样式
                self.model_path_line.setStyleSheet("")
                self.status_text.append(
                    f"<span style='color: #28a745;'>模型文件存在: {self.onnx_model_path}</span>")
                self.model_selection_combo.setEnabled(True)
                self.esrgan_upscale_factor_spinbox.setEnabled(True)
                self.model_path_line.setReadOnly(True)
                self.model_path_line.setToolTip("当前使用的ONNX模型文件路径。")

        else:  # 纯粹图片放缩模式下，模型路径无所谓
            self.model_path_line.setStyleSheet("")
            self.model_path_line.setEnabled(False)
            self.model_path_line.setToolTip("")

        self.enable_process_button_if_ready()

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
            self.drop_label.setStyleSheet(
                "QLabel { "
                "   border: 3px solid #457b9d;"  # 实线边框
                "   border-radius: 10px;"
                "   background-color: #cce7ee;"  # 更亮的背景色
                "   font-size: 13pt;"
                "   color: #2a5568;"
                "   font-weight: bold;"
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
            # 统一处理图片和视频文件
            if os.path.isfile(file_path) and (
                    file_path.lower().endswith(
                        ('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.mp4', '.avi', '.mov', '.mkv'))):
                self.set_input_file_path(file_path)
                break
        event.acceptProposedAction()

    def select_input_file(self):
        options = QFileDialog.Options()
        # 根据当前模式调整文件过滤器
        if self.current_mode == "Real-ESRGAN 图片超分辨率" or self.current_mode == "图片放缩":
            file_filter = "图片文件 (*.png *.jpg *.jpeg *.bmp *.tiff);;所有文件 (*)"
        else:  # 视频超分辨率
            file_filter = "视频文件 (*.mp4 *.avi *.mov *.mkv);;所有文件 (*)"

        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择输入文件", "",
            file_filter, options=options
        )
        if file_path:
            self.set_input_file_path(file_path)

    def select_model_file(self):
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择ONNX模型文件", "",
            "ONNX模型文件 (*.onnx);;所有文件 (*)", options=options
        )
        if file_path:
            # 检查是否是内置模型
            found_built_in = False
            for i in range(self.model_selection_combo.count()):
                key = self.model_selection_combo.itemData(i)
                # 检查此文件路径是否与任何内置模型的绝对路径匹配
                built_in_abs_path = get_resource_path(MODEL_SETTINGS.get(key, {}).get("path", ""))
                if os.path.normpath(built_in_abs_path) == os.path.normpath(file_path):
                    self.model_selection_combo.setCurrentIndex(i)
                    found_built_in = True
                    break

            if not found_built_in:
                # 检查是否已经是自定义模型
                is_custom_model_already_in_list = False
                existing_custom_key = None
                for key, settings in MODEL_SETTINGS.items():
                    if settings.get("path") == file_path and settings.get("display_name").startswith("自定义模型"):
                        existing_custom_key = key
                        is_custom_model_already_in_list = True
                        break

                if not is_custom_model_already_in_list:
                    reply = QMessageBox.question(self, "发现新模型",
                                                 f"检测到新的ONNX模型: {os.path.basename(file_path)}\n"
                                                 "是否将其添加到模型列表中？(瓦片大小和重叠将使用默认值128/10，动漫/512x512类模型默认512/32)",
                                                 QMessageBox.Yes | QMessageBox.No)
                    if reply == QMessageBox.Yes:
                        new_model_name = os.path.basename(file_path)
                        # 生成一个唯一的key，避免与现有内置模型冲突
                        new_key_base = f"custom_{new_model_name.replace('.', '_')}"
                        new_key = new_key_base
                        counter = 1
                        while new_key in MODEL_SETTINGS:
                            new_key = f"{new_key_base}_{counter}"
                            counter += 1

                        default_tile = 128
                        default_overlap = 10
                        if "512x512" in new_model_name.lower() or "anime" in new_model_name.lower():
                            default_tile = 512
                            default_overlap = 32

                        MODEL_SETTINGS[new_key] = {
                            "path": file_path,
                            "tile_size": default_tile,
                            "overlap": default_overlap,
                            "default_upscale_factor": 4.0,
                            "display_name": f"自定义模型 ({new_model_name})",
                            "description": "用户自定义模型"
                        }
                        self.model_selection_combo.addItem(MODEL_SETTINGS[new_key]["display_name"], userData=new_key)
                        self.model_selection_combo.setCurrentIndex(self.model_selection_combo.count() - 1)
                    else:
                        pass  # 什么也不做，保持当前模型设置
                else:  # 如果已经是自定义模型，则只更新路径显示并切换到它
                    self.onnx_model_path = file_path  # 确保path是最新的
                    self.model_path_line.setText(file_path)
                    idx = self.model_selection_combo.findData(existing_custom_key)
                    if idx != -1:
                        self.model_selection_combo.setCurrentIndex(idx)
                    self.check_model_exists()  # 重新检查以更新状态

    def update_esrgan_upscale_factor(self, value):
        self.upscale_factor = value
        self.generate_output_path(self.input_file_path)

    def update_basic_scale_factor(self, value):
        self.scale_factor = value
        self.generate_output_path(self.input_file_path)

    def update_interpolation_method(self, text):
        self.interpolation_method = text
        self.generate_output_path(self.input_file_path)

    def update_video_codec(self, text):  # 这里的参数是text，因为信号是currentTextChanged
        # 找到与文本对应的userData
        index = self.video_codec_combo.findText(text)
        if index != -1:
            self.output_video_codec = self.video_codec_combo.itemData(index)
        self.generate_output_path(self.input_file_path)  # 改变编码器也可能改变文件后缀

    def update_video_fps(self, value):
        self.output_video_fps = value
        self.generate_output_path(self.input_file_path)

    def set_input_file_path(self, path):
        self.input_file_path = path
        self.input_path_line.setText(path)
        self.generate_output_path(path)
        self.enable_process_button_if_ready()

    def generate_output_path(self, input_path):
        if input_path:
            dir_name = os.path.dirname(input_path)
            file_name_without_ext = os.path.splitext(os.path.basename(input_path))[0]

            if self.current_mode.startswith("Real-ESRGAN"):
                model_abbr = self.current_model_settings["display_name"].replace("Real-ESRGAN", "ESRGAN").replace(" ",
                                                                                                                  "").replace(
                    "(通用)", "").replace("(动漫专用)", "").replace("自定义模型", "Custom")

                if self.current_mode == "Real-ESRGAN 图片超分辨率":
                    output_file_name = f"{file_name_without_ext}_{model_abbr}_x{self.upscale_factor:.1f}.png"
                else:  # Real-ESRGAN 视频超分辨率
                    # 默认输出为 .mp4，根据编码器调整
                    output_ext = ".mp4"
                    if self.output_video_codec == "XVID" or self.output_video_codec == "MJPG":
                        output_ext = ".avi"
                    output_file_name = f"{file_name_without_ext}_{model_abbr}_x{self.upscale_factor:.1f}{output_ext}"
            else:  # 纯粹图片放缩
                interp_abbr = self.interpolation_method.split(' ')[0].replace('插值', '')
                output_file_name = f"{file_name_without_ext}_scaled_x{self.scale_factor:.1f}_{interp_abbr}.png"

            self.output_file_path = os.path.join(dir_name, output_file_name)
            self.output_path_line.setText(self.output_file_path)
        else:
            self.output_file_path = ""
            self.output_path_line.setText("")

    def update_output_path(self, text):
        self.output_file_path = text
        self.enable_process_button_if_ready()

    def enable_process_button_if_ready(self):
        # 清除之前的警告信息，只保留最新相关警告
        current_html = self.status_text.toHtml()
        current_html = current_html.replace(
            "<span style='color: orange;'>警告: 视频超分辨率模式需要视频文件作为输入。</span>", "")
        current_html = current_html.replace(
            "<span style='color: orange;'>警告: 图片超分辨率模式需要图片文件作为输入。</span>", "")
        current_html = current_html.replace(
            "<span style='color: orange;'>警告: 图片放缩模式需要图片文件作为输入。</span>", "")
        self.status_text.setHtml(current_html)
        self.status_text.verticalScrollBar().setValue(self.status_text.verticalScrollBar().maximum())

        is_ready = bool(self.input_file_path) and bool(self.output_file_path) and not (
                    self.processor_thread and self.processor_thread.isRunning())

        if self.current_mode.startswith("Real-ESRGAN"):
            is_ready = is_ready and os.path.exists(self.onnx_model_path) and self.upscale_factor > 0

            if not os.path.exists(self.onnx_model_path):
                self.process_button.setEnabled(False)
                return

            if self.current_mode == "Real-ESRGAN 视频超分辨率":
                if self.input_file_path and not self.input_file_path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                    is_ready = False
                    self.status_text.append(
                        "<span style='color: orange;'>警告: 视频超分辨率模式需要视频文件作为输入。</span>")
            else:  # Real-ESRGAN 图片超分辨率
                if self.input_file_path and not self.input_file_path.lower().endswith(
                        ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                    is_ready = False
                    self.status_text.append(
                        "<span style='color: orange;'>警告: 图片超分辨率模式需要图片文件作为输入。</span>")
        else:  # 纯粹图片放缩
            is_ready = is_ready and self.scale_factor > 0
            if self.input_file_path and not self.input_file_path.lower().endswith(
                    ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                is_ready = False
                self.status_text.append("<span style='color: orange;'>警告: 图片放缩模式需要图片文件作为输入。</span>")

        self.process_button.setEnabled(is_ready)

    def start_processing(self):
        if not self.input_file_path:
            QMessageBox.warning(self, "警告", "请先选择一个输入文件！")
            return
        if not self.output_file_path:
            QMessageBox.warning(self, "警告", "输出路径不能为空！")
            return

        # 禁用所有输入控件
        self.process_button.setEnabled(False)
        self.input_path_line.setEnabled(False)
        self.output_path_line.setEnabled(False)
        self.mode_combo.setEnabled(False)
        self.esrgan_upscale_factor_spinbox.setEnabled(False)
        self.basic_scale_factor_spinbox.setEnabled(False)
        self.interpolation_combo.setEnabled(False)
        self.model_selection_combo.setEnabled(False)
        self.model_path_line.setEnabled(False)  # 处理开始后禁用
        self.video_codec_combo.setEnabled(False)
        self.video_fps_spinbox.setEnabled(False)
        self.esrgan_scroll_area.setEnabled(False)  # Disable the scroll area too
        self.basic_scaler_page.setEnabled(False)  # Disable the basic scaler group box

        self.progress_bar.setValue(0)
        self.status_text.clear()
        self.status_text.append("开始处理...")

        if self.current_mode == "Real-ESRGAN 图片超分辨率":
            if not os.path.exists(self.onnx_model_path):
                QMessageBox.critical(self, "错误", "Real-ESRGAN ONNX模型文件不存在，请检查路径！")
                self.reset_ui()
                return
            self.processor_thread = RealESRGANProcessor(
                self.input_file_path, self.output_file_path, self.onnx_model_path,
                self.upscale_factor, self.tile_size, self.overlap
            )
        elif self.current_mode == "Real-ESRGAN 视频超分辨率":
            if not os.path.exists(self.onnx_model_path):
                QMessageBox.critical(self, "错误", "Real-ESRGAN ONNX模型文件不存在，请检查路径！")
                self.reset_ui()
                return
            self.processor_thread = RealESRGANVideoProcessor(
                self.input_file_path, self.output_file_path, self.onnx_model_path,
                self.upscale_factor, self.tile_size, self.overlap,
                self.output_video_codec, self.output_video_fps
            )
        else:  # 纯粹图片放缩
            self.processor_thread = BasicImageScaler(
                self.input_file_path, self.output_file_path, self.scale_factor, self.interpolation_method
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
        else:  # 对于BasicImageScaler这种没有帧进度的，可以直接显示0或100
            self.progress_bar.setValue(0 if current == 0 else 100)
        self.status_text.verticalScrollBar().setValue(self.status_text.verticalScrollBar().maximum())

    def update_status(self, message):
        self.status_text.append(message)
        self.status_text.verticalScrollBar().setValue(self.status_text.verticalScrollBar().maximum())

    def on_processing_finished(self, output_path):
        self.status_text.append(f"处理完成！输出文件已保存到：{output_path}")
        QMessageBox.information(self, "完成", f"文件处理成功！\n输出文件：{output_path}", QMessageBox.Ok)
        self.cleanup_thread()
        self.reset_ui()

    def on_error(self, message):
        self.status_text.append(f"错误：{message}")
        QMessageBox.critical(self, "错误", f"处理过程中发生错误：\n{message}", QMessageBox.Ok)
        self.cleanup_thread()
        self.reset_ui()

    def cleanup_thread(self):
        if hasattr(self, 'processor_thread') and self.processor_thread and self.processor_thread.isRunning():
            self.processor_thread.requestInterruption()  # 请求线程中断
            self.processor_thread.quit()  # 停止事件循环
            self.processor_thread.wait(5000)  # 等待线程结束，最多5秒
            if self.processor_thread.isRunning():  # 如果仍未停止，强制终止 (不推荐，但作为最后手段)
                self.processor_thread.terminate()
                self.processor_thread.wait()  # 再次等待终止
            self.processor_thread = None

    def reset_ui(self):
        # 重新启用基础控件
        self.input_path_line.setEnabled(True)
        self.output_path_line.setEnabled(True)
        self.mode_combo.setEnabled(True)

        # 重新启用滚动区域和基本缩放器组框
        self.esrgan_scroll_area.setEnabled(True)
        self.basic_scaler_page.setEnabled(True)

        # 根据当前选择的模式重新启用相关控件
        self.change_mode(self.mode_combo.currentIndex())

        self.input_file_path = ""
        self.output_file_path = ""
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

# pyinstaller --windowed --onefile --icon=mynewico.ico --add-data "real_esrgan_x4plus-real-esrgan-x4plus-float.onnx;." --add-data "RealESRGAN_ANIME_6B_512x512.onnx;." --upx-dir="d:/Users/12766/Desktop/upx-5.0.1-win64" WaliTool.py
