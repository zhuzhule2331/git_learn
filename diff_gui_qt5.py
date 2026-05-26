#!/usr/bin/env python3
"""
简单的 Qt5 GUI，用于控制 diff_analyzer.py 中的采集与离线分析功能。

依赖:
  pip install PyQt5

运行:
  .venv/bin/python diff_gui_qt5.py

此程序会在后台线程中调用 diff_analyzer 中的函数，显示日志和图片预览。
"""
import sys
import os
import traceback
from types import SimpleNamespace

SYSTEM_DIST_PACKAGES = '/usr/lib/python3/dist-packages'
if os.path.isdir(SYSTEM_DIST_PACKAGES) and SYSTEM_DIST_PACKAGES not in sys.path:
    sys.path.insert(0, SYSTEM_DIST_PACKAGES)

from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (
    QApplication,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTextEdit,
    QFileDialog,
    QSpinBox,
    QDoubleSpinBox,
    QCheckBox,
    QProgressBar,
    QLineEdit,
)

import cv2
import numpy as np

import diff_analyzer as da
from MvImport.MvCameraControl_class import MvCamera
from pathlib import Path


class WorkerThread(QThread):
    log_signal = pyqtSignal(str)
    progress_signal = pyqtSignal(int)
    image_signal = pyqtSignal(object)
    finished_signal = pyqtSignal(str)
    status_signal = pyqtSignal(str)
    camera_info_signal = pyqtSignal(str)

    def __init__(self, mode, args):
        super().__init__()
        self.mode = mode
        self.args = args
        self._stop_requested = False

    def request_stop(self):
        self._stop_requested = True
    def run(self):
        try:
            self.log_signal.emit('初始化 SDK...')
            MvCamera.MV_CC_Initialize()
            self.log_signal.emit('SDK 初始化完成')

            if self.mode == 'capture':
                self._run_capture()
            elif self.mode == 'analyze':
                self._run_analyze()
            else:
                self.log_signal.emit(f'未知模式: {self.mode}')

            self.finished_signal.emit('完成')
        except Exception as e:
            tb = traceback.format_exc()
            self.log_signal.emit(f'错误: {e}\n{tb}')
            self.finished_signal.emit('异常结束')
        finally:
            try:
                MvCamera.MV_CC_Finalize()
                self.log_signal.emit('SDK 已释放 (Finalize)')
            except Exception as e:
                self.log_signal.emit(f'释放 SDK 失败: {e}')

    def _run_capture(self):
        self.log_signal.emit('开始枚举设备...')
        try:
            device_list = da.enumerate_devices()
            self.log_signal.emit(f'发现设备数: {device_list.nDeviceNum}')
        except Exception as e:
            self.log_signal.emit(f'枚举设备失败: {e}')
            return

        try:
            camera, device_info = da.open_camera(device_list, int(self.args.device_index))
        except Exception as e:
            self.log_signal.emit(f'打开相机失败: {e}')
            return

        try:
            da.apply_stable_camera_params(camera, self.args)
            da.configure_streaming(camera)

            try:
                snapshot = da.collect_camera_snapshot(camera)
                self.camera_info_signal.emit(da.format_camera_snapshot(snapshot))
                self.log_signal.emit('已读取相机参数快照')
            except Exception as e:
                self.log_signal.emit(f'读取相机参数失败: {e}')

            ret = camera.MV_CC_StartGrabbing()
            if ret != 0:
                self.log_signal.emit(f'开始取流失败, ret=0x{da.to_hex_str(ret)}')
                return

            # 丢弃预热帧
            for i in range(int(self.args.discard_frames)):
                if self._stop_requested:
                    self.log_signal.emit('已取消: 预热帧阶段')
                    return
                try:
                    da.grab_one_frame(camera, int(self.args.timeout_ms))
                    self.log_signal.emit(f'已丢弃预热帧 {i+1}/{self.args.discard_frames}')
                except Exception:
                    self.log_signal.emit(f'丢弃预热帧 {i+1} 失败')

            saved_paths = []
            output_dir = Path(self.args.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

            # 逐帧采集、保存并发预览
            for idx in range(int(self.args.count)):
                if self._stop_requested:
                    self.log_signal.emit('已取消: 采集过程')
                    break

                try:
                    frame_info, image = da.grab_one_frame(camera, int(self.args.timeout_ms))
                except Exception as e:
                    self.log_signal.emit(f'取图失败: {e}')
                    continue

                save_path = output_dir / f'frame_{idx:04d}.png'
                try:
                    da.save_frame(image, save_path)
                except Exception as e:
                    self.log_signal.emit(f'保存帧失败: {e}')
                    continue

                saved_paths.append(save_path)

                # 发送预览和进度
                try:
                    self.image_signal.emit(image)
                except Exception:
                    pass

                percent = int((idx + 1) / max(1, int(self.args.count)) * 100)
                self.progress_signal.emit(percent)
                try:
                    self.status_signal.emit(f'帧 {idx+1}/{self.args.count} 保存: {save_path.name} {int(frame_info.nWidth)}x{int(frame_info.nHeight)}')
                except Exception:
                    pass
                self.log_signal.emit(f'保存帧 {idx+1}/{self.args.count}: {save_path.name} {int(frame_info.nWidth)}x{int(frame_info.nHeight)}')

            # 分析并写报告（如果有保存的图片）
            try:
                if saved_paths:
                    results = da.analyze_image_sequence(saved_paths)
                    source_name = f'camera_index={self.args.device_index}, device_type={device_info.nTLayerType}'
                    da.write_log(results, self.args.output_report, source_name)
                    self.log_signal.emit('采集与分析完成')
                    try:
                        self.status_signal.emit(f'报告: {self.args.output_report}')
                    except Exception:
                        pass
                else:
                    self.log_signal.emit('未保存任何图片，跳过分析')
            except Exception as e:
                self.log_signal.emit(f'分析或写报告失败: {e}')
        finally:
            try:
                camera.MV_CC_StopGrabbing()
            except Exception:
                pass
            try:
                camera.MV_CC_CloseDevice()
            except Exception:
                pass
            try:
                camera.MV_CC_DestroyHandle()
            except Exception:
                pass

    def _run_analyze(self):
        try:
            self.log_signal.emit(f'开始离线分析: {self.args.input_dir}')
            da.analyze_existing_images(self.args)
            self.log_signal.emit('离线分析完成')
        except Exception as e:
            self.log_signal.emit(f'离线分析失败: {e}')


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('相机帧差异分析 GUI')
        self.resize(900, 600)
        self.resize(1200, 820)

        main_layout = QHBoxLayout(self)
        left = QVBoxLayout()
        right = QVBoxLayout()

        self.scan_btn = QPushButton('扫描设备')
        self.scan_btn.clicked.connect(self.scan_devices)

        self.device_index_spin = QSpinBox()
        self.device_index_spin.setMinimum(0)
        self.device_index_spin.setMaximum(16)
        self.device_index_spin.setValue(0)

        self.count_spin = QSpinBox()
        self.count_spin.setMinimum(1)
        self.count_spin.setMaximum(10000)
        self.count_spin.setValue(20)

        self.discard_spin = QSpinBox()
        self.discard_spin.setMinimum(0)
        self.discard_spin.setMaximum(100)
        self.discard_spin.setValue(5)

        self.timeout_spin = QSpinBox()
        self.timeout_spin.setMinimum(100)
        self.timeout_spin.setMaximum(60000)
        self.timeout_spin.setValue(2000)

        self.exposure_box = QDoubleSpinBox()
        self.exposure_box.setMinimum(0.0)
        self.exposure_box.setMaximum(1e7)
        self.exposure_box.setValue(10000.0)
        self.exposure_box.setSingleStep(1.0)
        self.exposure_box.setToolTip('默认 10000us')

        self.gain_box = QDoubleSpinBox()
        self.gain_box.setMinimum(0.0)
        self.gain_box.setMaximum(1e4)
        self.gain_box.setValue(20.0)
        self.gain_box.setSingleStep(0.1)
        self.gain_box.setToolTip('默认 20')

        self.frame_rate_box = QDoubleSpinBox()
        self.frame_rate_box.setMinimum(0.0)
        self.frame_rate_box.setMaximum(10000.0)
        self.frame_rate_box.setValue(0.0)
        self.frame_rate_box.setToolTip('自动模式下由程序读取相机最大帧率')

        self.auto_frame_rate_check = QCheckBox('自动使用相机最大帧率')
        self.auto_frame_rate_check.setChecked(True)
        self.auto_frame_rate_check.toggled.connect(lambda checked: self.frame_rate_box.setEnabled(not checked))
        self.frame_rate_box.setEnabled(False)

        self.trigger_check = QCheckBox('开启软件触发')

        self.output_dir_edit = QLineEdit('./capture_data')
        self.output_dir_btn = QPushButton('选择输出目录')
        self.output_dir_btn.clicked.connect(self.select_output_dir)

        self.output_report_edit = QLineEdit('./difference.txt')
        self.output_report_btn = QPushButton('选择报告文件')
        self.output_report_btn.clicked.connect(self.select_report_file)

        self.input_dir_edit = QLineEdit('')
        self.input_dir_btn = QPushButton('选择已拍图片文件夹(离线分析)')
        self.input_dir_btn.clicked.connect(self.select_input_dir)

        self.start_btn = QPushButton('开始采集并分析')
        self.start_btn.clicked.connect(self.start_capture)

        self.analyze_btn = QPushButton('仅离线分析')
        self.analyze_btn.clicked.connect(self.start_analyze)
        self.stop_btn = QPushButton('取消当前任务')
        self.stop_btn.clicked.connect(self.request_stop)

        self.status_label = QLabel('就绪')

        self.camera_info_view = QTextEdit()
        self.camera_info_view.setReadOnly(True)
        self.camera_info_view.setMinimumHeight(220)
        self.camera_info_view.setPlaceholderText('相机参数快照将在这里显示')

        self.open_report_btn = QPushButton('打开报告')
        self.open_report_btn.clicked.connect(self.open_report)

        left.addWidget(QLabel('设备索引'))
        left.addWidget(self.device_index_spin)
        left.addWidget(self.scan_btn)
        left.addWidget(QLabel('采集张数'))
        left.addWidget(self.count_spin)
        left.addWidget(QLabel('预热丢帧数'))
        left.addWidget(self.discard_spin)
        left.addWidget(QLabel('超时 (ms)'))
        left.addWidget(self.timeout_spin)
        left.addWidget(QLabel('曝光 (ExposureTime)'))
        left.addWidget(self.exposure_box)
        left.addWidget(QLabel('增益 (Gain)'))
        left.addWidget(self.gain_box)
        left.addWidget(QLabel('帧率 (AcquisitionFrameRate)'))
        left.addWidget(self.frame_rate_box)
        left.addWidget(self.auto_frame_rate_check)
        left.addWidget(self.trigger_check)
        left.addWidget(QLabel('输出目录'))
        left.addWidget(self.output_dir_edit)
        left.addWidget(self.output_dir_btn)
        left.addWidget(QLabel('输出报告'))
        left.addWidget(self.output_report_edit)
        left.addWidget(self.output_report_btn)
        left.addWidget(QLabel('离线图片目录'))
        left.addWidget(self.input_dir_edit)
        left.addWidget(self.input_dir_btn)
        left.addWidget(self.start_btn)
        left.addWidget(self.analyze_btn)
        left.addWidget(self.stop_btn)
        left.addWidget(QLabel('相机参数快照'))
        left.addWidget(self.camera_info_view)
        left.addWidget(self.status_label)
        left.addWidget(self.open_report_btn)

        self.preview = QLabel('预览')
        self.preview.setFixedSize(560, 420)
        self.preview.setStyleSheet('background: #222; color: #ddd;')
        self.preview.setAlignment(Qt.AlignCenter)

        self.progress = QProgressBar()
        self.progress.setRange(0, 100)

        self.log = QTextEdit()
        self.log.setReadOnly(True)

        right.addWidget(self.preview)
        right.addWidget(self.progress)
        right.addWidget(QLabel('运行日志'))
        right.addWidget(self.log)

        main_layout.addLayout(left, 0)
        main_layout.addLayout(right, 1)

        self.worker = None

    def request_stop(self):
        if self.worker and self.worker.isRunning():
            try:
                self.worker.request_stop()
                self.log_msg('已请求取消任务，正在尝试停止...')
            except Exception as e:
                self.log_msg(f'取消失败: {e}')
        else:
            self.log_msg('当前没有运行任务')

    def open_report(self):
        path = self.output_report_edit.text()
        if not path or not os.path.exists(path):
            self.log_msg('报告文件不存在')
            return
        try:
            if sys.platform.startswith('linux'):
                os.system(f'xdg-open "{path}"')
            elif sys.platform.startswith('win'):
                os.startfile(path)
            else:
                self.log_msg('不支持的打开平台')
        except Exception as e:
            self.log_msg(f'打开报告失败: {e}')

    def log_msg(self, text: str):
        self.log.append(text)

    def scan_devices(self):
        self.log_msg('开始扫描设备 (请等待)...')
        try:
            MvCamera.MV_CC_Initialize()
            device_list = da.enumerate_devices()
            self.log_msg(f'发现 {device_list.nDeviceNum} 个设备，详情见控制台输出')
        except Exception as e:
            self.log_msg(f'扫描设备失败: {e}')
        finally:
            try:
                MvCamera.MV_CC_Finalize()
            except Exception:
                pass

    def select_output_dir(self):
        d = QFileDialog.getExistingDirectory(self, '选择输出目录', os.path.abspath(self.output_dir_edit.text() or '.'))
        if d:
            self.output_dir_edit.setText(d)

    def select_report_file(self):
        f, _ = QFileDialog.getSaveFileName(self, '选择报告文件', os.path.abspath(self.output_report_edit.text() or './difference.txt'), 'Text Files (*.txt);;All Files (*)')
        if f:
            self.output_report_edit.setText(f)

    def select_input_dir(self):
        d = QFileDialog.getExistingDirectory(self, '选择图片目录', os.path.abspath(self.input_dir_edit.text() or '.'))
        if d:
            self.input_dir_edit.setText(d)

    def start_capture(self):
        if self.worker and self.worker.isRunning():
            self.log_msg('已有任务在运行，请稍候')
            return

        args = SimpleNamespace()
        args.device_index = int(self.device_index_spin.value())
        args.count = int(self.count_spin.value())
        args.discard_frames = int(self.discard_spin.value())
        args.timeout_ms = int(self.timeout_spin.value())
        args.output_dir = self.output_dir_edit.text() or './capture_data'
        args.output_report = self.output_report_edit.text() or './difference.txt'
        args.exposure = float(self.exposure_box.value()) if self.exposure_box.value() > 0 else None
        args.gain = float(self.gain_box.value()) if self.gain_box.value() > 0 else None
        args.frame_rate = None if self.auto_frame_rate_check.isChecked() or self.frame_rate_box.value() <= 0 else float(self.frame_rate_box.value())
        args.enable_trigger = bool(self.trigger_check.isChecked())
        args.input_dir = None

        self.worker = WorkerThread('capture', args)
        self.worker.log_signal.connect(self.log_msg)
        self.worker.progress_signal.connect(lambda v: self.progress.setValue(v))
        # status_signal updates label and log
        if hasattr(self.worker, 'status_signal'):
            self.worker.status_signal.connect(lambda s: self.status_label.setText(s))
            self.worker.status_signal.connect(self.log_msg)
        if hasattr(self.worker, 'camera_info_signal'):
            self.worker.camera_info_signal.connect(self.camera_info_view.setPlainText)
        self.worker.image_signal.connect(self.update_preview_from_numpy)
        self.worker.finished_signal.connect(lambda s: self.log_msg(f'任务结束: {s}'))
        self.worker.start()
        self.log_msg('已启动采集线程')

    def start_analyze(self):
        if self.worker and self.worker.isRunning():
            self.log_msg('已有任务在运行，请稍候')
            return

        input_dir = self.input_dir_edit.text()
        if not input_dir:
            self.log_msg('请先选择离线图片目录')
            return

        args = SimpleNamespace()
        args.input_dir = input_dir
        args.output_report = self.output_report_edit.text() or './difference.txt'

        self.worker = WorkerThread('analyze', args)
        self.worker.log_signal.connect(self.log_msg)
        if hasattr(self.worker, 'status_signal'):
            self.worker.status_signal.connect(lambda s: self.status_label.setText(s))
            self.worker.status_signal.connect(self.log_msg)
        self.worker.finished_signal.connect(lambda s: self.log_msg(f'任务结束: {s}'))
        self.worker.start()
        self.log_msg('已启动离线分析线程')

    def update_preview_from_numpy(self, arr):
        try:
            if arr is None:
                return
            if arr.ndim == 2:
                h, w = arr.shape
                img = QImage(arr.data, w, h, w, QImage.Format_Grayscale8)
            else:
                rgb = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb.shape
                bytes_per_line = ch * w
                img = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)

            pix = QPixmap.fromImage(img).scaled(self.preview.size(), Qt.KeepAspectRatio)
            self.preview.setPixmap(pix)
        except Exception as e:
            self.log_msg(f'更新预览失败: {e}')


def main():
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
