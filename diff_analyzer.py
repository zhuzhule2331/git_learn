import argparse
import os
import sys
from datetime import datetime
from pathlib import Path
from ctypes import c_bool

SYSTEM_DIST_PACKAGES = '/usr/lib/python3/dist-packages'
if os.path.isdir(SYSTEM_DIST_PACKAGES) and SYSTEM_DIST_PACKAGES not in sys.path:
    sys.path.insert(0, SYSTEM_DIST_PACKAGES)

import cv2
import numpy as np

from MvImport.CameraParams_header import *
from MvImport.MvCameraControl_class import *
from MvImport.MvErrorDefine_const import *


def decoding_char(ctypes_char_array):
    byte_str = memoryview(ctypes_char_array).tobytes()
    null_index = byte_str.find(b'\x00')
    if null_index != -1:
        byte_str = byte_str[:null_index]

    for encoding in ['gbk', 'utf-8', 'latin-1']:
        try:
            return byte_str.decode(encoding)
        except UnicodeDecodeError:
            continue

    return byte_str.decode('latin-1', errors='replace')


def to_hex_str(num):
    cha_dic = {10: 'a', 11: 'b', 12: 'c', 13: 'd', 14: 'e', 15: 'f'}
    hex_str = ''
    if num < 0:
        num = num + 2 ** 32
    while num >= 16:
        digit = num % 16
        hex_str = cha_dic.get(digit, str(digit)) + hex_str
        num //= 16
    hex_str = cha_dic.get(num, str(num)) + hex_str
    return hex_str


def read_int_feature(camera, key):
    feature = MVCC_INTVALUE()
    memset(byref(feature), 0, sizeof(feature))
    ret = camera.MV_CC_GetIntValue(key, feature)
    if ret != 0:
        return None, ret
    return {
        'current': int(feature.nCurValue),
        'min': int(feature.nMin),
        'max': int(feature.nMax),
        'inc': int(feature.nInc),
    }, 0


def read_float_feature(camera, key):
    feature = MVCC_FLOATVALUE()
    memset(byref(feature), 0, sizeof(feature))
    ret = camera.MV_CC_GetFloatValue(key, feature)
    if ret != 0:
        return None, ret
    return {
        'current': float(feature.fCurValue),
        'min': float(feature.fMin),
        'max': float(feature.fMax),
    }, 0


def read_bool_feature(camera, key):
    value = c_bool(False)
    ret = camera.MV_CC_GetBoolValue(key, value)
    if ret != 0:
        return None, ret
    return bool(value.value), 0


def read_enum_feature(camera, key):
    feature = MVCC_ENUMVALUE()
    memset(byref(feature), 0, sizeof(feature))
    ret = camera.MV_CC_GetEnumValue(key, feature)
    if ret != 0:
        return None, ret
    return {
        'current': int(feature.nCurValue),
        'supported_num': int(feature.nSupportedNum),
    }, 0


def collect_camera_snapshot(camera):
    int_keys = ['Width', 'Height', 'OffsetX', 'OffsetY', 'PayloadSize', 'GevSCPSPacketSize']
    float_keys = ['ExposureTime', 'Gain', 'AcquisitionFrameRate', 'ResultingFrameRate']
    bool_keys = ['AcquisitionFrameRateEnable', 'GammaEnable']
    enum_keys = ['PixelFormat', 'TriggerMode', 'TriggerSource', 'BalanceWhiteAuto', 'ExposureAuto', 'GainAuto', 'SharpnessMode']

    snapshot = {'ints': {}, 'floats': {}, 'bools': {}, 'enums': {}, 'errors': {}}

    for key in int_keys:
        value, ret = read_int_feature(camera, key)
        if ret == 0:
            snapshot['ints'][key] = value
        else:
            snapshot['errors'][key] = ret

    for key in float_keys:
        value, ret = read_float_feature(camera, key)
        if ret == 0:
            snapshot['floats'][key] = value
        else:
            snapshot['errors'][key] = ret

    for key in bool_keys:
        value, ret = read_bool_feature(camera, key)
        if ret == 0:
            snapshot['bools'][key] = value
        else:
            snapshot['errors'][key] = ret

    for key in enum_keys:
        value, ret = read_enum_feature(camera, key)
        if ret == 0:
            snapshot['enums'][key] = value
        else:
            snapshot['errors'][key] = ret

    return snapshot


def format_camera_snapshot(snapshot):
    lines = []
    lines.append('【相机参数快照】')
    lines.append('-' * 72)

    lines.append('基本尺寸/缓存')
    for key in ['Width', 'Height', 'OffsetX', 'OffsetY', 'PayloadSize', 'GevSCPSPacketSize']:
        if key in snapshot['ints']:
            item = snapshot['ints'][key]
            lines.append(
                f'  {key:<24} current={item["current"]} min={item["min"]} max={item["max"]} inc={item["inc"]}'
            )

    lines.append('曝光/增益')
    for key in ['ExposureTime', 'Gain', 'AcquisitionFrameRate', 'ResultingFrameRate']:
        if key in snapshot['floats']:
            item = snapshot['floats'][key]
            lines.append(
                f'  {key:<24} current={item["current"]:.3f} min={item["min"]:.3f} max={item["max"]:.3f}'
            )

    lines.append('开关/枚举')
    for key in ['AcquisitionFrameRateEnable', 'GammaEnable']:
        if key in snapshot['bools']:
            lines.append(f'  {key:<24} {snapshot["bools"][key]}')

    for key in ['PixelFormat', 'TriggerMode', 'TriggerSource', 'BalanceWhiteAuto', 'ExposureAuto', 'GainAuto', 'SharpnessMode']:
        if key in snapshot['enums']:
            current = snapshot['enums'][key]['current']
            if key == 'PixelFormat':
                lines.append(f'  {key:<24} {current} ({describe_pixel_type(current)})')
            else:
                lines.append(f'  {key:<24} {current}')

    if snapshot['errors']:
        lines.append('')
        lines.append('读取失败项')
        for key, ret in snapshot['errors'].items():
            lines.append(f'  {key:<24} ret=0x{to_hex_str(ret)}')

    lines.append('-' * 72)
    return '\n'.join(lines)


def get_supported_images(folder_path):
    extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.raw')
    images = []
    for ext in extensions:
        images.extend(Path(folder_path).glob(f'*{ext}'))
        images.extend(Path(folder_path).glob(f'*{ext.upper()}'))
    return sorted(images)


def load_image(path):
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is not None:
        return image

    raw = np.fromfile(str(path), dtype=np.uint8)
    if raw.size == 0:
        return None
    return cv2.imdecode(raw, cv2.IMREAD_COLOR)


def compute_image_differences(img1, img2):
    if img1.shape != img2.shape:
        height, width = img1.shape[:2]
        img2 = cv2.resize(img2, (width, height))

    if len(img1.shape) == 3:
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    else:
        gray1 = img1
        gray2 = img2

    abs_diff = cv2.absdiff(gray1, gray2)
    mean_diff = np.mean(abs_diff)
    max_diff = np.max(abs_diff)
    mse = np.mean((gray1.astype(float) - gray2.astype(float)) ** 2)

    if mse == 0:
        psnr = float('inf')
    else:
        psnr = 20 * np.log10(255.0 / np.sqrt(mse))

    c1 = (0.01 * 255) ** 2
    c2 = (0.03 * 255) ** 2
    window_size = 7

    mu1 = cv2.GaussianBlur(gray1.astype(float), (window_size, window_size), 1.5)
    mu2 = cv2.GaussianBlur(gray2.astype(float), (window_size, window_size), 1.5)
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = cv2.GaussianBlur((gray1.astype(float) ** 2), (window_size, window_size), 1.5) - mu1_sq
    sigma2_sq = cv2.GaussianBlur((gray2.astype(float) ** 2), (window_size, window_size), 1.5) - mu2_sq
    sigma12 = cv2.GaussianBlur((gray1.astype(float) * gray2.astype(float)), (window_size, window_size), 1.5) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2))
    ssim = np.mean(ssim_map)

    edge1 = cv2.Canny(gray1, 50, 150)
    edge2 = cv2.Canny(gray2, 50, 150)
    edge_diff = np.sum(edge1 != edge2) / edge1.size * 100

    hist1 = cv2.calcHist([gray1], [0], None, [256], [0, 256])
    hist2 = cv2.calcHist([gray2], [0], None, [256], [0, 256])
    hist_diff = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CHISQR)

    return {
        'mean_abs_diff': round(mean_diff, 3),
        'max_abs_diff': int(max_diff),
        'mse': round(mse, 2),
        'psnr': round(psnr, 2) if psnr != float('inf') else 'inf',
        'ssim': round(ssim, 4),
        'edge_diff_percent': round(edge_diff, 2),
        'hist_diff': round(hist_diff, 4),
    }


def analyze_image_sequence(image_paths):
    results = []
    if len(image_paths) < 2:
        return results

    ref_img = load_image(image_paths[0])
    if ref_img is None:
        print(f'无法读取参考图片: {image_paths[0]}')
        return results

    ref_filename = image_paths[0].name
    for index in range(1, len(image_paths)):
        curr_img = load_image(image_paths[index])
        if curr_img is None:
            print(f'无法读取图片: {image_paths[index]}')
            continue

        prev_img = load_image(image_paths[index - 1])
        if prev_img is None:
            continue

        results.append({
            'frame_index': index,
            'current_file': image_paths[index].name,
            'prev_file': image_paths[index - 1].name,
            'ref_file': ref_filename,
            'frame_to_frame': compute_image_differences(prev_img, curr_img),
            'frame_to_ref': compute_image_differences(ref_img, curr_img),
        })

    return results


def write_log(results, output_path, source_desc):
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    with open(output_path, 'w', encoding='utf-8') as file_obj:
        file_obj.write('=' * 80 + '\n')
        file_obj.write('相机图像差异分析报告\n')
        file_obj.write(f'生成时间: {timestamp}\n')
        file_obj.write(f'数据来源: {source_desc}\n')
        file_obj.write('=' * 80 + '\n\n')

        if not results:
            file_obj.write('未找到有效的图片对进行分析\n')
            return

        frame_to_frame_ssim = [r['frame_to_frame']['ssim'] for r in results]
        frame_to_ref_ssim = [r['frame_to_ref']['ssim'] for r in results]
        frame_to_frame_edge = [r['frame_to_frame']['edge_diff_percent'] for r in results]
        frame_to_ref_psnr = [r['frame_to_ref']['psnr'] for r in results if r['frame_to_ref']['psnr'] != 'inf']

        file_obj.write('【汇总统计】\n')
        file_obj.write('-' * 60 + '\n')
        file_obj.write(f'总帧数(除去首帧): {len(results)}\n')
        file_obj.write(f'帧间SSIM范围: {min(frame_to_frame_ssim):.4f} ~ {max(frame_to_frame_ssim):.4f}\n')
        file_obj.write(f'帧间SSIM均值: {np.mean(frame_to_frame_ssim):.4f}\n')
        file_obj.write(f'累积SSIM范围: {min(frame_to_ref_ssim):.4f} ~ {max(frame_to_ref_ssim):.4f}\n')
        file_obj.write(f'累积SSIM均值: {np.mean(frame_to_ref_ssim):.4f}\n')
        file_obj.write(f'帧间边缘差异均值: {np.mean(frame_to_frame_edge):.2f}%\n')
        if frame_to_ref_psnr:
            file_obj.write(f'累积PSNR均值: {np.mean(frame_to_ref_psnr):.2f} dB\n')

        file_obj.write('\n【稳定性评估】\n')
        ssim_std = np.std(frame_to_frame_ssim)
        edge_std = np.std(frame_to_frame_edge)
        if ssim_std < 0.01 and edge_std < 5:
            file_obj.write('序列稳定性良好：帧间SSIM标准差<0.01，边缘差异标准差<5%\n')
        elif ssim_std < 0.05 and edge_std < 10:
            file_obj.write('序列稳定性一般：存在可察觉的帧间抖动\n')
        else:
            file_obj.write('序列稳定性较差：帧间抖动明显，建议检查相机参数设置\n')

        file_obj.write('\n【详细比对结果】\n')
        file_obj.write('-' * 90 + '\n')
        file_obj.write(f"{'序号':<6} {'当前图片':<24} {'帧间SSIM':<10} {'帧间边缘差异%':<14} {'累积SSIM':<10} {'累积PSNR':<8}\n")
        file_obj.write('-' * 90 + '\n')
        for result in results:
            psnr_val = result['frame_to_ref']['psnr']
            psnr_str = str(psnr_val) if psnr_val == 'inf' else f'{psnr_val:<8}'
            file_obj.write(
                f"{result['frame_index']:<6} {result['current_file']:<24} "
                f"{result['frame_to_frame']['ssim']:<10} "
                f"{result['frame_to_frame']['edge_diff_percent']:<14} "
                f"{result['frame_to_ref']['ssim']:<10} {psnr_str}\n"
            )

        file_obj.write('\n【异常帧检测】\n')
        anomaly_threshold_ssim = 0.85
        anomaly_threshold_edge = 15
        anomalies = []
        for result in results:
            if result['frame_to_frame']['ssim'] < anomaly_threshold_ssim:
                anomalies.append((result['current_file'], f"帧间SSIM过低: {result['frame_to_frame']['ssim']}"))
            if result['frame_to_frame']['edge_diff_percent'] > anomaly_threshold_edge:
                anomalies.append((result['current_file'], f"帧间边缘差异过大: {result['frame_to_frame']['edge_diff_percent']}%"))
            if result['frame_to_ref']['ssim'] < 0.7:
                anomalies.append((result['current_file'], f"累积漂移严重: SSIM={result['frame_to_ref']['ssim']}"))

        if anomalies:
            file_obj.write('发现以下异常帧:\n')
            for file_name, reason in anomalies:
                file_obj.write(f'  - {file_name}: {reason}\n')
        else:
            file_obj.write('未检测到明显的异常帧\n')

        file_obj.write('\n【诊断建议】\n')
        avg_ssim = np.mean(frame_to_frame_ssim)
        avg_edge = np.mean(frame_to_frame_edge)
        if avg_ssim < 0.95:
            file_obj.write('1. 帧间SSIM偏低(<0.95)，建议检查曝光、增益、白平衡和光照稳定性\n')
        if avg_edge > 10:
            file_obj.write('2. 帧间边缘差异较大(>10%)，建议检查锐化、光源频闪和机械振动\n')
        if np.std(frame_to_frame_ssim) > 0.03:
            file_obj.write('3. 帧间SSIM波动较大，建议检查网络传输、缓存策略和相机温度\n')

        file_obj.write('\n' + '=' * 80 + '\n')
        file_obj.write('报告结束\n')

    print(f'差异报告已保存至: {output_path}')


def print_device_info(device_info, index):
    if device_info.nTLayerType in (MV_GIGE_DEVICE, MV_GENTL_GIGE_DEVICE):
        user_defined_name = decoding_char(device_info.SpecialInfo.stGigEInfo.chUserDefinedName)
        model_name = decoding_char(device_info.SpecialInfo.stGigEInfo.chModelName)
        serial_number = decoding_char(device_info.SpecialInfo.stGigEInfo.chSerialNumber)
        ip = device_info.SpecialInfo.stGigEInfo.nCurrentIp
        nip1 = (ip & 0xff000000) >> 24
        nip2 = (ip & 0x00ff0000) >> 16
        nip3 = (ip & 0x0000ff00) >> 8
        nip4 = ip & 0x000000ff
        print(f'[{index}] GigE: {user_defined_name} {model_name} ({serial_number}) {nip1}.{nip2}.{nip3}.{nip4}')
    elif device_info.nTLayerType == MV_USB_DEVICE:
        user_defined_name = decoding_char(device_info.SpecialInfo.stUsb3VInfo.chUserDefinedName)
        model_name = decoding_char(device_info.SpecialInfo.stUsb3VInfo.chModelName)
        serial_number = decoding_char(device_info.SpecialInfo.stUsb3VInfo.chSerialNumber)
        print(f'[{index}] USB: {user_defined_name} {model_name} ({serial_number})')
    elif device_info.nTLayerType == MV_GENTL_CAMERALINK_DEVICE:
        user_defined_name = decoding_char(device_info.SpecialInfo.stCMLInfo.chUserDefinedName)
        model_name = decoding_char(device_info.SpecialInfo.stCMLInfo.chModelName)
        serial_number = decoding_char(device_info.SpecialInfo.stCMLInfo.chSerialNumber)
        print(f'[{index}] CML: {user_defined_name} {model_name} ({serial_number})')
    elif device_info.nTLayerType == MV_GENTL_CXP_DEVICE:
        user_defined_name = decoding_char(device_info.SpecialInfo.stCXPInfo.chUserDefinedName)
        model_name = decoding_char(device_info.SpecialInfo.stCXPInfo.chModelName)
        serial_number = decoding_char(device_info.SpecialInfo.stCXPInfo.chSerialNumber)
        print(f'[{index}] CXP: {user_defined_name} {model_name} ({serial_number})')
    elif device_info.nTLayerType == MV_GENTL_XOF_DEVICE:
        user_defined_name = decoding_char(device_info.SpecialInfo.stXoFInfo.chUserDefinedName)
        model_name = decoding_char(device_info.SpecialInfo.stXoFInfo.chModelName)
        serial_number = decoding_char(device_info.SpecialInfo.stXoFInfo.chSerialNumber)
        print(f'[{index}] XoF: {user_defined_name} {model_name} ({serial_number})')
    else:
        print(f'[{index}] Unknown device type: {device_info.nTLayerType}')


def enumerate_devices(layer_type=None):
    device_list = MV_CC_DEVICE_INFO_LIST()
    if layer_type is None:
        layer_type = MV_GIGE_DEVICE | MV_USB_DEVICE | MV_GENTL_GIGE_DEVICE | MV_GENTL_CAMERALINK_DEVICE | MV_GENTL_CXP_DEVICE | MV_GENTL_XOF_DEVICE

    ret = MvCamera.MV_CC_EnumDevicesEx2(layer_type, device_list, '', SortMethod_SerialNumber)
    if ret != 0:
        raise RuntimeError(f'enum devices fail, ret=0x{to_hex_str(ret)}')

    if device_list.nDeviceNum == 0:
        raise RuntimeError('未找到可用相机')

    for index in range(device_list.nDeviceNum):
        device_info = cast(device_list.pDeviceInfo[index], POINTER(MV_CC_DEVICE_INFO)).contents
        print_device_info(device_info, index)

    return device_list


def open_camera(device_list, device_index):
    if device_index < 0 or device_index >= device_list.nDeviceNum:
        raise ValueError(f'device_index 越界: {device_index}')

    device_info = cast(device_list.pDeviceInfo[device_index], POINTER(MV_CC_DEVICE_INFO)).contents
    camera = MvCamera()
    ret = camera.MV_CC_CreateHandle(device_info)
    if ret != 0:
        raise RuntimeError(f'创建句柄失败, ret=0x{to_hex_str(ret)}')

    ret = camera.MV_CC_OpenDevice(MV_ACCESS_Exclusive, 0)
    if ret != 0:
        camera.MV_CC_DestroyHandle()
        raise RuntimeError(f'打开相机失败, ret=0x{to_hex_str(ret)}')

    if device_info.nTLayerType in (MV_GIGE_DEVICE, MV_GENTL_GIGE_DEVICE):
        packet_size = camera.MV_CC_GetOptimalPacketSize()
        if int(packet_size) > 0:
            ret = camera.MV_CC_SetIntValue('GevSCPSPacketSize', packet_size)
            if ret != 0:
                print(f'警告: 设置包大小失败 ret=0x{to_hex_str(ret)}')
        else:
            print(f'警告: 获取最优包大小失败 ret=0x{to_hex_str(packet_size)}')

    return camera, device_info


def apply_stable_camera_params(camera, args):
    steps = [
        ('ExposureAuto', lambda: camera.MV_CC_SetEnumValue('ExposureAuto', 0)),
        ('GainAuto', lambda: camera.MV_CC_SetEnumValue('GainAuto', 0)),
        ('BalanceWhiteAuto', lambda: camera.MV_CC_SetEnumValue('BalanceWhiteAuto', 0)),
        ('GammaEnable', lambda: camera.MV_CC_SetBoolValue('GammaEnable', False)),
        ('SharpnessMode', lambda: camera.MV_CC_SetEnumValue('SharpnessMode', 0)),
    ]

    for name, setter in steps:
        ret = setter()
        if ret != 0:
            print(f'警告: 设置 {name} 失败 ret=0x{to_hex_str(ret)}')

    if args.exposure is not None:
        ret = camera.MV_CC_SetFloatValue('ExposureTime', float(args.exposure))
        if ret != 0:
            raise RuntimeError(f'设置 ExposureTime 失败, ret=0x{to_hex_str(ret)}')

    if args.gain is not None:
        ret = camera.MV_CC_SetFloatValue('Gain', float(args.gain))
        if ret != 0:
            raise RuntimeError(f'设置 Gain 失败, ret=0x{to_hex_str(ret)}')

    desired_frame_rate = args.frame_rate
    if desired_frame_rate is None:
        frame_rate_info, ret = read_float_feature(camera, 'AcquisitionFrameRate')
        if ret == 0 and frame_rate_info and frame_rate_info['max'] > 0:
            desired_frame_rate = frame_rate_info['max']

    if desired_frame_rate is not None:
        ret = camera.MV_CC_SetBoolValue('AcquisitionFrameRateEnable', True)
        if ret != 0:
            print(f'警告: 开启帧率控制失败 ret=0x{to_hex_str(ret)}')
        ret = camera.MV_CC_SetFloatValue('AcquisitionFrameRate', float(desired_frame_rate))
        if ret != 0:
            print(f'警告: 设置 AcquisitionFrameRate={desired_frame_rate} 失败 ret=0x{to_hex_str(ret)}')

    if args.enable_trigger:
        ret = camera.MV_CC_SetEnumValue('TriggerMode', 1)
        if ret != 0:
            raise RuntimeError(f'开启触发模式失败, ret=0x{to_hex_str(ret)}')
        ret = camera.MV_CC_SetEnumValue('TriggerSource', 7)
        if ret != 0:
            raise RuntimeError(f'设置软件触发源失败, ret=0x{to_hex_str(ret)}')
    else:
        ret = camera.MV_CC_SetEnumValue('TriggerMode', 0)
        if ret != 0:
            print(f'警告: 关闭触发模式失败 ret=0x{to_hex_str(ret)}')


def configure_streaming(camera):
    ret = camera.MV_CC_SetImageNodeNum(1)
    if ret != 0:
        print(f'警告: SetImageNodeNum(1) 失败 ret=0x{to_hex_str(ret)}')

    ret = camera.MV_CC_SetGrabStrategy(MV_GrabStrategy_LatestImagesOnly)
    if ret != 0:
        print(f'警告: SetGrabStrategy(LatestImagesOnly) 失败 ret=0x{to_hex_str(ret)}')

    ret = camera.MV_CC_SetBayerCvtQuality(1)
    if ret != 0:
        print(f'警告: SetBayerCvtQuality(1) 失败 ret=0x{to_hex_str(ret)}')

    ret = camera.MV_CC_ClearImageBuffer()
    if ret != 0:
        print(f'警告: ClearImageBuffer 失败 ret=0x{to_hex_str(ret)}')


def is_mono_pixel_type(pixel_type):
    return pixel_type in {
        PixelType_Gvsp_Mono8,
        PixelType_Gvsp_Mono10,
        PixelType_Gvsp_Mono10_Packed,
        PixelType_Gvsp_Mono12,
        PixelType_Gvsp_Mono12_Packed,
    }


def is_color_pixel_type(pixel_type):
    return pixel_type in {
        PixelType_Gvsp_RGB8_Packed,
        PixelType_Gvsp_BGR8_Packed,
        PixelType_Gvsp_BayerGR8,
        PixelType_Gvsp_BayerRG8,
        PixelType_Gvsp_BayerGB8,
        PixelType_Gvsp_BayerBG8,
        PixelType_Gvsp_BayerGR10,
        PixelType_Gvsp_BayerRG10,
        PixelType_Gvsp_BayerGB10,
        PixelType_Gvsp_BayerBG10,
        PixelType_Gvsp_BayerGR12,
        PixelType_Gvsp_BayerRG12,
        PixelType_Gvsp_BayerGB12,
        PixelType_Gvsp_BayerBG12,
        PixelType_Gvsp_BayerGR10_Packed,
        PixelType_Gvsp_BayerRG10_Packed,
        PixelType_Gvsp_BayerGB10_Packed,
        PixelType_Gvsp_BayerBG10_Packed,
        PixelType_Gvsp_BayerGR12_Packed,
        PixelType_Gvsp_BayerRG12_Packed,
        PixelType_Gvsp_BayerGB12_Packed,
        PixelType_Gvsp_BayerBG12_Packed,
        PixelType_Gvsp_BayerRBGG8,
        PixelType_Gvsp_BayerGR16,
        PixelType_Gvsp_BayerRG16,
        PixelType_Gvsp_BayerGB16,
        PixelType_Gvsp_BayerBG16,
        PixelType_Gvsp_YUV422_Packed,
        PixelType_Gvsp_YUV422_YUYV_Packed,
    }


def describe_pixel_type(pixel_type):
    pixel_type_names = {
        PixelType_Gvsp_Mono8: 'Mono8',
        PixelType_Gvsp_Mono10: 'Mono10',
        PixelType_Gvsp_Mono10_Packed: 'Mono10_Packed',
        PixelType_Gvsp_Mono12: 'Mono12',
        PixelType_Gvsp_Mono12_Packed: 'Mono12_Packed',
        PixelType_Gvsp_RGB8_Packed: 'RGB8_Packed',
        PixelType_Gvsp_BGR8_Packed: 'BGR8_Packed',
        PixelType_Gvsp_BayerGR8: 'BayerGR8',
        PixelType_Gvsp_BayerRG8: 'BayerRG8',
        PixelType_Gvsp_BayerGB8: 'BayerGB8',
        PixelType_Gvsp_BayerBG8: 'BayerBG8',
        PixelType_Gvsp_YUV422_Packed: 'YUV422_Packed',
        PixelType_Gvsp_YUV422_YUYV_Packed: 'YUV422_YUYV_Packed',
        PixelType_Gvsp_HB_Mono8: 'HB_Mono8',
        PixelType_Gvsp_HB_BayerGR8: 'HB_BayerGR8',
        PixelType_Gvsp_HB_BayerRG8: 'HB_BayerRG8',
        PixelType_Gvsp_HB_BayerGB8: 'HB_BayerGB8',
        PixelType_Gvsp_HB_BayerBG8: 'HB_BayerBG8',
        PixelType_Gvsp_HB_YUV422_Packed: 'HB_YUV422_Packed',
        PixelType_Gvsp_HB_YUV422_YUYV_Packed: 'HB_YUV422_YUYV_Packed',
        PixelType_Gvsp_HB_RGB8_Packed: 'HB_RGB8_Packed',
        PixelType_Gvsp_HB_BGR8_Packed: 'HB_BGR8_Packed',
    }
    return pixel_type_names.get(pixel_type, f'0x{to_hex_str(pixel_type)}')


def convert_frame_to_bgr(camera, frame_out):
    frame_info = frame_out.stFrameInfo
    width = int(frame_info.nWidth)
    height = int(frame_info.nHeight)
    pixel_type = frame_info.enPixelType

    src_data = cast(frame_out.pBufAddr, POINTER(c_ubyte))
    src_len = int(frame_info.nFrameLen)
    src_array = np.ctypeslib.as_array(src_data, shape=(src_len,))

    if pixel_type == PixelType_Gvsp_BGR8_Packed:
        return src_array.reshape((height, width, 3)).copy()

    if pixel_type == PixelType_Gvsp_RGB8_Packed:
        rgb_image = src_array.reshape((height, width, 3)).copy()
        return cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)

    if pixel_type == PixelType_Gvsp_Mono8:
        gray_image = src_array.reshape((height, width)).copy()
        return cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)

    if is_mono_pixel_type(pixel_type):
        gray_image = src_array.reshape((height, width)).copy()
        return cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)

    if is_color_pixel_type(pixel_type):
        output_size = width * height * 3
        dst_buffer = (c_ubyte * output_size)()
        convert_param = MV_CC_PIXEL_CONVERT_PARAM()
        memset(byref(convert_param), 0, sizeof(convert_param))
        convert_param.nWidth = width
        convert_param.nHeight = height
        convert_param.enSrcPixelType = pixel_type
        convert_param.pSrcData = src_data
        convert_param.nSrcDataLen = src_len
        convert_param.enDstPixelType = PixelType_Gvsp_BGR8_Packed
        convert_param.pDstBuffer = cast(dst_buffer, POINTER(c_ubyte))
        convert_param.nDstBufferSize = output_size

        ret = camera.MV_CC_ConvertPixelTypeEx(convert_param)
        if ret != 0:
            raise RuntimeError(
                f'图像格式转换失败, ret=0x{to_hex_str(ret)}, '
                f'pixel_type={describe_pixel_type(pixel_type)}, '
                f'size={width}x{height}, frame_len={src_len}'
            )

        converted = np.ctypeslib.as_array(dst_buffer)[: int(convert_param.nDstLen)]
        return converted.reshape((height, width, 3)).copy()

    raise RuntimeError(f'暂不支持的像素格式: {pixel_type}')


def get_frame_for_bgr(camera, width, height, timeout_ms=1000):
    output_size = max(1, width * height * 3)
    dst_buffer = (c_ubyte * output_size)()
    frame_info = MV_FRAME_OUT_INFO_EX()
    memset(byref(frame_info), 0, sizeof(frame_info))

    ret = camera.MV_CC_GetImageForBGR(dst_buffer, output_size, frame_info, timeout_ms)
    if ret != 0:
        raise RuntimeError(
            f'GetImageForBGR 失败, ret=0x{to_hex_str(ret)}, '
            f'request_size={output_size}, width={width}, height={height}'
        )

    image = np.ctypeslib.as_array(dst_buffer)[: int(frame_info.nFrameLen)]
    if frame_info.nFrameLen <= 0:
        raise RuntimeError('GetImageForBGR 返回空帧')

    return frame_info, image.reshape((int(frame_info.nHeight), int(frame_info.nWidth), 3)).copy()


def grab_one_frame(camera, timeout_ms=1000):
    width_info, width_ret = read_int_feature(camera, 'Width')
    height_info, height_ret = read_int_feature(camera, 'Height')
    if width_ret == 0 and height_ret == 0 and width_info and height_info:
        try:
            return get_frame_for_bgr(camera, width_info['current'], height_info['current'], timeout_ms)
        except Exception as exc:
            print(f'警告: GetImageForBGR 失败，改用原始帧转换: {exc}')

    frame_out = MV_FRAME_OUT()
    memset(byref(frame_out), 0, sizeof(frame_out))
    ret = camera.MV_CC_GetImageBuffer(frame_out, timeout_ms)
    if ret != 0:
        raise RuntimeError(f'取图失败, ret=0x{to_hex_str(ret)}')

    try:
        try:
            image = convert_frame_to_bgr(camera, frame_out)
            return frame_out.stFrameInfo, image
        except Exception as exc:
            width = int(frame_out.stFrameInfo.nWidth)
            height = int(frame_out.stFrameInfo.nHeight)
            pixel_type = frame_out.stFrameInfo.enPixelType
            print(
                f'警告: 原始转换失败，准备使用 GetImageForBGR 兜底 '
                f'(pixel_type={describe_pixel_type(pixel_type)}, width={width}, height={height})'
            )
            camera.MV_CC_FreeImageBuffer(frame_out)
            try:
                fallback_frame_info, fallback_image = get_frame_for_bgr(camera, width, height, timeout_ms)
                return fallback_frame_info, fallback_image
            except Exception as fallback_exc:
                raise RuntimeError(
                    f'{exc}; 兜底转换也失败: {fallback_exc}'
                ) from fallback_exc
    finally:
        try:
            camera.MV_CC_FreeImageBuffer(frame_out)
        except Exception:
            pass


def save_frame(image, save_path):
    save_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(save_path), image):
        raise RuntimeError(f'保存图片失败: {save_path}')


def capture_images(args):
    device_list = enumerate_devices()
    camera, device_info = open_camera(device_list, args.device_index)

    try:
        apply_stable_camera_params(camera, args)
        configure_streaming(camera)

        ret = camera.MV_CC_StartGrabbing()
        if ret != 0:
            raise RuntimeError(f'开始取流失败, ret=0x{to_hex_str(ret)}')

        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        try:
            for index in range(args.discard_frames):
                grab_one_frame(camera, args.timeout_ms)
                print(f'已丢弃预热帧 {index + 1}/{args.discard_frames}')

            saved_paths = []
            for index in range(args.count):
                frame_info, image = grab_one_frame(camera, args.timeout_ms)
                save_path = output_dir / f'frame_{index:04d}.png'
                save_frame(image, save_path)
                saved_paths.append(save_path)
                print(
                    f"保存帧 {index + 1}/{args.count}: {save_path.name} "
                    f"{int(frame_info.nWidth)}x{int(frame_info.nHeight)} pixel_type={frame_info.enPixelType}"
                )

            results = analyze_image_sequence(saved_paths)
            source_name = f'camera_index={args.device_index}, device_type={device_info.nTLayerType}'
            write_log(results, args.output_report, source_name)
        finally:
            try:
                camera.MV_CC_StopGrabbing()
            except Exception:
                pass
    finally:
        try:
            camera.MV_CC_CloseDevice()
        finally:
            camera.MV_CC_DestroyHandle()


def analyze_existing_images(args):
    data_folder = Path(args.input_dir)
    if not data_folder.exists():
        raise FileNotFoundError(f'文件夹不存在: {data_folder}')

    image_files = get_supported_images(data_folder)
    if len(image_files) < 2:
        raise RuntimeError(f"在 '{data_folder}' 中找到 {len(image_files)} 张图片，需要至少2张图片进行比对")

    results = analyze_image_sequence(image_files)
    write_log(results, args.output_report, os.path.abspath(str(data_folder)))


def build_parser():
    parser = argparse.ArgumentParser(description='海康威视相机固定参数采集与帧间差异分析工具')
    parser.add_argument('--device-index', type=int, default=0, help='相机索引，默认0')
    parser.add_argument('--count', type=int, default=20, help='采集并保存的图片数量')
    parser.add_argument('--discard-frames', type=int, default=5, help='开始取流后先丢弃的预热帧数量')
    parser.add_argument('--timeout-ms', type=int, default=2000, help='单帧等待超时时间(毫秒)')
    parser.add_argument('--output-dir', default='./capture_data', help='采集图片保存目录')
    parser.add_argument('--output-report', default='./difference.txt', help='差异分析报告输出文件')
    parser.add_argument('--exposure', type=float, default=10000.0, help='固定曝光时间(默认10000us)')
    parser.add_argument('--gain', type=float, default=20.0, help='固定增益(默认20)')
    parser.add_argument('--frame-rate', type=float, default=None, help='固定帧率(可选)')
    parser.add_argument('--enable-trigger', action='store_true', help='开启软件触发模式')
    parser.add_argument('--input-dir', default=None, help='已有图片目录，提供后转为离线分析模式')
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    MvCamera.MV_CC_Initialize()
    try:
        if args.input_dir:
            analyze_existing_images(args)
        else:
            capture_images(args)
        print('处理完成')
    except Exception as exc:
        print(f'错误: {exc}')
    finally:
        MvCamera.MV_CC_Finalize()


if __name__ == '__main__':
    main()