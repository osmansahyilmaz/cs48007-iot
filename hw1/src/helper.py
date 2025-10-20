# src/helper.py
import numpy as np
import math

def low_pass_filter(sampling_frequency, cutoff_hz=2.0, N=25):
    """
    Create a simple low-pass filter using a Blackman window and sinc function.
    Matches the lecture method for reducing high-frequency noise.
    """
    n = np.arange(N)
    fc = cutoff_hz / sampling_frequency  # normalized cutoff
    blackman = 0.42 - 0.5 * np.cos(2*np.pi*n/(N-1)) + 0.08 * np.cos(4*np.pi*n/(N-1))
    h = np.sinc(2 * fc * (n - (N - 1) / 2)) * blackman
    h = h / np.sum(h)
    return h


def discrete_convolution(signal, filt):
    """
    Manual discrete convolution (no scipy).
    """
    conv_result = []
    for i in range(len(signal) + len(filt) - 1):
        y = 0
        for j in range(len(filt)):
            if 0 <= i - j < len(signal):
                y += signal[i - j] * filt[j]
        conv_result.append(y)
    return np.array(conv_result[len(filt)//2 : -(len(filt)//2)])


def peak_count(signal, window_size=30, threshold_factor=1.2):
    """
    Adaptive peak counter for step detection.
    - Uses mean + std-based threshold
    - Enforces minimum spacing (window_size) between steps
    """
    peaks = 0
    avg = np.mean(signal)
    std = np.std(signal)
    threshold = avg + threshold_factor * std
    last_peak = -window_size

    for i in range(1, len(signal) - 1):
        if (
            signal[i] > signal[i - 1]
            and signal[i] > signal[i + 1]
            and signal[i] > threshold
            and i - last_peak > window_size
        ):
            peaks += 1
            last_peak = i
    return peaks


def accel_to_angles(ax, ay, az):
    """
    Compute roll and pitch from accelerometer.
    Roll = atan2(ay, az)
    Pitch = atan2(-ax, sqrt(ay² + az²))
    """
    roll = math.atan2(ay, az)
    pitch = math.atan2(-ax, math.sqrt(ay**2 + az**2))
    return roll, pitch


def integrate_gyro(prev_angles, gx, gy, gz, dt):
    """
    Integrate gyroscope readings to get orientation deltas.
    gx, gy, gz expected in rad/s.
    """
    roll, pitch, yaw = prev_angles
    roll += gx * dt
    pitch += gy * dt
    yaw += gz * dt
    return roll, pitch, yaw


def complementary_filter(accel_angles, gyro_angles, alpha=0.98):
    """
    Fuse accelerometer (low-freq) and gyro (high-freq) angles.
    alpha close to 1 -> trust gyro more (less drift, less noise).
    """
    roll = alpha * gyro_angles[0] + (1 - alpha) * accel_angles[0]
    pitch = alpha * gyro_angles[1] + (1 - alpha) * accel_angles[1]
    yaw = gyro_angles[2]
    return roll, pitch, yaw


def estimate_pose(accel_data, gyro_data, sampling_rate, alpha=0.98):
    """
    Estimate roll, pitch, yaw for each sample using complementary filtering.
    Input arrays: accel_data[N,3], gyro_data[N,3]
    Output arrays: roll[], pitch[], yaw[] in degrees.
    """
    dt = 1.0 / sampling_rate
    roll_list, pitch_list, yaw_list = [], [], []
    roll, pitch, yaw = 0.0, 0.0, 0.0

    for i in range(len(accel_data)):
        ax, ay, az = accel_data[i]
        gx, gy, gz = gyro_data[i]
        acc_roll, acc_pitch = accel_to_angles(ax, ay, az)
        gyro_angles = integrate_gyro((roll, pitch, yaw), gx, gy, gz, dt)
        roll, pitch, yaw = complementary_filter((acc_roll, acc_pitch), gyro_angles, alpha)
        roll_list.append(math.degrees(roll))
        pitch_list.append(math.degrees(pitch))
        yaw_list.append(math.degrees(yaw))

    return np.array(roll_list), np.array(pitch_list), np.array(yaw_list)
