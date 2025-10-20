# -*- coding: utf-8 -*-
"""
IoT Sensing Systems – Homework 1
Final Integrated Implementation (with summary CSV output)
Activities: Standing, Sitting, Walking, Running
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from src.helper import low_pass_filter, discrete_convolution, peak_count, estimate_pose

# --- Activity set ---
activities = ["standing", "sitting", "walking", "running"]

# --- Summary results storage ---
summary_records = []

def check_files(activity):
    acc_path = os.path.join("src", f"Accelerometer_{activity}.csv")
    gyr_path = os.path.join("src", f"Gyroscope_{activity}.csv")
    if not os.path.exists(acc_path):
        print(f"⚠️ Missing accelerometer file for {activity}. Skipping...")
        return None, None
    if not os.path.exists(gyr_path):
        print(f"⚠️ Missing gyroscope file for {activity}. Skipping pose estimation.")
    return acc_path, gyr_path


for activity in activities:
    print(f"\n{'='*60}\nProcessing Activity: {activity.upper()}\n{'='*60}")

    acc_path, gyr_path = check_files(activity)
    if acc_path is None:
        continue

    # --- Load accelerometer data ---
    data = pd.read_csv(acc_path)
    if not {'x', 'y', 'z'}.issubset(data.columns):
        print(f"Error: {acc_path} missing x,y,z columns.")
        continue

    ax, ay, az = data['x'], data['y'], data['z']
    time = data['seconds_elapsed'] if 'seconds_elapsed' in data.columns else np.arange(len(ax)) / 50.0
    magnitude = np.sqrt(ax**2 + ay**2 + az**2)

    # --- Summary statistics ---
    samples = len(data)
    duration = time.max() - time.min()
    mean_mag = magnitude.mean()
    std_mag = magnitude.std()

    print(f"Samples: {samples}")
    print(f"Duration (s): {duration:.2f}")
    print(f"Mean Magnitude: {mean_mag:.3f} m/s²")
    print(f"Std Magnitude: {std_mag:.3f} m/s²")
    print(f"Min/Max Magnitude: {magnitude.min():.3f} / {magnitude.max():.3f} m/s²")

    # --- Step counting ---
    print("\n--- Step Counting ---")
    sampling_frequency = len(magnitude) / (time.iloc[-1] - time.iloc[0])
    lpf = low_pass_filter(sampling_frequency, cutoff_hz=2.0)
    filtered_signal = discrete_convolution(magnitude, lpf)
    steps = peak_count(filtered_signal, window_size=25, threshold_factor=1.0)
    print(f"Estimated Sampling Frequency: {sampling_frequency:.2f} Hz")
    print(f"Detected Steps: {steps}")

    plt.figure(figsize=(10, 5))
    plt.plot(time[:len(filtered_signal)], filtered_signal, color="black")
    plt.xlabel("Time (s)")
    plt.ylabel("Filtered Magnitude (m/s²)")
    plt.title(f"Low-Pass Filtered Signal – {activity.capitalize()}")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # --- Pose estimation ---
    final_roll = final_pitch = final_yaw = np.nan
    if gyr_path and os.path.exists(gyr_path):
        print("\n--- Pose Estimation ---")
        gyro = pd.read_csv(gyr_path)
        min_len = min(len(data), len(gyro))
        accel_values = data[['x', 'y', 'z']].to_numpy()[:min_len]
        gyro_values = gyro[['x', 'y', 'z']].to_numpy()[:min_len]

        if gyro_values.max() > 20:  # deg/s → rad/s
            gyro_values = np.radians(gyro_values)

        sampling_rate = len(accel_values) / (time.max() - time.min())
        roll, pitch, yaw = estimate_pose(accel_values, gyro_values, sampling_rate, alpha=0.98)

        plt.figure(figsize=(10, 5))
        plt.plot(roll, label="Roll (°)")
        plt.plot(pitch, label="Pitch (°)")
        plt.plot(yaw, label="Yaw (°)")
        plt.title(f"Estimated Orientation – {activity.capitalize()}")
        plt.xlabel("Sample index")
        plt.ylabel("Angle (°)")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        final_roll, final_pitch, final_yaw = roll[-1], pitch[-1], yaw[-1]
        print(f"Final Orientation: Roll={final_roll:.2f}, Pitch={final_pitch:.2f}, Yaw={final_yaw:.2f}")
    else:
        print("⚠️ Gyroscope data missing, skipping pose estimation.")

    # --- Save summary for this activity ---
    summary_records.append({
        "Activity": activity.capitalize(),
        "Samples": samples,
        "Duration_s": round(duration, 2),
        "MeanMag_mps2": round(mean_mag, 3),
        "StdMag_mps2": round(std_mag, 3),
        "Steps": steps,
        "FinalRoll_deg": round(final_roll, 2) if not np.isnan(final_roll) else None,
        "FinalPitch_deg": round(final_pitch, 2) if not np.isnan(final_pitch) else None,
        "FinalYaw_deg": round(final_yaw, 2) if not np.isnan(final_yaw) else None
    })

# --- Export summary CSV ---
if summary_records:
    summary_df = pd.DataFrame(summary_records)
    summary_df.to_csv("summary_results.csv", index=False)
    print("\n✅ Summary saved as summary_results.csv:")
    print(summary_df)
else:
    print("\n⚠️ No activity data processed. Nothing saved.")
