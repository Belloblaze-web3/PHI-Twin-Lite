import numpy as np
import pandas as pd
from scipy.signal import find_peaks

FAULTS = {
    0: "Normal", 1: "Low_Oleo_Pressure", 2: "High_Oleo_Pressure",
    3: "Hydraulic_Leak", 4: "Worn_Shock_Absorber", 5: "Excessive_Stiction",
    6: "Tire_Underinflated", 7: "Tire_Overinflated",
    8: "Strut_Seal_Failure", 9: "Sensor_Drift", 10: "Structural_Crack"
}

# fault_id: (stiffness_k, damping_c, amplitude, friction, bias, noise)
PARAMS = {
    0: (1.0, 1.0, 1.0, 0.0, 0.0, 0.01), 1: (0.85, 0.6, 1.2, 0.0, 0.0, 0.015),
    2: (1.15, 1.3, 0.85, 0.0, 0.0, 0.015), 3: (0.7, 0.45, 1.35, 0.0, 0.0, 0.02),
    4: (0.95, 0.35, 1.25, 0.0, 0.0, 0.02), 5: (1.0, 1.1, 0.9, 0.15, 0.0, 0.02),
    6: (0.6, 0.8, 1.4, 0.0, 0.0, 0.015), 7: (1.4, 0.9, 0.7, 0.0, 0.0, 0.015),
    8: (0.8, 0.5, 1.3, 0.08, 0.0, 0.03), 9: (1.0, 1.0, 1.0, 0.0, 0.25, 0.01),
    10: (0.45, 0.7, 1.5, 0.0, 0.0, 0.025),
}


def simulate_one(fault_id, fs=500, duration=1.0):
    if fault_id not in PARAMS:
        raise ValueError(f"Unknown fault_id {fault_id}; expected one of {list(PARAMS)}")
    k_f, c_f, amp_f, friction, bias, noise_lvl = PARAMS[fault_id]
    t = np.linspace(0, duration, int(fs * duration))
    m = 1500  # effective mass (kg)
    k = 180000 * k_f * np.random.uniform(0.95, 1.05)
    c = 12000 * c_f * np.random.uniform(0.95, 1.05)

    wn = np.sqrt(k / m)  # natural frequency
    zeta = c / (2 * np.sqrt(k * m))  # damping ratio
    wd = wn * np.sqrt(max(0, 1 - zeta**2))

    A = 0.25 * amp_f  # initial landing compression (m)
    x = A * np.exp(-zeta * wn * t) * np.cos(wd * t)
    x += friction * np.sign(np.gradient(x)) * 0.02  # stiction effect
    x += bias + np.random.normal(0, noise_lvl, len(t))
    pressure = 15 + (x * 45) + np.random.normal(0, 0.3, len(t))  # MPa proxy
    return t, x, pressure, zeta, wn


def extract_features(x, pressure, zeta, wn, fs=500):
    peaks, _ = find_peaks(x)
    fft_freq = np.fft.rfftfreq(len(x), 1 / fs)
    fft_mag = np.abs(np.fft.rfft(x - np.mean(x)))
    dominant_freq = fft_freq[1:][np.argmax(fft_mag[1:])] if len(fft_mag) > 1 else 0.0
    return {
        "peak_disp": np.max(np.abs(x)), "rms_disp": np.sqrt(np.mean(x**2)),
        "std_disp": np.std(x), "max_pressure": np.max(pressure),
        "mean_pressure": np.mean(pressure), "damping_ratio": zeta,
        "natural_freq": wn / (2 * np.pi), "dominant_freq": dominant_freq,
        "kurtosis": pd.Series(x).kurtosis(), "num_peaks": len(peaks),
        "settling_proxy": np.sum(np.abs(x) > 0.02) / fs
    }


def build_dataset(n_per_class=600, output_path="landing_gear_data.csv"):
    rows = []
    for fault_id, name in FAULTS.items():
        for _ in range(n_per_class):
            _, x, pressure, zeta, wn = simulate_one(fault_id)
            features = extract_features(x, pressure, zeta, wn)
            features["label"] = fault_id
            features["fault_name"] = name
            rows.append(features)
    df = pd.DataFrame(rows).sample(frac=1, random_state=42).reset_index(drop=True)
    df.to_csv(output_path, index=False)
    print(f"Saved {len(df)} samples to {output_path}")
    return df


if __name__ == "__main__":
    build_dataset()
