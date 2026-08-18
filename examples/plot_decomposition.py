"""Fit CubeCast to a synthetic tensor and save its decomposition plot."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from cubecast import CubeCast


def make_tensor(length=144, period=24):
    rng = np.random.default_rng(7)
    time = np.arange(length)
    phase = 2 * np.pi * time / period
    seasonal = np.column_stack(
        (np.sin(phase), np.sign(np.sin(2 * phase + 0.4)))
    )
    mixing = np.array([[1.5, 0.1, -0.8], [0.1, 1.5, 0.8]])
    common = seasonal @ mixing

    values = np.empty((length, 4, 3))
    for location in range(4):
        group = 1.0 if location < 2 else -0.7
        trend = 0.018 * time[:, None] * np.array([1.0, -0.4, 0.6])
        noise = rng.normal(0.0, 0.03, size=(length, 3))
        values[:, location] = 20 + location + trend + group * common + noise
    return values


model = CubeCast(period=24, window_size=96, max_components=2)
model.fit(make_tensor())
output = Path(__file__).with_name("decomposition.png")
figure, _ = model.plot_decomposition(location=0, feature=2, path=output)
plt.close(figure)
print(output)
