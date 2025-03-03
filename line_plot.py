import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

x = np.array([2,3,4,5,6,7])
y_gpt = np.array([1.0, 0.833, 0.604, 0.547, 0.510, 0.447])
y_gemini = np.array([1.0, 0.667, 0.525, 0.463, 0.406, 0.32])

plt.rcParams["font.family"] = "times"
plt.rcParams["font.size"] = 18

plt.figure(figsize=(8, 5))
plt.plot(x, y_gpt, marker='o', linestyle='-', label='GPT 4o', color=plt.cm.tab20(1))
plt.plot(x, y_gemini, marker='s', linestyle='--', label='Gemini 2.0 Flash', color=plt.cm.tab20(3))

plt.xlabel('Number of Variables')
plt.ylabel('Accuracy')
plt.xticks(x)  # Ensure all x-ticks are shown
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()