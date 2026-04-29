import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load the data
df = pd.read_csv("nd_view_panicle_pred_hw5.csv")

models = ["Y7", "Y8", "Y9", "FasterR-CNN"]
gt = df["GT_Panicle"]

# Compute error statistics for each model
results = []
for model in models:
    preds = df[model]
    mae = mean_absolute_error(gt, preds)
    rmse = np.sqrt(mean_squared_error(gt, preds))
    bias = np.mean(preds - gt)
    r2 = r2_score(gt, preds)
    results.append({"Model": model, "MAE": round(mae, 2), "RMSE": round(rmse, 2), "Bias": round(bias, 2), "R2": round(r2, 4)})

results_df = pd.DataFrame(results)
print("\n=== Error Statistics ===")
print(results_df.to_string(index=False))
results_df.to_csv("error_statistics.csv", index=False)

# Plot 1: Bar chart of MAE and RMSE
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("Model Error Statistics", fontsize=14, fontweight="bold")

axes[0].bar(results_df["Model"], results_df["MAE"], color=["#2196F3", "#4CAF50", "#FF9800", "#E91E63"])
axes[0].set_title("Mean Absolute Error (MAE)\nLower is better")
axes[0].set_ylabel("MAE")
for i, v in enumerate(results_df["MAE"]):
    axes[0].text(i, v + 0.3, str(v), ha="center", fontweight="bold")

axes[1].bar(results_df["Model"], results_df["RMSE"], color=["#2196F3", "#4CAF50", "#FF9800", "#E91E63"])
axes[1].set_title("Root Mean Square Error (RMSE)\nLower is better")
axes[1].set_ylabel("RMSE")
for i, v in enumerate(results_df["RMSE"]):
    axes[1].text(i, v + 0.3, str(v), ha="center", fontweight="bold")

plt.tight_layout()
plt.savefig("error_bar_charts.png", dpi=150)
plt.show()

# Plot 2: Scatter plots of predicted vs ground truth
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle("Predicted vs Ground Truth Panicle Counts", fontsize=14, fontweight="bold")
colors = ["#2196F3", "#4CAF50", "#FF9800", "#E91E63"]

for i, (model, color) in enumerate(zip(models, colors)):
    ax = axes[i // 2][i % 2]
    ax.scatter(gt, df[model], alpha=0.6, color=color)
    min_val = min(gt.min(), df[model].min())
    max_val = max(gt.max(), df[model].max())
    ax.plot([min_val, max_val], [min_val, max_val], "k--", label="Perfect prediction")
    ax.set_xlabel("Ground Truth")
    ax.set_ylabel("Predicted")
    ax.set_title(f"{model} (R²={results_df[results_df['Model']==model]['R2'].values[0]})")
    ax.legend()

plt.tight_layout()
plt.savefig("scatter_plots.png", dpi=150)
plt.show()

# Plot 3: Bias chart
fig, ax = plt.subplots(figsize=(8, 5))
colors_bias = ["red" if b > 0 else "blue" for b in results_df["Bias"]]
ax.bar(results_df["Model"], results_df["Bias"], color=colors_bias)
ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
ax.set_title("Bias per Model\nPositive = Overcounting, Negative = Undercounting")
ax.set_ylabel("Bias")
for i, v in enumerate(results_df["Bias"]):
    ax.text(i, v + (0.3 if v >= 0 else -1.5), str(v), ha="center", fontweight="bold")
plt.tight_layout()
plt.savefig("bias_chart.png", dpi=150)
plt.show()

print("\nDone! Saved: error_statistics.csv, error_bar_charts.png, scatter_plots.png, bias_chart.png")