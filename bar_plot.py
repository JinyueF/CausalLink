import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load data into a pandas DataFrame
df = pd.read_csv("results/combined_structure_analysis.csv")

model_name_map = {
    "Llama-3.1-Nemotron-70B-Instruct-HF": "Llama 3.1 Nemotron 70B",
    "gemini-2.0-flash": "Gemini 2.0 Flash",
    "hf-deepseek-distill-qwen-14": "DeepSeek Distill Qwen 14",
    "hf-deepseek-distill-qwen-32-large": "DeepSeek Distill Qwen 32 Large",
    "hf-llama-31-8": "Llama 31-8",
    "hf-llama-32-3": "Llama 32-3",
    "hf-mistral-7": "Mistral 7",
    "hf-qwen-25-14": "Qwen 25-14",
    "hf-qwen-25-3": "Qwen 25-3",
    "hf-qwen-25-32-large": "Qwen 25-32 Large",
    "openai_gpt-4o": "GPT-4o",
    "openai_gpt-4o-mini": "GPT-4o Mini"
}

# Replace model names
df["model"] = df["model"].map(model_name_map)

# Get unique models and split into three roughly equal parts
models = df["model"].unique()
num_rows = 3
models_split = [models[i::num_rows] for i in range(num_rows)]

# Set up subplots
fig, axes = plt.subplots(num_rows, 1, figsize=(14, 12), sharey=True)

# Plot each subset
for i, ax in enumerate(axes):
    sns.barplot(data=df[df["model"].isin(models_split[i])], x="model", y="accuracy", hue="structure", palette="muted", ax=ax)
    ax.set_xticklabels(ax.get_xticklabels(), ha="right")
    ax.set_xlabel("")
    ax.set_ylabel("Accuracy" if i == 1 else "")

# Adjust layout and add common labels
fig.suptitle("Comparison of Model Accuracy by Causal Structure", fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.show()