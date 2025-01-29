import pandas as pd

# Load the CSV data into a DataFrame
df = pd.read_csv("./results/test_basic/hf-llama-31-8.csv")  # Replace with your file path

# Clean the data (handle empty/missing values in the 'error' column)
df['error'] = df['error'].fillna('No error')

# Calculate accuracy (where result matches ground_truth)
df['accuracy'] = df['result']

# Basic statistics
print("Summary Statistics:")
print(df.describe())

causal_flag_analysis = df.groupby('ground_truth').agg({
    'accuracy': 'mean',
    'n_step': 'mean',
    'error': lambda x: (x != 'No error').sum()
}).rename(columns={'error': 'error_count'})

print("\nAnalysis by Causal Flag")
print(causal_flag_analysis)

# Group by 'structure' (direct vs. mediation) and calculate metrics
structure_analysis = df.groupby('structure').agg({
    'accuracy': 'mean',
    'n_step': 'mean',
    'error': lambda x: (x != 'No error').sum()
}).rename(columns={'error': 'error_count'})

print("\nAnalysis by Structure:")
print(structure_analysis)

# Analyze errors
error_analysis = df[df['error'] != 'No error'].groupby(['structure', 'error']).size()
print("\nError Analysis:")
print(error_analysis)

# Analyze 'n_step' by setup and structure
n_step_analysis = df.groupby(['structure', 'setup']).agg({
    'n_step': ['mean', 'median', 'min', 'max']
})
print("\nStep Count Analysis:")
print(n_step_analysis)