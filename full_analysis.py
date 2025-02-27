import os
import pandas as pd
import numpy as np
from pathlib import Path

def process_results_directory(results_dir):
    """Process all CSV files in a results directory and combine into a single DataFrame"""
    all_dfs = []
    
    # Iterate through all CSV files in directory
    for file_path in Path(results_dir).glob("*.csv"):
        # Extract model name from filename
        model_name = 'gemini-2.0-flash'
        
        # Load and process data
        df = pd.read_csv(file_path)
        df['model'] = model_name
        
        # Data cleaning
        df['error'] = df['error'].fillna('No error')
        df['result'] = df['result'].fillna(False)
        df['accuracy'] = df['result'].astype(int)  # Convert boolean to 1/0
        
        all_dfs.append(df)
    
    # Combine all DataFrames
    combined_df = pd.concat(all_dfs, ignore_index=True)
    
    return combined_df

def analyze_combined_results(combined_df):
    """Run comprehensive analysis on combined DataFrame"""
    analysis_results = {}
    
    # Basic statistics
    analysis_results['summary_stats'] = combined_df.describe()
    
    # Model-level analysis
    model_analysis = combined_df.groupby('setup').agg({
        'accuracy': ['mean'],
        'n_step': ['mean'],
        'error': lambda x: (x != 'No error').sum()
    }).reset_index()
    
    # Structure-level analysis
    def std(x): return np.std(x, ddof=1) / np.sqrt(np.size(x))
    _structure_analysis = combined_df.groupby(['model', 'structure', 'setup'], as_index=False).agg({
        'accuracy': ['mean']
    }).reset_index()

    print(_structure_analysis)

    structure_analysis = combined_df.groupby(['model', 'structure']).agg({
        'accuracy': ['mean', std]
    }).reset_index()
    structure_analyses_std = _structure_analysis.groupby(['model', 'structure']).agg({('accuracy', 'mean'): ['mean',min,max,std]}).reset_index()


    # Error analysis
    error_analysis = combined_df[combined_df['error'] != 'No error']
    error_analysis = error_analysis.groupby(['model', 'structure', 'error']).size().reset_index(name='count')
    
    # Causal flag analysis
    causal_flag_analysis = combined_df.groupby(['model', 'ground_truth']).agg({
        'accuracy': 'mean',
        'n_step': 'mean',
        'error': lambda x: (x != 'No error').sum()
    }).reset_index()
    
    return {
        'combined_data': combined_df,
        'model_analysis': model_analysis,
        'structure_analysis': structure_analysis,
        'structure_analysis_std': structure_analyses_std,
        'error_analysis': error_analysis,
        'causal_flag_analysis': causal_flag_analysis
    }

def save_analysis_results(results, output_dir):
    """Save analysis results to CSV files"""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    results['combined_data'].to_csv(Path(output_dir)/"combined_results.csv", index=False)
    results['model_analysis'].to_csv(Path(output_dir)/"model_analysis.csv", index=False)
    results['structure_analysis'].to_csv(Path(output_dir)/"structure_analysis.csv", index=False)
    results['structure_analysis_std'].to_csv(Path(output_dir)/"structure_analysis_std.csv", index=False)
    results['error_analysis'].to_csv(Path(output_dir)/"error_analysis.csv", index=False)
    results['causal_flag_analysis'].to_csv(Path(output_dir)/"causal_flag_analysis.csv", index=False)

if __name__ == "__main__":
    # Configuration
    INPUT_DIR = "./results/advance_hard_4_5_basic_limit_step"
    OUTPUT_DIR = "./results/analyzed_advance_hard_limit_step_gemini/"
    
    # Process data
    combined_df = process_results_directory(INPUT_DIR)
    analysis = analyze_combined_results(combined_df)
    
    # Save results
    save_analysis_results(analysis, OUTPUT_DIR)
    
    print(f"Analysis complete. Results saved to {OUTPUT_DIR}")