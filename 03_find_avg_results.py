import pandas as pd
import os

def average_llm_performance(dir1, dir2, dir3, output_dir):
    # Define file paths
    file_name = "LLM_Performance.csv"
    paths = [os.path.join(d, file_name) for d in [dir1, dir2, dir3]]
    
    # Read dataframes
    dfs = [pd.read_csv(path) for path in paths]
    
    # Ensure all dataframes have the same shape
    if not all(df.shape == dfs[0].shape for df in dfs):
        raise ValueError("All input CSV files must have the same shape")
    
    # Keep the first column and first row as is
    avg_df = dfs[0].copy()
    
    # Compute the average for numerical cells
    avg_df.iloc[1:, 1:] = (sum(df.iloc[1:, 1:] for df in dfs) / len(dfs)).round(2)
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the resulting dataframe
    output_path = os.path.join(output_dir, "LLM_Performance.csv")
    avg_df.to_csv(output_path, index=False)
    
    print(f"Averaged file saved at: {output_path}")
    return avg_df


dir1_path="/mnt/data1/raiyan/breast_cancer/VLMs-for-Mammograms/results_1"
dir2_path="/mnt/data1/raiyan/breast_cancer/VLMs-for-Mammograms/results_2"
dir3_path="/mnt/data1/raiyan/breast_cancer/VLMs-for-Mammograms/results_3"
output_dir_path="/mnt/data1/raiyan/breast_cancer/VLMs-for-Mammograms/results_averaged"


average_llm_performance(dir1_path, dir2_path, dir3_path, output_dir_path)
