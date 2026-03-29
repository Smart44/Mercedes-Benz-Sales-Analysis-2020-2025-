import pandas as pd
import matplotlib.pyplot as plt
import os
import time

# Start timer to measure execution time
start_time = time.time()

# --- Directory & Path Configuration ---

# Get the absolute path of the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# Move up one level to the base project directory
base_dir = os.path.dirname(script_dir)

# Define paths for data input and output folders
CSV_PATH = os.path.join(base_dir, 'data', 'mercedes_benz_sales_2020_2025.csv')
OUTPUT_DIR = os.path.join(base_dir, 'output')

# Create the output directory if it doesn't exist
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

def load_data():
    """Reads CSV file from the data folder and cleans column names."""
    print(f"⏳ Searching for file at: {CSV_PATH}")
    
    if not os.path.exists(CSV_PATH):
        print(f"❌ Error: File not found in 'data' folder.")
        return None
    
    # Load the dataset
    df = pd.read_csv(CSV_PATH)
    
    # Clean column names: replace spaces with underscores and convert to lowercase
    df.columns = [c.replace(' ', '_').lower() for c in df.columns]
    return df

# 1. Load data
df = load_data()

if df is not None:
    print("✅ Data loaded successfully.")

    # 2. Data Analysis: Aggregating sales volume
    # Top 10 models by sales volume
    top_models = df.groupby('model')['sales_volume'].sum().sort_values(ascending=False).head(10)
    
    # Sales distribution by fuel type
    fuel_sales = df.groupby('fuel_type')['sales_volume'].sum()

    # 3. Visualization: Generating and saving charts to 'output' folder
    print(f"📊 Saving results to: {OUTPUT_DIR}...")

    # Chart 1: Bar plot for Top 10 Models
    plt.figure(figsize=(10, 6))
    top_models.plot(kind='bar', color='skyblue')
    plt.title('Top 10 Mercedes-Benz Models by Sales Volume')
    plt.ylabel('Total Sales')
    plt.xlabel('Model')
    plt.savefig(os.path.join(OUTPUT_DIR, 'top_models.png'), bbox_inches='tight')
    plt.close()

    # Chart 2: Pie chart for Fuel Type distribution
    plt.figure(figsize=(8, 8))
    fuel_sales.plot(kind='pie', autopct='%1.1f%%', startangle=140)
    plt.title('Sales Distribution by Fuel Type')
    plt.ylabel('')  # Remove y-axis label for better aesthetics
    plt.savefig(os.path.join(OUTPUT_DIR, 'fuel_analysis.png'))
    plt.close()

    print(f"🎉 Analysis complete. Charts saved successfully.")
else:
    print("Execution halted due to missing data.")

# Final execution time
print(f"⏱️ Total duration: {time.time() - start_time:.2f} seconds.")