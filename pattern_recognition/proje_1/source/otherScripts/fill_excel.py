import os
import pandas as pd
from pathlib import Path

def main():
    # Path to the Excel file
    excel_path = "source/leaves.xlsx"
    
    # Path to the leaves directory
    leaves_dir = "leaves"
    
    # Create or load the Excel file
    try:
        df = pd.read_excel(excel_path)
        # If the file exists but doesn't have the required columns, add them
        if 'image_path' not in df.columns:
            df['image_path'] = None
        if 'filtered_image_path' not in df.columns:
            df['filtered_image_path'] = None
    except FileNotFoundError:
        df = pd.DataFrame(columns=['image_path', 'filtered_image_path'])
    
    df = df.iloc[0:0]
    
    image_paths = []
    for root, _, files in os.walk(leaves_dir):
        class_name = os.path.basename(root)
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp', '.gif')):
                file_path = os.path.join(root, file)
                file_path = file_path.replace('\\', '/')
                processed_path = file_path.replace('leaves', 'processed_images')
                processed_path = processed_path.replace('.jpg', '_processed.png')
                # Add to the list
                image_paths.append({
                    'image_path': file_path,
                    'filtered_image_path': processed_path,
                    'class': class_name,
                    'file_name': file,
                    'status': "0"
                })
    
    # Create a new DataFrame with the collected paths
    new_df = pd.DataFrame(image_paths)
    
    # Combine with the existing DataFrame
    df = pd.concat([df, new_df], ignore_index=True)
    
    # Ensure the directory exists
    os.makedirs(os.path.dirname(excel_path), exist_ok=True)
    
    # Save the DataFrame to Excel
    df.to_excel(excel_path, index=False)
    
    print(f"Excel file updated at {excel_path} with {len(df)} image entries.")

if __name__ == "__main__":
    main()
