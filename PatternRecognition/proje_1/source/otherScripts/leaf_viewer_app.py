import streamlit as st
import pandas as pd
import os
from PIL import Image
import io

# Set page configuration
st.set_page_config(page_title="Leaf Image Viewer", layout="wide")

def load_excel_data():
    """Load data from Excel file"""
    excel_path = "source/leaves.xlsx"
    if os.path.exists(excel_path):
        df = pd.read_excel(excel_path)
        # Add status column if it doesn't exist
        if 'status' not in df.columns:
            df['status'] = 0
        return df
    else:
        st.error(f"Excel file not found at {excel_path}. Please run the fill_excel.py script first.")
        return pd.DataFrame()

def save_excel_data(df):
    """Save data back to Excel file"""
    excel_path = "source/leaves.xlsx"
    df.to_excel(excel_path, index=False)
    st.success("Excel file updated successfully!")

def display_image(image_path):
    """Display the image from the given path"""
    if os.path.exists(image_path):
        try:
            image = Image.open(image_path)
            return image
        except Exception as e:
            st.error(f"Error loading image: {e}")
            return None
    else:
        st.error(f"Image not found: {image_path}")
        return None

def main():
    st.title("Leaf Image Viewer and Status Editor")
    
    # Load data from Excel
    df = load_excel_data()
    
    if df.empty:
        return
    
    # Initialize session state for selected row if it doesn't exist
    if 'selected_row_index' not in st.session_state:
        st.session_state.selected_row_index = 0
    if 'view_option' not in st.session_state:
        st.session_state.view_option = "Original Image"
    
    # Create two columns for layout
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.subheader("Image Data")
        
        # Simple row selector
        selected_row = st.number_input("Select row to view:", min_value=0, max_value=len(df)-1, value=st.session_state.selected_row_index)
        if st.button("View Selected Row"):
            st.session_state.selected_row_index = selected_row
            st.rerun()
        
        # Use data editor to allow editing the status column
        edited_df = st.data_editor(
            df,
            column_config={
                "image_path": st.column_config.TextColumn(
                    "Image Path",
                    help="Path to the original image",
                    width="medium",
                    disabled=True,
                ),
                "filtered_image_path": st.column_config.TextColumn(
                    "Filtered Image Path",
                    help="Path to the processed image",
                    width="medium",
                    disabled=True,
                ),
                "status": st.column_config.SelectboxColumn(
                    "Status",
                    help="1=Valid, 0=Invalid",
                    width="small",
                    options=[0, 1],
                    default=0
                )
            },
            hide_index=True,
            num_rows="dynamic",
            use_container_width=True,
            disabled=["image_path", "filtered_image_path"],
        )
        
        # Save button
        if st.button("Save Changes"):
            save_excel_data(edited_df)
    
    # Add functionality to view selected image
    with col2:
        st.subheader("Image Viewer")
        
        # View option radio buttons
        view_option = st.radio(
            "View Option", 
            ["Original Image", "Processed Image"], 
            horizontal=True
        )
        st.session_state.view_option = view_option
        
        # Get the selected row index
        selected_idx = st.session_state.selected_row_index
        
        if not df.empty and selected_idx < len(df):
            selected_row = df.iloc[selected_idx]
            
            # Display the selected row number
            st.write(f"**Selected Row:** {selected_idx}")
            
            # Determine which image to show
            if st.session_state.view_option == "Original Image":
                image_path = selected_row["image_path"]
                st.write(f"**Original Image Path:** {image_path}")
            else:
                image_path = selected_row["filtered_image_path"]
                st.write(f"**Processed Image Path:** {image_path}")
            
            # Check if file exists before attempting to display
            image = display_image(image_path)
            if image:
                st.image(image, caption=os.path.basename(image_path), use_column_width=True)
                
                # Display current status
                current_status = selected_row["status"]
                st.write(f"**Current Status:** {'Valid (1)' if current_status == 1 else 'Invalid (0)'}")
                
                # Quick status update for this image
                new_status = st.radio("Update Status:", [0, 1], horizontal=True, index=int(current_status))
                
                if new_status != current_status and st.button(f"Update Status to {new_status}"):
                    df.at[selected_idx, "status"] = new_status
                    save_excel_data(df)
                    st.success(f"Status updated to {new_status}")
                    st.rerun()
        else:
            st.write("No row selected or data is empty.")

    # Add some statistics
    if not df.empty:
        st.sidebar.subheader("Dataset Statistics")
        total_images = len(df)
        valid_images = len(df[df["status"] == 1])
        invalid_images = len(df[df["status"] == 0])
        
        st.sidebar.write(f"Total Images: {total_images}")
        st.sidebar.write(f"Valid Images (Status=1): {valid_images}")
        st.sidebar.write(f"Invalid Images (Status=0): {invalid_images}")
        
        # Progress bar showing completion status
        progress = valid_images + invalid_images
        if total_images > 0:
            progress_pct = progress / total_images
            st.sidebar.progress(progress_pct)
            st.sidebar.write(f"Progress: {progress}/{total_images} images processed ({progress_pct:.1%})")

if __name__ == "__main__":
    main()
