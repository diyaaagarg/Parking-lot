from roboflow import Roboflow
import os

def main():
    # Dynamically find the project root and data folder so it works flawlessly in VS Code
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, ".."))
    data_dir = os.path.join(project_root, "data")
    
    # Create the data directory if it doesn't exist
    os.makedirs(data_dir, exist_ok=True)
    download_location = os.path.join(data_dir, "pklot_dataset")

# rf = Roboflow(api_key="DEh7nTW7HCKvP4Q6xiv8")
# project = rf.workspace("zoja-scekic").project("pklot-vsh7g")
# version = project.version(1)
# dataset = version.download("yolov11")
                
                
    print(f"Preparing to download dataset to: {download_location}")

    # Initialize Roboflow
    # IMPORTANT: Replace "YOUR_ROBOFLOW_API_KEY" with your free key from app.roboflow.com
    rf = Roboflow(api_key="DEh7nTW7HCKvP4Q6xiv8")
    
    # Connect to the public PKLot YOLO workspace
    project = rf.workspace("zoja-scekic").project("pklot-vsh7g")
    # Download specifically to our target location
    print("Downloading... This may take a few minutes depending on your internet speed.")
    version = project.version(1)
    dataset = version.download("yolov11") 

    print("\nDataset downloaded successfully!")
    print(f"Your data.yaml file is located at: {os.path.join(download_location, 'data.yaml')}")

if __name__ == "__main__":
    main()