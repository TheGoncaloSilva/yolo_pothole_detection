import os

def count_images_in_directories(dataset_folder):
    """
    Lists the number of images in each sub-directory (train, test, valid) of the dataset.

    Args:
        dataset_folder (str): Path to the dataset folder containing 'train', 'test', and 'valid' subdirectories.

    This function will print the number of images found in each 'images' directory
    for every dataset split (train, test, valid).
    """
    # Iterate through dataset splits (train, test, valid)
    for split in ["train", "test", "valid"]:
        images_dir = os.path.join(dataset_folder, split, "images")

        # Check if the images directory exists
        if os.path.exists(images_dir):
            # List all files in the images directory and count the number of jpg/jpeg files
            image_files = [f for f in os.listdir(images_dir) if f.lower().endswith(('.jpg', '.jpeg'))]
            print(f"{split.capitalize()} images: {len(image_files)}")
        else:
            print(f"Missing 'images' directory for split '{split}'")

if __name__ == "__main__":
    """
    Main function to execute the script.
    Modify the dataset folder path to point to your dataset.
    """
    # Path to the dataset
    dataset_folder = "reduced_dataset"

    # Count the images in the dataset
    count_images_in_directories(dataset_folder)
