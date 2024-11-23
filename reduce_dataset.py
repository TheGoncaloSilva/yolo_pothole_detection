import os
import shutil

def create_reduced_dataset(src_folder, dest_folder, num_train, num_test, num_valid):
    """
    Creates a reduced version of the dataset with the specified number of images for train, test, and validation splits.

    Args:
        src_folder (str): Path to the source dataset folder.
        dest_folder (str): Path to the destination folder for the reduced dataset.
        num_train (int): Number of images to include in the training split.
        num_test (int): Number of images to include in the testing split.
        num_valid (int): Number of images to include in the validation split.

    The function ensures the directory structure and file correspondence between images and labels is preserved.
    """

    def copy_images_and_labels(src_images, src_labels, dest_images, dest_labels, num):
        """
        Copies a specified number of images and corresponding labels from the source to the destination directory.

        Args:
            src_images (str): Path to the source images folder.
            src_labels (str): Path to the source labels folder.
            dest_images (str): Path to the destination images folder.
            dest_labels (str): Path to the destination labels folder.
            num (int): Number of files to copy.
        """
        # List all images and labels, ensuring they are sorted for consistent mapping
        images = sorted(os.listdir(src_images))
        labels = sorted(os.listdir(src_labels))

        # Select exactly 'num' images and their corresponding labels (first 'num' after sorting)
        selected_images = images[:num]
        selected_labels = labels[:num]

        # Copy selected images and their corresponding labels
        for img, lbl in zip(selected_images, selected_labels):
            img_src = os.path.join(src_images, img)
            lbl_src = os.path.join(src_labels, lbl)
            img_dest = os.path.join(dest_images, img)
            lbl_dest = os.path.join(dest_labels, lbl)
            
            # Copy image and label files to the destination
            shutil.copy(img_src, img_dest)
            shutil.copy(lbl_src, lbl_dest)

    # Iterate over each dataset split (train, test, valid)
    for split, num_images in [("train", num_train), ("test", num_test), ("valid", num_valid)]:
        # Define source and destination paths for images and labels
        src_images = os.path.join(src_folder, split, "images")
        src_labels = os.path.join(src_folder, split, "labels")
        dest_images = os.path.join(dest_folder, split, "images")
        dest_labels = os.path.join(dest_folder, split, "labels")

        # Create destination directories if they don't exist
        os.makedirs(dest_images, exist_ok=True)
        os.makedirs(dest_labels, exist_ok=True)

        # Copy the desired number of images and labels
        copy_images_and_labels(src_images, src_labels, dest_images, dest_labels, num_images)

    # Copy additional metadata files from the source to the destination
    for file in ["data.yaml", "README.dataset.txt", "README.roboflow.txt"]:
        src_file = os.path.join(src_folder, file)
        dest_file = os.path.join(dest_folder, file)
        if os.path.exists(src_file):
            shutil.copy(src_file, dest_file)

if __name__ == "__main__":
    """
    Main function to execute the script.
    Modify the source folder, destination folder, and desired number of images for each split.
    """

    # Path to the source dataset
    src_dataset_folder = "dataset"

    # Path to the destination reduced dataset
    dest_dataset_folder = "reduced_dataset"

    # Desired number of images in each split
    num_train_images = 2400  # Number of training images
    num_test_images = 300    # Number of testing images
    num_valid_images = 300   # Number of validation images

    # Call the function to create the reduced dataset
    create_reduced_dataset(src_dataset_folder, dest_dataset_folder, num_train_images, num_test_images, num_valid_images)
