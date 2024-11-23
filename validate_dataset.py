import os

def check_image_label_correspondence(dataset_folder):
    """
    Checks that every image file in the dataset has a corresponding label file.

    Args:
        dataset_folder (str): Path to the dataset folder.

    Returns:
        dict: A dictionary summarizing missing files for each dataset split.
    """
    # Dataset splits to check
    splits = ["train", "test", "valid"]

    # File mismatch summary
    mismatches = {}

    for split in splits:
        # Define paths for images and labels
        images_path = os.path.join(dataset_folder, split, "images")
        labels_path = os.path.join(dataset_folder, split, "labels")

        # Skip the split if the directories don't exist
        if not os.path.exists(images_path) or not os.path.exists(labels_path):
            print(f"Warning: {split} split is missing required directories.")
            continue

        # Get lists of image and label files (strip extensions for comparison)
        image_files = {os.path.splitext(f)[0] for f in os.listdir(images_path) if f.endswith(('.jpg', '.png', '.jpeg'))}
        label_files = {os.path.splitext(f)[0] for f in os.listdir(labels_path) if f.endswith('.txt')}

        # Find images without labels and labels without images
        missing_labels = image_files - label_files
        missing_images = label_files - image_files

        # Print results
        if missing_labels:
            print(f"\n{split.capitalize()} Images Without Labels:")
            for img in missing_labels:
                print(f"  - {img}")

        if missing_images:
            print(f"\n{split.capitalize()} Labels Without Images:")
            for lbl in missing_images:
                print(f"  - {lbl}")

        # Save mismatches in the summary
        mismatches[split] = {
            "missing_labels": list(missing_labels),
            "missing_images": list(missing_images),
        }

    return mismatches


if __name__ == "__main__":
    """
    Main function to execute the script.
    Specify the dataset folder and check correspondence.
    """

    # Path to the dataset folder
    dataset_folder = "reduced_dataset"

    # Check for correspondence and get summary of mismatches
    mismatches_summary = check_image_label_correspondence(dataset_folder)

    # Summary output
    print("\nValidation Complete!")
    if any(mismatches_summary[split]["missing_labels"] or mismatches_summary[split]["missing_images"] for split in mismatches_summary):
        print("\nIssues found:")
        for split, mismatches in mismatches_summary.items():
            if mismatches["missing_labels"]:
                print(f"  - {split.capitalize()}: {len(mismatches['missing_labels'])} images without labels.")
            if mismatches["missing_images"]:
                print(f"  - {split.capitalize()}: {len(mismatches['missing_images'])} labels without images.")
    else:
        print("\nAll images have corresponding labels, and vice versa.")
