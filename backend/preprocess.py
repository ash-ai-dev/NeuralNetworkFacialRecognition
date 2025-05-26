import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_lfw_people
from sklearn.model_selection import train_test_split

def load_and_preprocess_lfw(image_shape=(64, 64), min_faces=70, test_size=0.25):
    # Load the full LFW dataset with people having at least 70 images,
    # resizing images based on the target image shape.
    lfw = fetch_lfw_people(min_faces_per_person=70,
                           resize=image_shape[0]/250,
                           slice_=None)

    images = lfw.images # Face images as arrays
    labels = lfw.target # Numeric class labels for each face
    target_names = lfw.target_names  # Names corresponding to class labels

    # Pick only a subset of classes to keep (first two unique classes here)
    unique_labels = np.unique(labels)
    selected_classes = unique_labels[1:5]  # Select specific class IDs

    # Create a mask to filter images and labels for the chosen classes
    mask = np.isin(labels, selected_classes)
    images = images[mask]
    labels = labels[mask]

    # Remap original class labels to 0 and 1 for easier handling
    label_map = {old: new for new, old in enumerate(selected_classes)}
    labels = np.array([label_map[label] for label in labels])
    target_names = target_names[selected_classes]

    # Normalize pixel values to [0,1] range
    images = images / 255.0
    # Add a channel dimension to images for model compatibility
    images = images[:, np.newaxis, :, :]

    # Split dataset into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        images, labels, test_size=test_size, random_state=42)

    return X_train, X_test, y_train, y_test, target_names

def describe_dataset(images, labels, target_names, n_row=3, n_col=5):
    n_samples = images.shape[0]
    h, w = images.shape[2], images.shape[3]

    print(f"Number of samples: {n_samples}")
    print(f"Image shape: {h} x {w}")
    print(f"Number of classes: {len(target_names)}")

    # Prepare titles for the plotted images using their class names
    titles = [target_names[labels[i]] for i in range(n_row * n_col)]

    # Set up the plot size and spacing
    plt.figure(figsize=(1.8 * n_col, 2.4 * n_row))
    plt.subplots_adjust(bottom=0.01, left=0.01, right=0.99, top=0.90, hspace=0.35)

    # Plot each image with its label as the title
    for i in range(n_row * n_col):
        plt.subplot(n_row, n_col, i + 1)
        plt.imshow(images[i][0], cmap=plt.cm.gray)  # [0] removes the channel dimension
        plt.title(titles[i], size=12)
        plt.xticks([])  # Hide x-axis ticks
        plt.yticks([])  # Hide y-axis ticks

    plt.show()
