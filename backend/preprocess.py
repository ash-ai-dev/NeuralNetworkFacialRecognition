import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_lfw_people
from sklearn.model_selection import train_test_split

def load_and_preprocess_lfw(image_shape=(64, 64), min_faces=70, test_size=0.25):
    # Set slice_=None to disable the default cropping,
    # and then adjust resize to get 64x64 from the original 250x250
    lfw = fetch_lfw_people(min_faces_per_person=min_faces,
                           resize=image_shape[0]/250, # This will be 64/250 = 0.256
                           slice_=None) # Disable default slicing

    images = lfw.images  # shape: (n_samples, h, w)
    labels = lfw.target
    target_names = lfw.target_names

    # Normalize pixel values to [0, 1]
    images = images / 255.0

    # Reshape for CNN input: (n_samples, 1, h, w) for PyTorch
    images = images[:, np.newaxis, :, :]

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        images, labels, test_size=test_size, random_state=42)

    return X_train, X_test, y_train, y_test, target_names


def describe_dataset(images, labels, target_names, n_row=3, n_col=5):
    n_samples = images.shape[0]
    h, w = images.shape[2], images.shape[3]
    print(f"Number of samples: {n_samples}")
    print(f"Image shape: {h} x {w}")
    print(f"Number of classes: {len(target_names)}")

    # Create a title for each image
    titles = [target_names[labels[i]] for i in range(n_row * n_col)]

    # Plot the gallery
    plt.figure(figsize=(1.8 * n_col, 2.4 * n_row))
    plt.subplots_adjust(bottom=0.01, left=0.01, right=0.99, top=0.90, hspace=0.35)
    for i in range(n_row * n_col):
        plt.subplot(n_row, n_col, i + 1)
        plt.imshow(images[i][0], cmap=plt.cm.gray)  # [0] to remove channel dimension
        plt.title(titles[i], size=12)
        plt.xticks(())
        plt.yticks(())
    plt.show()
    