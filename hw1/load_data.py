import numpy as np
import os
import random
import tensorflow as tf
from scipy import misc
import imageio
import matplotlib.pyplot as plt

def get_images(paths, labels, nb_samples=None, shuffle=True):
    """
    Takes a set of character folders and labels and returns paths to image files
    paired with labels.
    Args:
        paths: A list of character folders
        labels: List or numpy array of same length as paths
        nb_samples: Number of images to retrieve per character
    Returns:
        List of (label, image_path) tuples
    """
    if nb_samples is not None:
        sampler = lambda x: random.sample(x, nb_samples)
    else:
        sampler = lambda x: x
    images_labels = [(i, os.path.join(path, image))
                     for i, path in zip(labels, paths)
                     for image in sampler(os.listdir(path))]
    if shuffle:
        random.shuffle(images_labels)
    return images_labels


def image_file_to_array(filename, dim_input):
    """
    Takes an image path and returns numpy array
    Args:
        filename: Image filename
        dim_input: Flattened shape of image
    Returns:
        1 channel image
    """
    # image = misc.imread(filename)
    image = imageio.imread(filename)
    image = image.reshape([dim_input])
    image = image.astype(np.float32) / 255.0
    image = 1.0 - image
    return image


class DataGenerator(object):
    """
    Data Generator capable of generating batches of Omniglot data.
    A "class" is considered a class of omniglot digits.
    """

    def __init__(self, num_classes, num_samples_per_class, config={}):
        """
        Args:
            num_classes: Number of classes for classification (K-way)
            num_samples_per_class: num samples to generate per class in one batch
            batch_size: size of meta batch size (e.g. number of functions)
        """
        self.num_samples_per_class = num_samples_per_class
        self.num_classes = num_classes

        data_folder = config.get('data_folder', './omniglot_resized')
        self.img_size = config.get('img_size', (28, 28))

        self.dim_input = np.prod(self.img_size)
        self.dim_output = self.num_classes

        character_folders = [os.path.join(data_folder, family, character)
                             for family in os.listdir(data_folder)
                             if os.path.isdir(os.path.join(data_folder, family))
                             for character in os.listdir(os.path.join(data_folder, family))
                             if os.path.isdir(os.path.join(data_folder, family, character))]

        random.seed(1)
        random.shuffle(character_folders)
        num_val = 100
        num_train = 1100
        self.metatrain_character_folders = character_folders[: num_train]
        self.metaval_character_folders = character_folders[
            num_train:num_train + num_val]
        self.metatest_character_folders = character_folders[
            num_train + num_val:]

    def sample_batch(self, batch_type, batch_size):
        """
        Samples a batch for training, validation, or testing
        Args:
            batch_type: train/val/test
        Returns:
            A a tuple of (1) Image batch and (2) Label batch where
            image batch has shape [B, K, N, 784] and label batch has shape [B, K, N, N]
            where B is batch size, K is number of samples per class, N is number of classes
        """
        if batch_type == "train":
            folders = self.metatrain_character_folders
        elif batch_type == "val":
            folders = self.metaval_character_folders
        else:
            folders = self.metatest_character_folders

        #############################
        #### YOUR CODE GOES HERE ####

        B, K, N = batch_size, self.num_samples_per_class, self.num_classes
        all_image_batches = []
        all_label_batches = np.zeros((B, K*N, N))

        for b in range(B):
            # 1. Sample N different classes from either the specified train, test, or validation folders.
            paths = random.sample(folders, N)    # (N,)
            classes = list(range(N))             # (N,)

            # 2a. Load K images per class and collect the associated labels
            images_labels = get_images(paths, classes, K, shuffle=False) # (N*K,2)
            images_labels_sets = np.array(images_labels).reshape(N, K, 2) \
                                                        .transpose(1, 0, 2) # (K, N, 2)

            # 2b. Shuffle the order of N image classes for each (K+1) sets
            for images_labels_set in images_labels_sets:
                np.random.shuffle(images_labels_set)
            labels = images_labels_sets[:, :, 0].reshape(-1).astype(int)
            images = images_labels_sets[:, :, 1].reshape(-1)

            # 3. Format the data and return two numpy matrices, one of flattened images with shape
            #    [B;K;N; 784] and one of one-hot labels [B;K;N;N]
            image_arrays = [image_file_to_array(file, self.dim_input) for file in images]   # (K*N, 784)
            all_image_batches.append(image_arrays)
            all_label_batches[b, np.arange(K*N), labels] = 1

        all_image_batches = np.array(all_image_batches).reshape((B, K, N, self.dim_input))
        all_label_batches = np.reshape(all_label_batches, (B, K, N, N))

        #############################

        return all_image_batches, all_label_batches

if __name__ == "__main__":
    batch_size, num_classes, num_samples_per_class = 2, 3, 5
    batch_type = "train"
    generator = DataGenerator(num_classes, num_samples_per_class)

    all_image_batches, all_label_batches = generator.sample_batch(batch_type, batch_size)
    assert(all_image_batches.shape == (batch_size, num_samples_per_class, num_classes, 784))
    assert(all_label_batches.shape == (batch_size, num_samples_per_class, num_classes, num_classes))
    print(all_label_batches[0, :, :, :])   # prints K sets of N images

    def plot_images(imgs, labels, n_col=num_classes, title=None):
        plt.figure(figsize=(8, 2))
        n_row = np.ceil(len(imgs) / n_col).astype(int)
        for img_idx, (img, label) in enumerate(zip(imgs, labels)):
            plt.subplot(n_row, n_col, img_idx+1)
            plt.imshow(img)
            plt.axis('off')
            plt.title(label.argmax())

        if title:
            plt.suptitle(title)
        plt.show()

    for i in range(batch_size):
        images, labels = all_image_batches[i], all_label_batches[i]
        train_images, train_labels = images[:-1].reshape(-1, 28, 28), labels[:-1].reshape(-1, num_classes)  # K-1, N, N
        test_images, test_labels = images[-1:].reshape(-1, 28, 28), labels[-1:].reshape(-1, num_classes) # 1, N, N

        plot_images(train_images, train_labels, n_col=num_classes, title=f'Train #{i+1}')
        plot_images(test_images, test_labels, n_col=num_classes, title=f'Test #{i+1}')
