import torch
from torchvision import datasets, transforms
import random
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import matplotlib.pyplot as plt

def data_generator(root, batch_size,returnSet=False):
    train_set = datasets.MNIST(root=root, train=True, download=True,
                               transform=transforms.Compose([
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.1307,), (0.3081,))
                               ]))
    test_set = datasets.MNIST(root=root, train=False, download=True,
                              transform=transforms.Compose([
                                  transforms.ToTensor(),
                                  transforms.Normalize((0.1307,), (0.3081,))
                              ]))

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size)
    if(returnSet):
      return train_set, test_set
    return train_loader, test_loader

def mix(i, j, dataset):
    """ i,j: indices (no order swap in this function)
    dataset: train or test dataset to be mixed
    return: mixed image such that even column comes from i and odd column comes from j
    """
    ret = torch.zeros_like(dataset.data[i])
    ret[:, torch.arange(0, 28, 2)] = dataset.data[i][:, torch.arange(0, 28, 2)]
    ret[:, torch.arange(0, 28, 2) + 1] = dataset.data[j][:, torch.arange(0, 28, 2) + 1]

    return ret


def get_mixed_data(batch_size):
    mnist_train_data = datasets.MNIST('./data', train=True, download=True, transform=transforms.Compose([
                                                                               transforms.ToTensor(),
                                                                               transforms.Normalize((0.1307,), (0.3081,))
                                                                           ]))
    mnist_test_data = datasets.MNIST('./data', train=False, download=True, transform=transforms.Compose([
                                                                               transforms.ToTensor(),
                                                                               transforms.Normalize((0.1307,), (0.3081,))
                                                                           ]))

    # Downsample and oversample
    num_thresh = 6000

    new_train_data = []  # mixed image
    new_train_targets = []  # even column label (0 based)
    new_train_noise = []  # odd column label  (0 based)

    count = np.zeros((9,), dtype=int)

    for i in range(mnist_train_data.targets.shape[0] - 1):
        image1 = mnist_train_data.data[i]
        image2 = mnist_train_data.data[i + 1]
        label1 = mnist_train_data.targets[i]
        label2 = mnist_train_data.targets[i + 1]
        if label1 == label2:
            continue
        j = min(label1, label2)
        k = max(label1, label2)
        # Smaller label always on the even column
        if (j < 4 and count[j] < num_thresh) or j >= 4:  # down sample 0,1,2,3
            if label1 < label2:
                new_train_data.append(mix(i, i + 1, mnist_train_data))
            else:
                new_train_data.append(mix(i + 1, i, mnist_train_data))

            new_train_targets.append(j)
            new_train_noise.append(k)
            count[j] += 1

    # randomly select to oversample 4,5,6
    for i in range(4, 7):
        n = num_thresh - count[i]
        i_index = [id for id, e in enumerate(new_train_targets) if e == i]
        i_repeat = random.sample(i_index, n)
        for j in i_repeat:
            new_train_data.append(new_train_data[j])
            new_train_targets.append(new_train_targets[j])
            new_train_noise.append(new_train_noise[j])

    # repeat and truncate 7,8
    for i in range(7, 9):
        n = num_thresh - count[i]
        i_index = np.array([id for id, e in enumerate(new_train_targets) if e == i])
        i_repeat = np.tile(i_index, 4)[:n]
        for j in i_repeat:
            new_train_data.append(new_train_data[j])
            new_train_targets.append(new_train_targets[j])
            new_train_noise.append(new_train_noise[j])

    # Generating testing dataset
    new_test_data_large_label = []  # labels are always on the even column
    new_test_targets_large_label = []
    new_test_noise_large_label = []

    new_test_data_small_label = []  # labels are always on the even column
    new_test_targets_small_label = []
    new_test_noise_small_label = []

    for i in range(mnist_test_data.targets.shape[0] - 1):
        image1 = mnist_test_data.data[i]
        image2 = mnist_test_data.data[i + 1]
        label1 = mnist_test_data.targets[i]
        label2 = mnist_test_data.targets[i + 1]
        if label1 == label2:
            continue

        new_test_data_large_label.append(mix(i, i + 1, mnist_test_data))
        if label1 < label2:
            new_test_targets_small_label.append(label1)
            new_test_noise_small_label.append(label2)
        else:
            new_test_targets_large_label.append(label1)
            new_test_noise_large_label.append(label2)

        new_test_data_small_label.append(mix(i + 1, i, mnist_test_data))
        if label2 < label1:
            new_test_targets_small_label.append(label2)
            new_test_noise_small_label.append(label1)
        else:
            new_test_targets_large_label.append(label2)
            new_test_noise_large_label.append(label1)

    arr = np.array(new_train_targets)
    t = torch.from_numpy(arr)
    mnist_train = torch.utils.data.TensorDataset(torch.stack((new_train_data)).type(torch.FloatTensor), t)
    minst_test = torch.utils.data.TensorDataset(torch.stack((new_test_data_small_label)).type(torch.FloatTensor),
                                                torch.stack((new_test_targets_small_label)))
    mnist_train_noise = torch.stack((new_train_noise)).type(torch.FloatTensor)
    mnist_test_noise = torch.stack((new_test_noise_small_label)).type(torch.FloatTensor)

    mnist_train_dl = torch.utils.data.DataLoader(mnist_train, batch_size=batch_size)  # Add shuffle = true back
    mnist_test_dl = torch.utils.data.DataLoader(minst_test, batch_size=batch_size)

    return mnist_train_dl, mnist_test_dl, mnist_train_noise, mnist_test_noise


def visualize_data(dl, num, noise):
    for images, labels in dl:
        print(images.type())
        # for i in range(num):
        #     plt.imshow(images[i], cmap='gray')
        #     plt.title(f"Label(Even):{labels[i]} Noise(Odd):{noise[i]}")
        #     plt.show()
        break


def visualize(dl, num):
    for images, labels in dl:
        print(images.type())
        # for i in range(num):
        #     plt.imshow(images[i].squeeze(0), cmap='gray')
        #     plt.title("Label:{}".format(labels[i]))
        #     plt.show()
        break

if __name__ == '__main__':
    train_dl, test_dl, train_noise, test_noise = get_mixed_data(256)
    visualize_data(train_dl, 1, train_noise)
    train_dl, test_dl = data_generator("../Data/", 256)
    visualize(train_dl, 5)