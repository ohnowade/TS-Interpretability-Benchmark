import torch
from torchvision import datasets, transforms
import random
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import matplotlib.pyplot as plt
import os

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

def mix(i,j,dataset):
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


def data_generator_random_0_2(batch_size, shuffle=True):
    if (os.path.exists('../Data/MIXED/random_2_classes_train_dataset.pt') and
        os.path.exists('../Data/MIXED/random_2_classes_test_dataset.pt') and
        os.path.exists('../Data/MIXED/random_2_classes_train_noise.pt') and
        os.path.exists('../Data/MIXED/random_2_classes_test_noise.pt')):
        train_set = torch.load(open('../Data/MIXED/random_2_classes_train_dataset.pt', 'rb'))
        test_set = torch.load(open('../Data/MIXED/random_2_classes_test_dataset.pt', 'rb'))
        train_noise = torch.load(open('../Data/MIXED/random_2_classes_train_noise.pt', 'rb'))
        test_noise = torch.load(open('../Data/MIXED/random_2_classes_test_noise.pt', 'rb'))

        train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=shuffle)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size)

        return train_loader, test_loader, train_noise, test_noise

    train_set = datasets.MNIST('./data', train=True, download=True,
                               transform=transforms.Compose([
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.1307,), (0.3081,))
                               ]))
    test_set = datasets.MNIST('./data', train=False, download=True,
                              transform=transforms.Compose([
                                  transforms.ToTensor(),
                                  transforms.Normalize((0.1307,), (0.3081,))
                              ]))

    new_train_data = []  # mixed image
    new_train_targets = []  # even column label (0 based)
    new_train_noise = []  # odd column label  (0 based)
    id1 = []
    for i in range(0, 10):
        id1.append([id for id, e in enumerate(train_set.targets) if e == i])
    repeat = 2
    # target 0
    for i in id1[0]:
        for j in [2, 3, 4, 5, 7]:
            noise_idx = random.sample(id1[j], repeat)
            for k in noise_idx:
                new_train_data.append(mix(i, k, train_set))
                new_train_targets.append(0)
                new_train_noise.append(j)
    # target 2
    for i in id1[2]:
        for j in [4, 6, 7, 8, 9]:
            noise_idx = random.sample(id1[j], repeat)
            for k in noise_idx:
                new_train_data.append(mix(i, k, train_set))
                new_train_targets.append(1)
                new_train_noise.append(j)

    new_test_data = []  # mixed image
    new_test_targets = []  # even column label (0 based)
    new_test_noise = []  # odd column label  (0 based)

    id2 = []
    for i in range(0, 10):
        id2.append([id for id, e in enumerate(test_set.targets) if e == i])
    repeat = 2
    for i in id2[0]:
        for j in [2, 3, 4, 5, 7]:
            noise_idx = random.sample(id2[j], repeat)
            for k in noise_idx:
                new_test_data.append(mix(i, k, test_set))
                new_test_targets.append(0)
                new_test_noise.append(j)
    # target 2
    for i in id2[2]:
        for j in [4, 6, 7, 8, 9]:
            noise_idx = random.sample(id2[j], repeat)
            for k in noise_idx:
                new_test_data.append(mix(i, k, test_set))
                new_test_targets.append(1)
                new_test_noise.append(j)

    new_train_data = torch.stack((new_train_data)).type(torch.FloatTensor)
    new_train_data -= new_train_data.min(1, keepdim=True)[0]
    new_train_data /= new_train_data.max(1, keepdim=True)[0]
    new_train_data = torch.nan_to_num(new_train_data, nan=0.0)

    arr = np.array(new_train_targets)
    t = torch.from_numpy(arr)
    mnist_train = torch.utils.data.TensorDataset(new_train_data, t.type(torch.LongTensor))

    new_test_data = torch.stack((new_test_data)).type(torch.FloatTensor)
    new_test_data -= new_test_data.min(1, keepdim=True)[0]
    new_test_data /= new_test_data.max(1, keepdim=True)[0]
    new_test_data = torch.nan_to_num(new_test_data, nan=0.0)

    arr2 = np.array(new_test_targets)
    t2 = torch.from_numpy(arr2)
    mnist_test = torch.utils.data.TensorDataset(new_test_data, t2.type(torch.LongTensor))

    with open('../Data/MIXED/random_2_classes_train_dataset.pt', "wb") as f:
        torch.save(mnist_train, f)
    with open('../Data/MIXED/random_2_classes_test_dataset.pt', "wb") as f:
        torch.save(mnist_test, f)
    with open('../Data/MIXED/random_2_classes_train_noise.pt', "wb") as f:
        torch.save(new_train_noise, f)
    with open('../Data/MIXED/random_2_classes_test_noise.pt', "wb") as f:
        torch.save(new_test_noise, f)

    train_loader = torch.utils.data.DataLoader(mnist_train, batch_size=batch_size, shuffle=shuffle)
    test_loader = torch.utils.data.DataLoader(mnist_test, batch_size=batch_size)

    return train_loader, test_loader, new_train_noise, new_test_noise


def data_generator_random(batch_size, shuffle=True):
    if (os.path.exists('../Data/MIXED/random_10_classes_train_dataset.pt') and
        os.path.exists('../Data/MIXED/random_10_classes_test_dataset.pt') and
        os.path.exists('../Data/MIXED/random_10_classes_train_noise.pt') and
        os.path.exists('../Data/MIXED/random_10_classes_test_noise.pt')):
        train_set = torch.load(open('../Data/MIXED/random_10_classes_train_dataset.pt', 'rb'))
        test_set = torch.load(open('../Data/MIXED/random_10_classes_test_dataset.pt', 'rb'))
        train_noise = torch.load(open('../Data/MIXED/random_10_classes_train_noise.pt', 'rb'))
        test_noise = torch.load(open('../Data/MIXED/random_10_classes_test_noise.pt', 'rb'))

        train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=shuffle)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size)

        return train_loader, test_loader, train_noise, test_noise

    train_set = datasets.MNIST('./data', train=True, download=True,
                               transform=transforms.Compose([
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.1307,), (0.3081,))
                               ]))
    test_set = datasets.MNIST('./data', train=False, download=True,
                              transform=transforms.Compose([
                                  transforms.ToTensor(),
                                  transforms.Normalize((0.1307,), (0.3081,))
                              ]))

    new_train_data = []  # mixed image
    new_train_targets = []  # even column label (0 based)
    new_train_noise = []  # odd column label  (0 based)
    id1 = []
    for i in range(0, 10):
        id1.append([id for id, e in enumerate(train_set.targets) if e == i])
    repeat = 2
    for t in range(0, 10):  # targets
        for i in id1[t]:
            for num in range(repeat):
                noise = random.randint(0, 9)
                while noise == t:
                    noise = random.randint(0, 9)
                noise_idx = random.sample(id1[noise], 1)[0]
                new_train_data.append(mix(i, noise_idx, train_set))
                new_train_targets.append(t)
        new_train_noise.append(noise)

    new_test_data = []  # mixed image
    new_test_targets = []  # even column label (0 based)
    new_test_noise = []  # odd column label  (0 based)

    id2 = []
    for i in range(0, 10):
        id2.append([id for id, e in enumerate(test_set.targets) if e == i])
    repeat = 2
    for t in range(0, 10):  # targets
        for i in id2[t]:
            for num in range(repeat):
                noise = random.randint(0, 9)
                while noise == t:
                    noise = random.randint(0, 9)
                noise_idx = random.sample(id2[noise], 1)[0]
                new_test_data.append(mix(i, noise_idx, test_set))
                new_test_targets.append(t)
                new_test_noise.append(noise)

    new_train_data = torch.stack((new_train_data)).type(torch.FloatTensor)
    new_train_data -= new_train_data.min(1, keepdim=True)[0]
    new_train_data /= new_train_data.max(1, keepdim=True)[0]
    new_train_data = torch.nan_to_num(new_train_data, nan=0.0)

    arr = np.array(new_train_targets)
    t = torch.from_numpy(arr).type(torch.LongTensor)
    mnist_train = torch.utils.data.TensorDataset(new_train_data, t)

    new_test_data = torch.stack((new_test_data)).type(torch.FloatTensor)
    new_test_data -= new_test_data.min(1, keepdim=True)[0]
    new_test_data /= new_test_data.max(1, keepdim=True)[0]
    new_test_data = torch.nan_to_num(new_test_data, nan=0.0)

    arr2 = np.array(new_test_targets)
    t2 = torch.from_numpy(arr2).type(torch.LongTensor)
    mnist_test = torch.utils.data.TensorDataset(new_test_data, t2)

    with open('../Data/MIXED/random_10_classes_train_dataset.pt', "wb") as f:
        torch.save(mnist_train, f)
    with open('../Data/MIXED/random_10_classes_test_dataset.pt', "wb") as f:
        torch.save(mnist_test, f)
    with open('../Data/MIXED/random_10_classes_train_noise.pt', "wb") as f:
        torch.save(new_train_noise, f)
    with open('../Data/MIXED/random_10_classes_test_noise.pt', "wb") as f:
        torch.save(new_test_noise, f)

    train_loader = torch.utils.data.DataLoader(mnist_train, batch_size=batch_size, shuffle=shuffle)
    test_loader = torch.utils.data.DataLoader(mnist_test, batch_size=batch_size)

    return train_loader, test_loader, new_train_noise, new_test_noise


def visualize_data(dl, num, noise):
    classes = [0, 2]
    for images, labels in dl:
        for i in range(256):
            if num == 0:
                return
            if labels[i] == 0:
                continue
            plt.imshow(images[i], cmap='gray')
            plt.title(f"Label(Even):{classes[labels[i]]} Noise(Odd):{noise[i]}")
            plt.show()
            num -= 1


def visualize(dl, num):
    for images, labels in dl:
        print(images.type())
        # for i in range(num):
        #     plt.imshow(images[i].squeeze(0), cmap='gray')
        #     plt.title("Label:{}".format(labels[i]))
        #     plt.show()
        break

if __name__ == '__main__':
    train_dl, test_dl, train_noise, test_noise = data_generator_random_0_2(256, False)
    visualize_data(train_dl, 10, train_noise)

