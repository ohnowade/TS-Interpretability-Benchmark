import torch
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
from model import Transformer,TCN,LSTMWithInputCellAttention,LSTM
from utils import get_mixed_data, data_generator_random, data_generator_random_0_2
import matplotlib.pyplot as plt

def attack(model, images, labels, device, att_col=[], att_row=[], batch=False, eps=0.3,
           alpha=2 / 255, steps=10, random_start=True):
    images = images.clone().detach().to(device)
    labels_id = labels
    outputs = model(images)
    labels = torch.zeros_like(outputs)
    if batch:
        for i in range(len(labels)):
            labels[i, labels_id[i]] = 1
    else:
        labels[0, labels_id] = 1
    print(labels)

    # if self._targeted:
    #     target_labels = self._get_target_label(images, labels)

    loss = nn.CrossEntropyLoss()

    # Calculate orginal loss
    _, pred = torch.max(outputs, 1)
    print("Before attack, label is: ", pred)
    labels = labels.clone().detach().to(device)
    old_loss = loss(outputs, labels)

    adv_images = images.clone().detach()

    if random_start:
        # Starting at a uniformly random point
        adv_images = adv_images + torch.empty_like(adv_images).uniform_(eps, eps)
        adv_images = torch.clamp(adv_images, min=0, max=1).detach()

    for _ in range(steps):
        adv_images.requires_grad = True
        outputs = model(adv_images)

        # Calculate loss
        # if self._targeted:
        #     cost = -loss(outputs, target_labels)
        # else:
        #     cost = loss(outputs, labels)
        cost = loss(outputs, labels)

        # Update adversarial images
        full_grad = torch.autograd.grad(cost, adv_images,
                                        retain_graph=False, create_graph=False)[0]

        grad = torch.zeros_like(full_grad)

        if len(att_row) == 0:  # Attacking att_col
            for c in att_col:
                grad[:, c] = full_grad[:, c]
        else:  # Attacking att_col[0], att_row = [...]
            for r in att_row:
                grad[r, att_col[0]] = full_grad[r, att_col[0]]

        adv_images = adv_images.detach() + alpha * grad.sign()
        delta = torch.clamp(adv_images - images, min=-eps, max=eps)
        adv_images = torch.clamp(images + delta, min=0, max=1).detach()

    outputs = model(adv_images)
    new_loss = loss(outputs, labels)
    _, pred = torch.max(outputs, 1)
    print("After attack, label is: ", pred)

    return adv_images, new_loss - old_loss

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = torch.load(open('../Models/m_model_Transformer_NumClasses_10.pt', 'rb'), map_location=device)
    train_loader, test_loader, train_noise, test_noise = data_generator_random(256)

    images, labels = next(iter(train_loader))
    adv_images, score = attack(model, images, labels, device, att_col=[8, 10, 12])

    print(score.item())

    plt.imshow(images[0].detach().to("cpu").numpy(), cmap='gray')
    plt.title(f"Label(Even):{labels[0]}")
    plt.show()

    plt.imshow(adv_images[0].detach().to("cpu").numpy(), cmap='gray')
    plt.title(f"Label(Even):{labels[0]}")
    plt.show()

