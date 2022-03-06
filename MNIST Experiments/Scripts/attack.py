import torch
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
from model import Transformer,TCN,LSTMWithInputCellAttention,LSTM
from utils import get_mixed_data, data_generator_random, data_generator_random_0_2
import matplotlib.pyplot as plt


def attack(model, images, labels, device, att_col=[], att_row=[], batch=False, eps=0.1,
           alpha=2 / 255, steps=50):
    images = images.clone().detach().to(device)
    labels_id = labels
    outputs = model(images)
    # print(outputs)
    # labels = torch.zeros_like(outputs)
    # if batch:
    #     for i in range(len(labels)):
    #         labels[i, labels_id[i]] = 1
    # else:
    #     labels[0, labels_id] = 1
    # print(labels)

    # if self._targeted:
    #     target_labels = self._get_target_label(images, labels)

    loss = nn.NLLLoss()

    # Calculate orginal loss
    _, pred = torch.max(outputs, 1)
    # print("Before attack, label is: ", pred)
    labels = labels.clone().detach().to(device)
    old_loss = loss(outputs, labels)

    adv_images = images.clone().detach()

    # Random Start
    # Starting at a uniformly random point
    adv_images[:, att_col] = adv_images[:, att_col] + torch.empty_like(adv_images).uniform_(eps, eps)[:, att_col]
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
    # print("After attack, label is: ", pred)

    return adv_images, new_loss - old_loss


def find_important_colunms(model, image, label, device, n_cols):
    available_cols = [i for i in range(28)]
    chosen_cols = []
    prev_best_score = 0
    scores = []
    for n_col in range(n_cols):
        best_col = None
        best_score = 0
        for col in available_cols:
            _, score = attack(model, image, label, device, att_col=chosen_cols+[col])
            score = score.item()
            if score < 0:
                print("-------------- score < 0 -----------")
            if score > best_score:
                best_col = col
                best_score = score
        chosen_cols.append(best_col)
        available_cols.remove(best_col)
        scores.append(best_score - prev_best_score)
        prev_best_score = best_score

    return chosen_cols, scores


def compare_attack_results(image, label, noise, important_features):
    x = []
    y = []
    for feature in important_features:
        x.append(feature[0])
        y.append(feature[1])

    plt.scatter(x, y, c='red', marker='s')
    plt.imshow(image, cmap='gray')
    plt.title(f"Label(Even):{label}")
    plt.show()


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = torch.load(open('../Models/m_model_Transformer_NumClasses_2.pt', 'rb'), map_location=device)
    train_loader, test_loader, train_noise, test_noise = data_generator_random_0_2(256)

    images, labels = next(iter(train_loader))
    # cols, score = find_important_colunms(model, images[2].unsqueeze(0), labels[2].unsqueeze(0), device, 14)

    # print(cols)
    # print(score)

    compare_attack_results(images[0].detach().to("cpu").numpy(), labels[0], [], [(1, 2), (2, 5), (3, 7), (10, 14), (6, 9)])

