import torch
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
from model import Transformer,TCN,LSTMWithInputCellAttention,LSTM
from utils import get_mixed_data, data_generator_random, data_generator_random_0_2
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def attack(model, images, labels, device, att_col=[], att_row=[], batch = False,
           eps = 0.1, alpha=2 / 255, steps=50):
    if len(images.shape) < 3:
        images = images.unsqueeze(0)
        labels = labels.unsqueeze(0)
    images = images.clone().detach().to(device)
    labels = labels.clone().detach().to(device)

    num_images = images.shape[0]

    outputs = model(images)
    # print(outputs)

    loss = nn.NLLLoss()

    # Calculate orginal loss
    _, pred = torch.max(outputs, 1)
    # print("Before attack, label is: ", pred)
    old_loss = loss(outputs, labels)

    adv_images = images.clone().detach()

    # Random Start
    # Starting at a uniformly random point
    # adv_images[:, att_col] = adv_images[:, att_col] + torch.empty_like(adv_images).uniform_(eps, eps)[:, att_col]
    # adv_images = torch.clamp(adv_images, min=0, max=1).detach()

    for _ in range(steps):
        adv_images.requires_grad = True
        outputs = model(adv_images)

        cost = loss(outputs, labels)

        # Update adversarial images
        full_grad = torch.autograd.grad(cost, adv_images,
                                        retain_graph=False, create_graph=False)[0]

        grad = torch.zeros_like(full_grad)

        for i in range(num_images):
            if len(att_row) == 0:  # Attacking att_col
                for c in att_col:
                    grad[i, :, c] = full_grad[i, :, c]
            else:  # Attacking att_col[0], att_row = [...]
                for r in att_row:
                    grad[i, r, att_col[0]] = full_grad[i, r, att_col[0]]

        adv_images = adv_images.detach() + alpha * grad.sign()
        delta = torch.clamp(adv_images - images, min=-eps, max=eps)
        adv_images = torch.clamp(images + delta, min=0, max=1).detach()

    outputs = model(adv_images)
    new_loss = loss(outputs, labels)
    _, pred = torch.max(outputs, 1)
    # print("After attack, label is: ", pred)

    return adv_images, new_loss - old_loss


def find_important_joint_colunms(model, image, label, device, n_cols):
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
                print(score)
            if score > best_score:
                best_col = col
                best_score = score
        chosen_cols.append(best_col)
        available_cols.remove(best_col)
        scores.append(best_score - prev_best_score)
        prev_best_score = best_score

    return chosen_cols, scores


def find_important_independent_colunms(model, image, label, device, n_cols):
    scores = []
    for col in range(28):
        _, score = attack(model, image, label, device, att_col=[col])
        scores.append((score.item(), col))
    scores.sort(reverse=True)
    cols = [c for _, c in scores[:n_cols]]
    best_scores = [s for s, _ in scores[:n_cols]]
    return cols, best_scores

def get_column_importance(model, image, label, device):
    scores = []
    for col in range(28):
        _, score = attack(model, image, label, device, att_col=[col])
        scores.append(score)
    return scores

def find_important_rows(model, image, label, device, cols, n_row):
    important_rows = []
    for col in cols:
        scores = []
        for row in range(28):
            _, score = attack(model, image, label, device, att_col=[col], att_row=[row])
            scores.append((score.item(), row))
        scores.sort(reverse=True)
        best_rows = [r for _, r in scores[:n_row]]
        important_rows.append(best_rows)

    return important_rows

def mark_important_features(image, label, noise, x, y, ax1, ax2):
    ax1.imshow(image, cmap='gray')
    ax1.set_title('Data Image, Label: {}, Noise:{}'.format(label, noise))
    ax2.scatter(x, y, c='red', marker='s')
    ax2.imshow(image, cmap='gray')
    ax2.set_title('Important features')

def visualize_important_columns(column_importance, ax):
    rs = np.zeros((28, 28))
    for i in range(0, 28):
        rs[:, i] = np.array([column_importance[i]] * 28)
    sns.heatmap(rs, square=True, ax=ax)


def start_attack(index, n_classes, n_cols, n_rows):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if n_classes == 2:
        model = torch.load(open('../Models/m_model_Transformer_NumClasses_2.pt', 'rb'), map_location=device)
        train_loader, test_loader, train_noise, test_noise = data_generator_random_0_2(256, shuffle=False)
    else:
        model = torch.load(open('../Models/m_model_Transformer_NumClasses_10.pt', 'rb'), map_location=device)
        train_loader, test_loader, train_noise, test_noise = data_generator_random(256, shuffle=False)

    images, labels = next(iter(train_loader))
    image = images[index]
    label = labels[index]
    noise = train_noise[index]

    col_imp = get_column_importance(model, image, label, device)
    column_importance = []
    for ci in col_imp:
        column_importance.append(ci.item())
    print('Independent Score of each column is: {}'.format(column_importance))

    cols, score = find_important_joint_colunms(model, image, label, device, n_cols)
    print('Important columns are: {}'.format(cols))
    print('With Scores: {}'.format(score))
    rows = find_important_rows(model, image, label, device, cols, n_rows)
    print('Important rows are: {}'.format(rows))

    x = []
    y = []
    for i in range(n_cols):
        col = cols[i]
        row = rows[i]
        for r in row:
            x.append(col)
            y.append(r)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
    mark_important_features(image, label if n_classes == 10 else [0, 2][label], noise, x, y, ax1, ax2)
    visualize_important_columns(column_importance, ax3)
    fig.tight_layout()

    plt.show()

if __name__ == '__main__':
    start_attack(index=0, n_classes=2, n_cols=8, n_rows=8)

