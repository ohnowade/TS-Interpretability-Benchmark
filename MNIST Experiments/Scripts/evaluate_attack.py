from attack import find_important_joint_colunms
from argparse import ArgumentParser
from utils import data_generator_random
import torch, os

def get_images(target, num):
    train_loader, _, _, _ = data_generator_random(256, shuffle=True)
    rs = []
    for images, labels in train_loader:
        for i in range(256):
            if labels[i].item() == target:
                rs.append(images[i])
                if len(rs) == num:
                    return rs
    return rs

def main(args):
    print('Evaluation of finding {} important columns for {} images of label {}...'
          .format(args.n_cols, args.n_images, args.label))

    file = open('./evaluate_results.txt', 'a')

    file.write('Evaluation of finding {} important columns for {} images of label {}...\n'
          .format(args.n_cols, args.n_images, args.label))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = torch.load(open(os.path.join(args.model_path, 'm_model_Transformer_NumClasses_10.pt'), 'rb'), map_location=device)
    images = get_images(args.label, args.n_images)
    label = torch.tensor(args.label)
    n_cols = args.n_cols

    num_even = 0
    num_odd = 0

    for image in images:
        cols, _ = find_important_joint_colunms(model, image, label, device, n_cols)
        for c in cols:
            if c % 2 == 0:
                num_even += 1
            else:
                num_odd += 1

    file.write('Total number of even columns found: {}, total number of odd columns found: {}, accuracy: {:.2f}\n'
          .format(num_even, num_odd, num_even/(num_even+num_odd)))

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--label', type=int, default=0, help='the label of the images to be evaluated (default: 0)')
    parser.add_argument('--n_images', type=int, default=10, help='the number of images to be evaluated (default: 10)')
    parser.add_argument('--n_cols', type=int, default=8, help='the number of columns to be attacked (default: 8)')
    parser.add_argument('--model_path', type=str, default='../Models/', help='the path to the model')
    args = parser.parse_args()

    main(args)

