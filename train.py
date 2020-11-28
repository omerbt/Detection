import matplotlib.pyplot as plt
import argparse
import os

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

from net12 import Net as Net12
from net24 import Net as Net24
from net12FCN import NetFCN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser(description='')
parser.add_argument('--model', type=str, default='12FCN', help='model to train [12net, 12FCN, 24net]')
parser.add_argument('--batch', type=int, default=256,
                    help='input batch size for training (default: 256)')
parser.add_argument('--epochs', type=int, default=300,
                    help='number of epochs to train (default: 300)')
parser.add_argument('--learning_rate', type=float, default=1 * 1e-4,
                    help='learning rate for model training (default: 1*1e-4)')
parser.add_argument('--seed', type=int, default=42,
                    help='random seed (default: 42). [ignored]')
parser.add_argument('--interval', type=int, default=20,
                    help='interval for saving checkpoint (default: 20)')
parser.add_argument('--output_dir', type=str, default='12-net_results',
                    help='Where to save model')
parser.add_argument('--face_data', type=str, default='data/train_12.pt')
parser.add_argument('--nonface_data', type=str, default='data/patches_12_new.pt')

args = parser.parse_args()
if not os.path.isfile(args.face_data):
    raise ValueError(f'Invalid value for face_data. {args.face_data}'
                     f' is not a file.')
if not os.path.isfile(args.nonface_data):
    raise ValueError(f'Invalid value for nonface_data. {args.nonface_data}'
                     f' is not a file.')

if args.model not in ['12net', '12FCN', '24net']:
    raise ValueError('Invalid value for model. Model must be one of: 12net, 12FCN, 24net')

if not os.path.isdir(args.output_dir):
    os.mkdir(args.output_dir)

if args.model == '12FCN':
    model = NetFCN().to(device)
elif args.model == '12net':
    model = Net12().to(device)
elif args.model == '24net':
    model = Net24().to(device)

optimizer = torch.optim.Adam(list(model.parameters()), lr=args.learning_rate)
criterion = torch.nn.CrossEntropyLoss()


class dataset(Dataset):
    def __init__(self, faces_path, nonfaces_path):
        faces = torch.load(faces_path).float()
        non_faces = torch.load(nonfaces_path).float()
        self.x = torch.cat((faces, non_faces), 0)
        positive = torch.ones(faces.shape[0], dtype=torch.long)
        negative = torch.zeros(non_faces.shape[0], dtype=torch.long)
        self.y = torch.cat((positive, negative), 0)
        self.length = self.x.shape[0]

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

    def __len__(self):
        return self.length


def train_val_split(dataset, train_val_ratio):
    indices = np.arange(len(dataset))
    np.random.shuffle(indices)
    split = int(np.floor(train_val_ratio * len(dataset)))
    train_indices, val_indices = indices[split:], indices[:split]

    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)

    train_loader = DataLoader(dataset, batch_size=args.batch, sampler=train_sampler)
    val_loader = DataLoader(dataset, batch_size=args.batch, sampler=val_sampler)
    return train_loader, val_loader


dataset = dataset(args.face_data, args.nonface_data)
train_loader, val_loader = train_val_split(dataset, 0.8)
epochs = args.epochs
train_loss, val_loss, train_accuracy, val_accuracy = [], [], [], []

print('starting training...')
print(device.type, 'detected')
for epoch in range(1, epochs + 1):
    for run in ['val', 'train']:
        if run == 'val':
            model.eval()
            loader = train_loader
            loss_list = train_loss
            accuracy_list = train_accuracy
        else:
            model.training = True
            loader = val_loader
            loss_list = val_loss
            accuracy_list = val_accuracy
        epoch_loss = 0.0
        hits = 0.0
        cnt = 0
        total = 0.0
        for data, labels in loader:
            data, labels = data.to(device), labels.to(device)
            if run == 'val':
                with torch.no_grad():
                    outputs = model(data)
            else:
                optimizer.zero_grad()
                outputs = model(data)
            if args.model == '12FCN':
                outputs = outputs.squeeze()
            _, preds = torch.max(outputs.data, 1)
            hits += torch.sum(preds == labels.data)
            loss = criterion(outputs, labels)
            if run == 'train':
                loss.backward()
                optimizer.step()
            epoch_loss += loss.item()
            cnt += 1
            total += labels.size(0)

        loss_list.append(epoch_loss / cnt)
        accuracy_list.append(hits / total)

    print('Epoch: {} \t Loss: [Train: {:.6f} \t Val: {:.6f}] \t Accuracy: [Train: {:.6f} \t Val: {:.6f}]'.format(
        epoch, train_loss[-1], val_loss[-1], train_accuracy[-1], val_accuracy[-1]))

    if epoch % args.interval == 0 or epoch == epochs:
        torch.save({
            'device': device.type,
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss[-1],
            'val_loss': val_loss[-1]
        }, os.path.join(args.output_dir, f'{args.model}_{epoch}.pt'))

    plt.plot(train_loss, label='train')
    plt.plot(val_loss, label='validation')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(args.output_dir, 'loss.png'))
