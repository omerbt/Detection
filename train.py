import matplotlib.pyplot as plt
import argparse
import os

import torch
from torch.utils.data import Dataset, DataLoader

from net12 import Net
from net12FCN import NetFCN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser(description='12-Net')
parser.add_argument('--model', type=str, default='FCN')
parser.add_argument('--batch', type=int, default=256,
                    help='input batch size for training (default: 256)')
parser.add_argument('--epochs', type=int, default=300,
                    help='number of epochs to train (default: 10000)')
parser.add_argument('--learning_rate', type=float, default=1 * 1e-5,
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

if not os.path.isdir(args.output_dir):
    os.mkdir(args.output_dir)

if args.model == 'FCN':
    model = NetFCN().to(device)
else:
    model = Net().to(device)

optimizer = torch.optim.Adam(list(model.parameters()), lr=args.learning_rate)
criterion = torch.nn.CrossEntropyLoss()


class dataset(Dataset):
    def __init__(self, faces_path, nonfaces_path):
        faces = torch.load(faces_path)
        non_faces = torch.load(nonfaces_path)
        self.x = torch.cat((faces, non_faces), 0)
        positive = torch.ones(faces.shape[0], dtype=torch.long)
        negative = torch.zeros(non_faces.shape[0], dtype=torch.long)
        self.y = torch.cat((positive, negative), 0)
        self.length = self.x.shape[0]

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

    def __len__(self):
        return self.length


trainset = dataset(args.face_data, args.nonface_data)
# DataLoader
data_loader = DataLoader(trainset, batch_size=args.batch, shuffle=True)

epochs = args.epochs
running_loss = []
accuracy = []
print('starting training...')
print(device.type, 'detected')
for n in range(1, epochs + 1):
    # monitor training loss
    train_loss = 0.0
    hits = 0.0
    # Training
    for data, labels in data_loader:
        data, labels = data.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(data)
        if args.model == 'FCN':
            outputs = outputs.squeeze()
        _, preds = torch.max(outputs.data, 1)
        hits += torch.sum(preds == labels.data)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    running_loss.append(loss.item())
    accuracy.append(hits / len(data_loader.dataset))
    print('[Epochs: {} \tTraining Loss: {:.6f} \t Accuracy: {:.6f}]'.format(n, running_loss[-1], accuracy[-1]))
    if n % args.interval == 0 or n == epochs:
        torch.save({
            'device': device.type,
            'epoch': n,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss.item(),
        }, os.path.join(args.output_dir, f'12Net_{n}.pt'))

plt.plot(running_loss)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.savefig(os.path.join(args.output_dir, 'loss.png'))
