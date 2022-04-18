import argparse
from model import CCT, OCT
from eval_metrics import compute_eer
from loss import FocalLoss
from utils import load_data, adjust_learning_rate, get_run_logdir

import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, dataloader
import torchaudio.transforms as T

#the following code is only used to solve the limited memory issues in my Docker, may be it is not necessary for your computer
import os
import sys 
from torch.multiprocessing import reductions 
from multiprocessing.reduction import ForkingPickler

default_collate_func=dataloader.default_collate

def default_collate_override(batch):
    dataloader._use_shared_memory=False 
    return default_collate_func(batch)

setattr(dataloader, 'default_collate', default_collate_override)

for t in torch._storage_classes:
    if sys.version_info[0]==2:
        if t in ForkingPickler.dispatch:
            del ForkingPickler.dispatch[t]
    else:
        if t in ForkingPickler._extra_reducers:
            del ForkingPickler._extra_reducers[t]

parser = argparse.ArgumentParser(description='Train the OCT model')
parser.add_argument('--lr', default=4e-4, type=float, help='learning rate')
parser.add_argument('--epochs', default=300, type=int, help='training epochs')
parser.add_argument('--embedding', default=128, type=int,
                    help='embedding dim of transformer encoder')
parser.add_argument('--conv', default=3, type=int,
                    help='number of conv layers in tokenizer')
parser.add_argument('--trans', default=2, type=int,
                    help='number of transformer encoder layer')
parser.add_argument('--heads', default=1, type=int,
                    help='number of heads of each transformer encoder layer')
parser.add_argument('--posit', default='sine', type=str,
                    help='type of positional embedding')
parser.add_argument('-b', default=64, type=int, help='batch_size')
parser.add_argument('-m', action='store_true',
                    help='true for training OCT instead of CCT')
parser.add_argument('--tmasking', default=80, type=int,
                    help='parameter for time masking')
parser.add_argument('--fmasking', default=20, type=int,
                    help='parameter for frequency masking')
parser.add_argument('--resume', action='store_true',
                    help='if resume from former checkpoint, if true, more needed')
parser.add_argument('--checkpoint', default=None,
                    type=str, help='the path of the checkpoint')
parser.add_argument('--base', default='~/logs',
                    type=str, help='path for saving logs')
parser.add_argument('--finetune', action='store_true', help='whether to train on the merged dataset')
args = parser.parse_args()

log_dir = os.path.expanduser(get_run_logdir(args.base))
checkpoint_dir = os.path.join(log_dir, 'checkpoints')
os.makedirs(checkpoint_dir, exist_ok=True)
writer = SummaryWriter(log_dir=log_dir)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

if args.m:
    model = OCT(frame=512, embedding_dim=args.embedding, n_input_channels=60, n_conv_layers=args.conv, kernel_size=3, stride=1, padding=1, pooling_kernel_size=3, pooling_stride=2, pooling_padding=1,
                dropout_rate=0.1, attention_dropout=0.1, stochastic_depth=0.1, num_layers=args.trans, num_heads=args.heads, num_classes=2, positional_embedding=args.posit, mlp_ratio=1.0)
else:
    model = CCT(frame=512, feature=60, embedding_dim=args.embedding, n_input_channels=1, n_conv_layers=args.conv, kernel_size=3, stride=1, pooling_kernel_size=3, pooling_stride=2,
                dropout_rate=0.1, attention_dropout=0.1, stochastic_depth=0.1, num_layers=args.trans, num_heads=args.heads, num_classes=2, positional_embedding=args.posit, mlp_ratio=1.0)
model.to(device)

time_masking = T.TimeMasking(time_mask_param=args.tmasking)
freq_masking = T.FrequencyMasking(freq_mask_param=args.fmasking)
criterion = FocalLoss(gamma=2.0, alpha=0.80).to(device)
# criterion=torch.nn.CrossEntropyLoss(weight=torch.tensor([5.0, 1.0], dtype=torch.float32)).to(device)
optimizer = torch.optim.AdamW(
    model.parameters(), lr=args.lr, weight_decay=1e-4)
start_epoch=0 
if args.resume:
    model.load_state_dict(torch.load(args.checkpoint)[0])
    optimizer.load_state_dict(torch.load(args.checkpoint)[1])
    start_epoch=torch.load(args.checkpoint)[-1]+1


train_ds, test_ds = load_data(args.finetune)
trld = DataLoader(train_ds, batch_size=args.b, shuffle=True, num_workers=4)
teld = DataLoader(test_ds, batch_size=args.b, shuffle=False, num_workers=4)


for epoch in range(start_epoch, args.epochs):
    
    adjust_learning_rate(optimizer, epoch, lr=args.lr,
                        warmup=30, epochs=args.epochs)
    model.train()
    running_loss = 0.
    epoch_steps = 0
    for i, (lfcc, target, name) in enumerate(trld):
        lfcc, target=lfcc.to(torch.float32).to(device).transpose(-1, -2), target.to(device)
        with torch.no_grad():

            lfcc = time_masking(freq_masking(lfcc))
            if not args.m:
            # not necessary, however, i still do it for convention
                lfcc = lfcc.transpose(-1, -2).unsqueeze(dim=1)
        logits, feature = model(lfcc)
        loss=criterion(logits, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        epoch_steps += 1
        if i % 50 == 49:
            print(
                f'Training Epoch: [{epoch+1}] Step: [{i+1}] Loss: {running_loss/epoch_steps}')

    if args.finetune:
        #the following code aloows you to see the model's performance on the validation set
        #if you turn the finetune flag to false, then you can save the model and do inference on the test dataset
        model.eval()
        index_loader, score_loader = [], []
        val_loss = 0.
        val_steps = 0
        for i, (lfcc, target, ways) in enumerate(teld):
            index_loader.append(target)
            lfcc, target = lfcc.to(torch.float32).to(
                device).transpose(-1, -2), target.to(device)
            if not args.m:
                lfcc = lfcc.transpose(-1, -2).unsqueeze(dim=1)
            with torch.no_grad():
                logits, feature = model(lfcc)
                batch_score = torch.softmax(logits, dim=-1)[:, 0]
                loss = criterion(logits, target)
            score_loader.append(batch_score)
            val_loss += loss.item()
            val_steps += 1
        score = torch.cat(score_loader, dim=0).cpu().numpy()
        targets = torch.cat(index_loader, dim=0).cpu().numpy()
        val_eer = compute_eer(score[targets == 0], score[targets == 1])[0]
        other_val_eer = compute_eer(-score[targets == 0], -score[targets == 1])[0]
        val_eer = min(val_eer, other_val_eer)

        
        writer.add_scalar('Val Loss', val_loss/val_steps, epoch+1)
        writer.add_scalar('Val EER', val_eer, epoch+1)
    writer.add_scalar('Training Loss', running_loss / epoch_steps, epoch + 1)
    torch.save((model.state_dict(), optimizer.state_dict(), epoch), os.path.join(checkpoint_dir, f'checkpoint_{epoch+1}'))


print('finish')
