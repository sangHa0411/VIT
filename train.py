import os
import sys
import random
import argparse
import multiprocessing
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from model import VIT
from dataset import *
from cutmix import *
from loss import *
from scheduler import *

def progressLearning(value, endvalue, loss, acc, bar_length=50):
    percent = float(value + 1) / endvalue
    arrow = '-' * int(round(percent * bar_length)-1) + '>'
    spaces = ' ' * (bar_length - len(arrow))
    sys.stdout.write("\r[{0}] {1}/{2} \t Loss : {3:.3f} , Acc : {4:.3f}".format(arrow + spaces, value+1, endvalue, loss, acc))
    sys.stdout.flush()

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

def train(args) :
    # -- Seed
    seed_everything(args.seed)

    # -- Device
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    train_image = np.load(os.path.join(args.data_dir, 'train_image.npy'))
    train_label = np.load(os.path.join(args.data_dir, 'train_label.npy'))
    train_dset = ImageDataset(train_image, train_label, 1000)
    train_loader = DataLoader(
        train_dset, 
        batch_size=args.batch_size,
        shuffle = True,
        num_workers=multiprocessing.cpu_count()//2
    )

    val_image = np.load(os.path.join(args.data_dir, 'val_image.npy'))
    val_label = np.load(os.path.join(args.data_dir, 'val_label.npy'))
    val_dset = ImageDataset(val_image, val_label, 1000)
    val_loader = DataLoader(
        val_dset, 
        batch_size=args.batch_size,
        shuffle = True,
        num_workers=multiprocessing.cpu_count()//2
    )

    # -- Scalor & CutMix
    train_transform = TrainTransforms(args.org_size, args.input_size)
    val_transform = ValTransforms(args.input_size)

    img_cutmix = CutMix(args.input_size, args.input_size)

    # -- Model
    model = VIT(
        layer_size=args.layer_size,
        class_size=1000,
        channel_size=3,
        img_size=args.input_size,
        patch_size=args.patch_size,
        em_dim=args.embedding_size,
        num_heads=args.num_heads,
        hidden_size=args.hidden_size,
        drop_rate=args.drop_rate,
        norm_rate=args.norm_rate,
        cuda_flag=use_cuda).to(device)
    dumb_lr = 1e-4

    # -- Optimizer
    optimizer = optim.Adam(model.parameters(), lr=dumb_lr, betas=(0.9,0.999), weight_decay=1e-4)

    # -- Scheduler
    schedule_fn = Scheduler(args.embedding_size, args.warmup_steps, dumb_lr)
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch : schedule_fn(epoch))

    # -- Logging
    writer = SummaryWriter(args.log_dir)

    # -- Training
    min_loss = np.inf
    stop_count = 0
    log_count = 0
    # for each epoch
    for epoch in range(args.epochs) :
        idx = 0
        model.train()
        print('Epoch : %d/%d \t Learning Rate : %e' %(epoch, args.epochs, optimizer.param_groups[0]["lr"]))
        # training process
        for img_data, img_label in train_loader :
            img_data = img_data.float().to(device) / 255
            img_label = img_label.to(device)
            img_data, img_label = img_cutmix(img_data, img_label)

            optimizer.zero_grad()

            img_data = train_transform(img_data)
            img_out = model(img_data)
            
            loss = loss_fn(img_out, img_label)
            acc = acc_fn(img_out, img_label)

            loss.backward()
            optimizer.step()
            scheduler.step()
        
            progressLearning(idx, len(train_loader), loss.item(), acc.item())

            if (idx + 1) % 100 == 0 :
                writer.add_scalar('train/loss', loss.item(), log_count)
                writer.add_scalar('train/acc', acc.item(), log_count)
                log_count += 1
            idx += 1

        # validation process
        with torch.no_grad() :
            model.eval()
            loss_eval = 0.0
            acc_eval = 0.0
            for img_data, img_label in val_loader :
                img_data = img_data.float().to(device) / 255
                img_label = img_label.to(device)

                img_data = val_transform(img_data)
                img_out = model(img_data)
        
                loss_eval += loss_fn(img_out, img_label)
                acc_eval += acc_fn(img_out, img_label)

            loss_eval /= len(val_loader)
            acc_eval /= len(val_loader)

        writer.add_scalar('val/loss', loss_eval.item(), epoch)
        writer.add_scalar('val/acc', acc_eval.item(), epoch)
    
        if loss_eval < min_loss :
            min_loss = loss_eval
            torch.save({'epoch' : (epoch) ,  
                'model_state_dict' : model.state_dict() , 
                'loss' : loss_eval.item() , 
                'acc' : acc_eval.item()} , 
                os.path.join(args.model_dir, 'vit16_base_imagenet.pt'))        
            stop_count = 0 
        else :
            stop_count += 1
            if stop_count >= 5 :      
                print('\tTraining Early Stopped')
                break

        print('\nTest Loss : %.3f \t Test Accuracy : %.3f\n' %(loss_eval, acc_eval))

if __name__ == '__main__' :
    parser = argparse.ArgumentParser()
    # Seed argument
    parser.add_argument('--seed', type=int, default=777, help='random seed (default: 777)')
    # Image argument
    parser.add_argument('--org_size', type=int, default=256, help='orginial image size (default: 256)')
    parser.add_argument('--input_size', type=int, default=224, help='input image size (default: 224)')
    # Transformer argument
    parser.add_argument('--patch_size', type=int, default=16, help='patch size (default: 16)')
    parser.add_argument('--layer_size', type=int, default=12, help='layer size of transformer (default: 12)')
    parser.add_argument('--embedding_size', type=int, default=768, help='embedding size of transformer (default: 768)')
    parser.add_argument('--hidden_size', type=int, default=3072, help='hidden size of transformer (default: 3072)')
    parser.add_argument('--num_heads', type=int, default=12, help='num heads of transformer (default: 12)')
    parser.add_argument('--drop_rate', type=float, default=1e-1, help='dropout rate of transformer (default: 1e-1)')
    parser.add_argument('--norm_rate', type=float, default=1e-6, help='normalization rate of transformer (default: 1e-6)')
    # Training argumnet
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs to train (default: 100)')
    parser.add_argument('--warmup_steps', type=int, default=4000, help='warmup steps of training (default: 4000)')
    parser.add_argument('--init_lr', type=float, default=1e-2, help='initial learning rate')
    parser.add_argument('--batch_size', type=int, default=256, help='input batch size for training (default: 256)')
    # Container argument
    parser.add_argument('--data_dir', type=str, default='./Data')
    parser.add_argument('--model_dir', type=str, default='./Model')
    parser.add_argument('--log_dir' , type=str , default='./Log')

    args = parser.parse_args()
    train(args)
