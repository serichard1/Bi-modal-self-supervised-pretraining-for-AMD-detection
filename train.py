from datetime import datetime
from statistics import mean
from torch.utils.data import DataLoader
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from utils import BimodalDataset, DataAugmentation, BatchSampleWrapper, ProjectorHead, BarlowSampleLoss
from vision_transformer import ViT
import argparse
import pathlib
import torch
from tqdm import tqdm
import numpy as np

def train(model, train_loader, optimizer, criterion, scaler, device):
    m = []
    model.train()
    for _, training_data in enumerate(tqdm(train_loader, desc="training batches: ")):
        inputs, labels = training_data

        labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)

        loss = criterion(outputs)

        scaler.scale(loss).backward()

        m.append(loss.item())

        scaler.step(optimizer)
        scaler.update()

    m = mean(m)
    return m

def validation(model,valid_loader,criterion, device):
    m = []
    model.eval()
    with torch.no_grad():
        for _, valid_data in enumerate(tqdm(valid_loader, desc="validation batches")):
            inputs, labels = valid_data

            labels.to(device)
            outputs = model(inputs)

            loss = criterion(outputs)
            m.append(loss.item())
        
    m = mean(m)

    return m

def main(args):

    device = torch.device(args.device)
    transform_aug = DataAugmentation(args.image_size, args.n_augmentations)
    train_set = BimodalDataset(pathlib.Path(args.path_data), transform=transform_aug)

    train_size = int(0.90* len(train_set))
    valid_size = len(train_set) - train_size
    train_set, valid_set = random_split(train_set, [train_size, valid_size])

    train_loader = DataLoader(
        train_set,
        batch_size = args.batch_size,
        shuffle = True,
        num_workers = args.n_workers,
        pin_memory = True,
    )

    valid_loader = DataLoader(
        valid_set,
        batch_size = 16,
        shuffle = False,
        drop_last = False,
    )

    print("number of samples: ", train_size + valid_size)
    print("train: ", train_size)
    print("valid: ", valid_size)

    logger = SummaryWriter(args.tensorboard_dir)
    dataiter = iter(train_loader)
    images, labels = next(dataiter)

    cfps = torch.stack(images[:args.n_augmentations])
    cfps = torch.permute(cfps, (1, 0, 2, 3, 4))
    octs = torch.stack(images[args.n_augmentations:])
    octs = torch.permute(octs, (1, 0, 2, 3, 4))

    print("color fundus images shape (batch, n_augm, channels, height, width): ", cfps.shape)
    print("OCTs shape (batch, n_augm, channels, height, width): ", octs.shape)
    print("labels first batch: ", labels, 'shape: ', labels.shape)

    vit_encoder = ViT(
                size=args.image_size,
                patch_size=args.patch_size,
                embed_dim=args.embed_dim,
                qvk_p=args.dropout_p
                )
    
    model = BatchSampleWrapper(
                backbone=vit_encoder,
                projector=ProjectorHead(
                    in_dim=args.embed_dim, 
                    hidden_dim=int(args.embed_dim/4), 
                    out_dim=args.feature_space
                    ),
                n_aug=args.n_augmentations,
                device=device
                )

    model.to(device)
    torch.autograd.set_detect_anomaly(True)
    criterion = BarlowSampleLoss(
                args.embed_dim,
                temp=args.loss_temp,
                batch=args.batch_size
            )
    
    lr = 0.0005 * args.batch_size / 256
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=args.weight_decay,
    )

    min_loss = 100
    patience_trigger = 0
    scaler = torch.cuda.amp.GradScaler()

    date = datetime.now()

    for epoch in range(args.n_epochs):
        train_bloss = train(model,train_loader,optimizer,criterion, scaler, device)
        valid_bloss = validation(model,valid_loader,criterion, device)
        if train_bloss < min_loss:
            min_loss = train_bloss
            patience_trigger = 0
            torch.save(model.state_dict(), f"results/weights_{date}.pt") 
        else:
            patience_trigger += 1

        logger.add_scalars(f"Vit_{date}", {'Loss training': train_bloss, 'Loss valid': valid_bloss}, epoch)

        print(f'\nEpoch {epoch}')
        print(f'Train Loss => {round(train_bloss,5)}', end=' | ')
        print(f'valid Loss  => {round(valid_bloss,5)} (earlystop => {patience_trigger}/{args.patience})')
        if patience_trigger >= args.patience:
            print('Early stop !')
            break

    print(f'end of training')
    torch.save(model.state_dict(), f"results/weights_{date}.pt") 

    print(f'best weights have been saved in results/weights_{date}')
    print(f'loss and accuracy score curves have been saved in Tensorboard logs')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        "SSL pretraining these LaBRI/CHU",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("-v", "--version-model", 
                        type=str, choices=("ViT8", "ViT16"), default='ViT8')
    parser.add_argument("-m", "--mode", 
                        type=str, choices=("training", "training_evaluation"), default="training_evaluation")
    parser.add_argument("-d", "--device", 
                        type=str, choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("-t", "--tensorboard-dir", 
                        type=str, default="logs")
    parser.add_argument("--n-workers", 
                        type=int, default=1)

    parser.add_argument("-p", "--path-data", 
                        type=str, default="data")
    parser.add_argument("-b", "--batch-size", 
                        type=int, default=2)
    parser.add_argument("-s", "--image-size", 
                        type=int, default=250)
    parser.add_argument("--patch-size", 
                        type=int, default=50)
    parser.add_argument("--embed-dim", 
                        type=int, default=768)
    parser.add_argument("--feature-space", 
                        type=int, default=10)
    parser.add_argument("--dropout-p", 
                        type=float, default=0.2)
    parser.add_argument("--n-augmentations", 
                        type=float, default=3)
    parser.add_argument("--loss-temp", 
                        type=float, default=0.01)
    
    parser.add_argument("-e", "--n-epochs", 
                        type=int, default=100)
    parser.add_argument("--patience", 
                        type=int, default=2)
    parser.add_argument("-w", "--weight-decay",
                        type=float, default=0.4)
    
    args = parser.parse_args()
    print(vars(args))
    print("########################")
    main(args)