import os
import math
import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import scan_all_data, MIMIIMultiTaskDataset
from models import SincNetTransformer, FocalLoss
from utils import run_epoch

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_root = "./data/MIMII" 
    class_names, all_files = scan_all_data(data_root)
    print(f"Machines: {class_names} | Files: {len(all_files)}")
    if len(class_names) == 0 or len(all_files) == 0: return

    random.seed(42)
    random.shuffle(all_files)
    split_idx = int(len(all_files) * 0.8)
    train_files, val_files = all_files[:split_idx], all_files[split_idx:]

    sample_rate, duration, epochs, batch_size = 16000, 4.0, 20, 16
    num_machines = len(class_names)

    train_dataset = MIMIIMultiTaskDataset(train_files, class_names, sample_rate, duration, is_train=True)
    val_dataset = MIMIIMultiTaskDataset(val_files, class_names, sample_rate, duration, is_train=False)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True)

    model = SincNetTransformer(sample_rate=sample_rate, num_machines=num_machines).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    
    def warmup_cosine(epoch):
        if epoch < 3: return (epoch + 1) / 3.0
        return 0.5 * (1.0 + math.cos(math.pi * (epoch - 3) / (epochs - 3)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warmup_cosine)
    
    criterion_m = nn.CrossEntropyLoss() 
    criterion_a = FocalLoss(alpha=0.75, gamma=2.0).to(device)
    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == 'cuda'))

    best_score = 0.0
    os.makedirs("weights", exist_ok=True)
    print("Start Training...")

    for epoch in range(1, epochs + 1):
        tr = run_epoch(model, train_loader, criterion_m, criterion_a, device, optimizer, scaler, is_train=True)
        va = run_epoch(model, val_loader, criterion_m, criterion_a, device, None, None, is_train=False)
        scheduler.step()
        score = (va[1] + va[3]) / 2.0
        print(f"Epoch {epoch:02d} | Val M-Acc: {va[1]:.4f} M-F1: {va[2]:.4f} A-F1: {va[3]:.4f} A-AUC: {va[4]:.4f}")
        if score > best_score:
            best_score = score
            torch.save({'model_state_dict': model.state_dict(), 'class_names': class_names, 'num_machines': num_machines, 'sample_rate': sample_rate, 'duration': duration}, "weights/best_sincnet_transformer.pkl")
            print(f"Saved Best Model! (Score: {best_score:.4f})")

if __name__ == "__main__":
    main()