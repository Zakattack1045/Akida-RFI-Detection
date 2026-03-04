"""
RFI (Radio Frequency Interference) Detection Training Script
Using Lava-DL Spiking Neural Network Framework

Expected Data Structure:
    data/spectrogram_dataset/
        train/
            spectrograms/  - Input spectrograms (interference + signal)
            masks/         - Binary ground truth masks (1=interference, 0=clean)
        test/
            spectrograms/
            masks/

Network Configuration (CPU Testing):
    - Input: 32x32 grayscale spectrograms
    - Output: 32x32 binary masks
    - Architecture: Convolutional encoder-decoder with SNN
    - Timesteps: 4 (reduced for CPU)
    - Channels: 4->8->32->8->4->1 (scaled down)
    - Epochs: 10 (for quick sanity check)
    - Batch size: 8

For full-scale training, increase:
    - HEIGHT/WIDTH (e.g., 128 or 256)
    - Network channels (e.g., 16->32->128->32->16->1)
    - Dense layer neurons (e.g., 256 or 512)
    - Timesteps (e.g., 16 or 32)
    - Epochs (e.g., 100)
    - Batch size (e.g., 32 or 64)
    - Set device to 'cuda'
"""

import os, sys
import h5py
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from PIL import Image
from utils import SpectrogramDataset, CombinedLoss, compute_segmentation_metrics
from network_test_2 import Network
import logging as log

# import slayer from lava-dl
import lava.lib.dl.slayer as slayer
import lava.lib.dl.bootstrap as bootstrap


from torchvision.io import decode_image

from params import *


def main():
    log.getLogger('PIL').setLevel(log.WARNING)

    ### Setup Training ###
    trained_folder = OUTPUT_PATH
    os.makedirs(trained_folder, exist_ok=True)
    if SAVE_SAMPLES:
        os.makedirs(os.path.join(trained_folder, 'samples'), exist_ok=True)

    device = torch.device('cpu')
    # device = torch.device('cuda')

    net = Network().to(device)

    # ## IEM WEIGHT AMPLIFICATION ##
    # def amplify_module_weights(module, factor=2.0, module_name=""):
    #     """Multiply all synapse weights in a module by factor."""
    #     count = 0
    #     with torch.no_grad():
    #         for name, param in module.named_parameters():
    #             if 'synapse.weight' in name and param.dim() >= 2:
    #                 param.data *= factor
    #                 count += 1
    #     if count > 0:
    #         print(f"✓ Amplified {count} weight tensors in {module_name} by {factor}x")

    # # Amplify IEM weights (signal dies in IEM modules)
    # print("Amplifying IEM weights for better signal propagation...")
    # for i, iem in enumerate(net.iems):
    #     amplify_module_weights(iem, factor=3, module_name=f"IEM{i+1}")

    # More conservative learning rate to avoid collapse
    optimizer = torch.optim.Adam(net.parameters(), lr=0.0003)  # Balanced: not too high, not too low
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    print(f"✓ Optimizer configured with LR={0.0003}")

    # CE Class weights
    class_weights = torch.tensor([1.0, 9.5], dtype=torch.float32).to(device)

    # Combined Dice + Weighted CE loss
    criterion = CombinedLoss(
        dice_weight=0.5,      # 40% weight to Dice loss (region-based)
        ce_weight=0.5,        # 60% weight to Weighted CE (pixel-wise)
        class_weights=class_weights
    )

    training_set = SpectrogramDataset(dir=TRAINING_PATH,height=HEIGHT,width=WIDTH)
    testing_set = SpectrogramDataset(dir=TESTING_PATH,height=HEIGHT,width=WIDTH)

    train_loader = DataLoader(dataset=training_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=8, persistent_workers=True)  # Reduced for CPU
    test_loader  = DataLoader(dataset=testing_set , batch_size=BATCH_SIZE, shuffle=True) 

    stats = slayer.utils.LearningStats()
    scheduler = bootstrap.routine.Scheduler(num_sample_iter=10, sample_period=5) # Higher sample rate for longer ANN
    print(f"✓ Scheduler configured: {scheduler.num_sample_iter} sample iterations, period {scheduler.sample_period}")

    # Tracking metrics across epochs
    train_metrics_history = {'accuracy': [],'f1': [], 'iou': [], 'precision': [], 'recall': [], 'tp': [], 'fp': []}
    test_metrics_history = {'accuracy': [],'f1': [], 'iou': [], 'precision': [], 'recall': [], 'tp': [], 'fp': []}


    ### Training Loop ###
    epochs = EPOCHS  
    for epoch in range(epochs):
        print(f'\n\nStarting Epoch {epoch+1}/{epochs}:')

        # Accumulators for epoch-level metrics
        train_all_preds = []
        train_all_labels = []
        samples_saved = 0

        for i, (input, label) in enumerate(train_loader, 0):
            net.train()
            mode = scheduler.mode(epoch, i, net.training)

            input = input.to(device)
            label = label.to(device)

            # Debug: Check input before amplification
            if i == 0 and epoch < 3:
                print(f"  Input - min: {input.min():.4f}, max: {input.max():.4f}, mean: {input.mean():.4f}")
                print(f"  Mode: {mode.base_mode}")

            output = net.forward(input, mode)

            # Debug: Check raw output before averaging
            if i == 0 and epoch < 3:
                print(f"  Output shape: {output.shape}")
                print(f"  Output BEFORE averaging - min: {output.min():.4f}, max: {output.max():.4f}, mean: {output.mean():.4f}")
                print(f"  Output spikes - Ch0: {output[:, 0].mean():.4f}, Ch1: {output[:, 1].mean():.4f}")
                num_nonzero = (output > 0).sum().item()
                total_elements = output.numel()
                print(f"  Non-zero elements: {num_nonzero} / {total_elements} ({100*num_nonzero/total_elements:.2f}%)")

            # Average over time dimension to get spatial output

            rate = torch.mean(output, dim=-1)

            # Prepare label for CrossEntropyLoss
            # Remove channel dimension: (N, C, H, W) -> (N, H, W)
            label_squeezed = label.squeeze(1).long()

            # Resize label to match output dimensions if needed
            if rate.shape[2:] != label_squeezed.shape[1:]:
                label_squeezed = F.interpolate(label.float(), size=rate.shape[2:], mode='nearest').squeeze(1).long()

            # Combined Dice + Weighted Cross-Entropy Loss
            loss = criterion(rate, label_squeezed)

            # Debug loss components on first batch
            if i == 0 and epoch < 3:
                with torch.no_grad():
                    dice_component = criterion.dice_loss(rate, label_squeezed).item()
                    ce_component = torch.nn.functional.cross_entropy(rate, label_squeezed, weight=criterion.class_weights).item()
                    print(f"  Loss breakdown - Total: {loss.item():.6f}, Dice: {dice_component:.6f}, CE: {ce_component:.6f}")
                    print(f"  Rate stats - Ch0 mean: {rate[:, 0].mean():.6f}, Ch1 mean: {rate[:, 1].mean():.6f}")
                    print(f"  Rate stats - Ch0 std: {rate[:, 0].std():.6f}, Ch1 std: {rate[:, 1].std():.6f}")
                    after_softmax = torch.nn.functional.softmax(rate, dim=1)
                    print(f"  After softmax - Ch0: {after_softmax[:, 0].mean():.6f}, Ch1: {after_softmax[:, 1].mean():.6f}")

            # Prediction: argmax across channel dimension
            prediction = torch.argmax(rate, dim=1)  # (N, H, W) with values 0 or 1

            # Debug: Monitor predictions across all batches in epoch 1
            if epoch < 2:
                num_rfi_preds = (prediction == 1).sum().item()
                if num_rfi_preds > 0:
                    print(f"    Batch {i}: Predicted {num_rfi_preds} RFI pixels! (Ch0: {rate[:, 0].mean():.6f}, Ch1: {rate[:, 1].mean():.6f})")

            # Accumulate for epoch metrics
            train_all_preds.append(prediction.cpu())
            train_all_labels.append(label_squeezed.cpu())

            # Track loss (average per image, not per pixel)
            stats.training.num_samples += input.shape[0]  # Number of images in batch
            stats.training.loss_sum += loss.cpu().data.item() * input.shape[0]
            # Note: Don't use stats.training.correct_samples - accuracy computed from accumulated preds

            optimizer.zero_grad()
            loss.backward()

            # Monitor gradients on first batch
            if i == 0 and epoch < 3:
                total_grad_norm = 0.0
                head_grad_norm = 0.0
                for name, param in net.named_parameters():
                    if param.grad is not None:
                        grad_norm = param.grad.norm().item()
                        total_grad_norm += grad_norm ** 2
                        if 'head' in name:
                            head_grad_norm += grad_norm ** 2
                total_grad_norm = total_grad_norm ** 0.5
                head_grad_norm = head_grad_norm ** 0.5
                print(f"  Gradient norms - Total: {total_grad_norm:.6f}, Head layer: {head_grad_norm:.6f}")

            optimizer.step()

            # Monitor channel differentiation on first batch
            if i == 0:
                print(f"  Train Batch 0 - Ch0: {rate[:, 0].mean():.4f}, Ch1: {rate[:, 1].mean():.4f}, "
                      f"Diff: {(rate[:, 1] - rate[:, 0]).mean():.4f}")

            # Save sample visualizations
            if SAVE_SAMPLES and samples_saved < SAMPLES_PER_EPOCH:
                # Save first sample in batch
                fig, axes = plt.subplots(1, 4, figsize=(16, 4))

                # Input spectrogram
                axes[0].imshow(input[0, 0].cpu().numpy(), cmap='viridis')
                axes[0].set_title('Input Spectrogram')
                axes[0].axis('off')

                # Ground truth mask
                axes[1].imshow(label_squeezed[0].cpu().numpy(), cmap='gray', vmin=0, vmax=1)
                axes[1].set_title('Ground Truth (1=RFI)')
                axes[1].axis('off')

                # Prediction
                axes[2].imshow(prediction[0].cpu().numpy(), cmap='gray', vmin=0, vmax=1)
                pred_rfi_pixels = (prediction[0] == 1).sum().item()
                axes[2].set_title(f'Prediction (RFI pixels: {pred_rfi_pixels})')
                axes[2].axis('off')

                # Overlay: Green=TP, Red=FP, Blue=FN, Black=TN
                overlay = np.zeros((*prediction[0].shape, 3))
                pred_np = prediction[0].cpu().numpy()
                gt_np = label_squeezed[0].cpu().numpy()

                # True Positives: Green
                overlay[(pred_np == 1) & (gt_np == 1)] = [0, 1, 0]
                # False Positives: Red
                overlay[(pred_np == 1) & (gt_np == 0)] = [1, 0, 0]
                # False Negatives: Blue
                overlay[(pred_np == 0) & (gt_np == 1)] = [0, 0, 1]
                # True Negatives: Black (stays 0)

                axes[3].imshow(overlay)
                axes[3].set_title('Overlay: Green=TP, Red=FP, Blue=FN')
                axes[3].axis('off')

                plt.tight_layout()
                plt.savefig(os.path.join(trained_folder, 'samples', f'epoch{epoch+1}_sample{samples_saved+1}.png'))
                plt.close()
                samples_saved += 1

        # Compute training metrics for this epoch
        train_preds_epoch = torch.cat(train_all_preds, dim=0)
        train_labels_epoch = torch.cat(train_all_labels, dim=0)
        train_metrics = compute_segmentation_metrics(train_preds_epoch, train_labels_epoch)

        # Store metrics
        train_metrics_history['accuracy'].append(train_metrics['accuracy'])
        train_metrics_history['f1'].append(train_metrics['f1'])
        train_metrics_history['iou'].append(train_metrics['iou'])
        train_metrics_history['precision'].append(train_metrics['precision'])
        train_metrics_history['recall'].append(train_metrics['recall'])
        train_metrics_history['tp'].append(train_metrics['tp'])
        train_metrics_history['fp'].append(train_metrics['fp'])

        print(f'[Epoch {epoch+1:2d}/{epochs}] Train - Loss: {stats.training.loss:.6f}, Acc: {train_metrics["accuracy"]:.4f}, '
              f'F1: {train_metrics["f1"]:.4f}, IoU: {train_metrics["iou"]:.4f}, '
              f'Recall: {train_metrics["recall"]:.4f}, TP: {int(train_metrics["tp"])}, FP: {int(train_metrics["fp"])}')

        # Accumulators for test metrics
        test_all_preds = []
        test_all_labels = []
        

        for i, (input, label) in enumerate(test_loader, 0):
            net.eval()
            mode = scheduler.mode(epoch, i, net.training)

            with torch.no_grad():
                input = input.to(device)
                label = label.to(device)

                output = net.forward(input, mode=mode)
                # Average over time dimension to get spatial output
                # output shape: (N, C=2, H, W, T) -> after mean: (N, C=2, H, W)
                rate = torch.mean(output, dim=-1)

                # Prepare label for CrossEntropyLoss
                # Remove channel dimension: (N, C, H, W) -> (N, H, W)
                label_squeezed = label.squeeze(1).long()

                # Resize label to match output dimensions if needed
                if rate.shape[2:] != label_squeezed.shape[1:]:
                    label_squeezed = F.interpolate(label.float(), size=rate.shape[2:], mode='nearest').squeeze(1).long()

                # Combined Dice + Weighted Cross-Entropy Loss
                loss = criterion(rate, label_squeezed)

                # Prediction: argmax across channel dimension
                prediction = torch.argmax(rate, dim=1)  # (N, H, W) with values 0 or 1

                # Debug: Print logit statistics (AFTER prediction is computed)
                if i == 0:  # First batch only
                    print(f"\n  Test Batch 0 - Ch0: {rate[:, 0].mean():.3f}, Ch1: {rate[:, 1].mean():.3f}")
                    print(f"  Diff (Ch1-Ch0): {(rate[:, 1] - rate[:, 0]).mean():.3f}")
                    print(f"  Predicted RFI pixels: {(prediction == 1).sum().item()}")

                # Accumulate for epoch metrics
                test_all_preds.append(prediction.cpu())
                test_all_labels.append(label_squeezed.cpu())

                

            # Track loss (average per image, not per pixel)
            stats.testing.num_samples += input.shape[0]  # Number of images in batch
            stats.testing.loss_sum += loss.cpu().data.item() * input.shape[0]
            # Note: Don't use stats.testing.correct_samples - accuracy computed from accumulated preds

        # Compute test metrics for this epoch
        test_preds_epoch = torch.cat(test_all_preds, dim=0)
        test_labels_epoch = torch.cat(test_all_labels, dim=0)
        test_metrics = compute_segmentation_metrics(test_preds_epoch, test_labels_epoch)

        # Store metrics
        test_metrics_history['accuracy'].append(test_metrics['accuracy'])
        test_metrics_history['f1'].append(test_metrics['f1'])
        test_metrics_history['iou'].append(test_metrics['iou'])
        test_metrics_history['precision'].append(test_metrics['precision'])
        test_metrics_history['recall'].append(test_metrics['recall'])
        test_metrics_history['tp'].append(test_metrics['tp'])
        test_metrics_history['fp'].append(test_metrics['fp'])

        print(f'[Epoch {epoch+1:2d}/{epochs}] Test  - Loss: {stats.testing.loss:.6f}, Acc: {test_metrics["accuracy"]:.4f}, '
              f'F1: {test_metrics["f1"]:.4f}, IoU: {test_metrics["iou"]:.4f}, '
              f'Recall: {test_metrics["recall"]:.4f}, TP: {int(test_metrics["tp"])}, FP: {int(test_metrics["fp"])}')

        if mode.base_mode == bootstrap.routine.Mode.SNN:
            scheduler.sync_snn_stat(stats.testing)
            print('\r', ' '*len(f'\r[Epoch {epoch:2d}/{epochs}] {stats}'))
            print(f'SNN Testing: {scheduler.snn_stat}')
            scheduler.update_snn_stat()

        if stats.testing.best_accuracy:
                torch.save(net.state_dict(), trained_folder + '/network.pt')

        # Step learning rate scheduler
        lr_scheduler.step()
        if (epoch + 1) % 5 == 0:
            print(f"  Learning rate adjusted to: {optimizer.param_groups[0]['lr']:.6f}")

        stats.update()
        stats.save(trained_folder + '/')

        # Save additional metrics mid-training (after each epoch)
        np.savetxt(os.path.join(trained_folder, 'train_accuracy.txt'),
                   train_metrics_history['accuracy'], fmt='%.6f')
        np.savetxt(os.path.join(trained_folder, 'train_f1.txt'),
                   train_metrics_history['f1'], fmt='%.6f')
        np.savetxt(os.path.join(trained_folder, 'train_iou.txt'),
                   train_metrics_history['iou'], fmt='%.6f')
        np.savetxt(os.path.join(trained_folder, 'train_recall.txt'),
                   train_metrics_history['recall'], fmt='%.6f')
        np.savetxt(os.path.join(trained_folder, 'train_tp.txt'),
                   train_metrics_history['tp'], fmt='%.0f')
        np.savetxt(os.path.join(trained_folder, 'train_fp.txt'),
                   train_metrics_history['fp'], fmt='%.0f')

        np.savetxt(os.path.join(trained_folder, 'test_accuracy.txt'),
                   test_metrics_history['accuracy'], fmt='%.6f')
        np.savetxt(os.path.join(trained_folder, 'test_f1.txt'),
                   test_metrics_history['f1'], fmt='%.6f')
        np.savetxt(os.path.join(trained_folder, 'test_iou.txt'),
                   test_metrics_history['iou'], fmt='%.6f')
        np.savetxt(os.path.join(trained_folder, 'test_recall.txt'),
                   test_metrics_history['recall'], fmt='%.6f')
        np.savetxt(os.path.join(trained_folder, 'test_tp.txt'),
                   test_metrics_history['tp'], fmt='%.0f')
        np.savetxt(os.path.join(trained_folder, 'test_fp.txt'),
                   test_metrics_history['fp'], fmt='%.0f')

    # Note: Metrics are already saved after each epoch (lines 390-414)

    ### Plot training statistics ###
    stats.plot(figsize=(15, 5))

    # Plot additional metrics
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    axes[0, 0].plot(train_metrics_history['f1'], label='Train')
    axes[0, 0].plot(test_metrics_history['f1'], label='Test')
    axes[0, 0].set_title('F1 Score')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    axes[0, 1].plot(train_metrics_history['iou'], label='Train')
    axes[0, 1].plot(test_metrics_history['iou'], label='Test')
    axes[0, 1].set_title('IoU (mIoU for binary)')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].legend()
    axes[0, 1].grid(True)

    axes[1, 0].plot(train_metrics_history['recall'], label='Train')
    axes[1, 0].plot(test_metrics_history['recall'], label='Test')
    axes[1, 0].set_title('Recall (RFI Detection Rate)')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].legend()
    axes[1, 0].grid(True)

    axes[1, 1].plot(train_metrics_history['precision'], label='Train')
    axes[1, 1].plot(test_metrics_history['precision'], label='Test')
    axes[1, 1].set_title('Precision (RFI Purity)')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].legend()
    axes[1, 1].grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(trained_folder, 'metrics.png'))
    plt.close()

    ### Export trained network to hdf5 ###
    net.load_state_dict(torch.load(trained_folder + '/network.pt'))
    net.export_hdf5(trained_folder + '/network.net')

if __name__ == '__main__':
    main()