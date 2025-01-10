
import timm
import torch
import torch.nn as nn
import cv2
import numpy as np
import os
import time

from scipy.stats import norm
import matplotlib.pyplot as plt

from tqdm import tqdm

from dataloader import get_imnet1k_dataloader

BATCH_SIZE = 128
LOG_DIR = './logs'


model_list = [
    # 'tiny_vit_5m_224.dist_in22k_ft_in1k',
    'caformer_b36.sail_in22k_ft_in1k',
    # 'vit_large_patch14_clip_224.openai_ft_in12k_in1k',
    # 'vgg11_bn.tv_in1k',
]

dataset_list = [
    '/data/ImageNet-1k/val',
    '/data/ImageNet-1k/val_sampled/gaussian_2',
    '/data/ImageNet-1k/val_sampled/gaussian_4',
    '/data/ImageNet-1k/val_sampled/gaussian_8',
    '/data/ImageNet-1k/val_sampled/gaussian_16',
    '/data/ImageNet-1k/val_sampled/subsample_2',
    '/data/ImageNet-1k/val_sampled/subsample_4',
    '/data/ImageNet-1k/val_sampled/subsample_8',
    '/data/ImageNet-1k/val_sampled/subsample_16',
    '/data/ImageNet-1k/val_sampled/avgsample_2',
    '/data/ImageNet-1k/val_sampled/avgsample_4',
    '/data/ImageNet-1k/val_sampled/avgsample_8',
    '/data/ImageNet-1k/val_sampled/avgsample_16',
]

criterion = nn.CrossEntropyLoss().cuda()

def gaussian_subsample(img, level=1):
    # img: Tensor, shape (B, C, H, W)
    # subsample image using cv2.pyrDown
    img = img.cpu().numpy().transpose(0, 2, 3, 1)
    for _ in range(level):
        newimage = np.zeros((img.shape[0], img.shape[1] // 2, img.shape[2] // 2, img.shape[3]))
        for i in range(img.shape[0]):
            newimage[i] = cv2.pyrDown(img[i])
        img = newimage

    for i in range(img.shape[0]):
        newimage = np.zeros((img.shape[0], img.shape[1] * 2, img.shape[2] * 2, img.shape[3]))
        for _ in range(level):
            newimage[i] = cv2.pyrUp(img[i])

    img = torch.tensor(img).cuda().transpose(1, 3).transpose(2, 3).float()

    return img


# validation
@torch.no_grad()
def validate(test_loader, model, criterion):
    model.eval()
    
    num_data = 0
    num_correct = 0
    sum_loss = 0
    correct_probs = []
    incorrect_probs = []
    
    pbar = tqdm(test_loader, leave=False, total=len(test_loader))
    for data, target in pbar:
        data, target = data.cuda(), target.cuda()
        output = model(data)
        loss = criterion(output, target)
        
        _, predicted = torch.max(output.data, 1)
        max_probs = torch.max(torch.softmax(output, dim=1), dim=1)[0]
        
        num_data += target.size(0)
        num_correct += (predicted == target).sum().item()
        sum_loss += loss.item() * target.size(0)
        
        correct_probs.extend(max_probs[predicted == target].cpu().numpy())
        incorrect_probs.extend(max_probs[predicted != target].cpu().numpy())

        avg_loss = sum_loss / num_data
        accuracy = num_correct / num_data

        pbar.set_description(f'Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}, O Conf: {np.mean(correct_probs):.4f}, X Conf: {np.mean(incorrect_probs):.4f}')
    
    avg_loss = sum_loss / num_data
    accuracy = num_correct / num_data
    
    additional_info = {
        'correct_probs': correct_probs,
        'incorrect_probs': incorrect_probs,
    }
    
    return avg_loss, accuracy, additional_info



for mname in model_list:
    # check if log file exists
    log_dir_name = f'{LOG_DIR}/{mname}'

    if not os.path.exists(log_dir_name):
        os.makedirs(log_dir_name)

    log_file_name = f'{log_dir_name}/log.txt'
    log_file = open(log_file_name, 'w')

    model = timm.create_model(mname, pretrained=True, num_classes=1000).cuda()
    nparams = sum(p.numel() for p in model.parameters())
    
    # measure latency
    num_repeat = 100
    start_time = time.time()
    for _ in range(num_repeat):
        model(torch.randn(1, 3, 224, 224).cuda())
    latency = (time.time() - start_time) / num_repeat

    log_file.write(
        f"[Model Info]\n"
        f" - MODEL    : {mname}\n"
        f" - PARAMS   : {nparams:,d}\n"
        f" - LATENCY  : {latency:.4f}\n"
        f"\n"
    )
    log_file.flush()

    
    for dataset in dataset_list:

        _, val_loader = get_imnet1k_dataloader(root=dataset, batch_size=BATCH_SIZE, augmentation=False, val_only=True)


        val_loss, val_acc, additional_info = validate(val_loader, model, criterion)

        # draw histogram and save
        # find the threshold of O and X confidence by Bayesian method
        correct_probs = np.array(additional_info['correct_probs'])
        incorrect_probs = np.array(additional_info['incorrect_probs'])

        # Fit distributions to correct and incorrect probabilities
        correct_mean, correct_std = np.mean(correct_probs), np.std(correct_probs)
        incorrect_mean, incorrect_std = np.mean(incorrect_probs), np.std(incorrect_probs)

        # Define CDFs for correct and incorrect
        correct_cdf = lambda x: norm.cdf(x, loc=correct_mean, scale=correct_std)
        incorrect_cdf = lambda x: norm.cdf(x, loc=incorrect_mean, scale=incorrect_std)

        # Find priors based on data distribution
        P_A = len(correct_probs) / (len(correct_probs) + len(incorrect_probs))
        P_B = len(incorrect_probs) / (len(correct_probs) + len(incorrect_probs))

        # Fine the threshold that minimizes the failure probability
        def failure_probability(x):
            return correct_cdf(x) * P_A + (1 - incorrect_cdf(x)) * P_B
        
        thresholds = np.linspace(0, 1, 1000)
        failure_probs = [failure_probability(t) for t in thresholds]
        threshold = thresholds[np.argmin(failure_probs)]

        # Create histogram data
        correct_hist, correct_bins = np.histogram(correct_probs, bins=50, density=True)
        incorrect_hist, incorrect_bins = np.histogram(incorrect_probs, bins=50, density=True)

        # Scale the histogram to match the density
        correct_hist = correct_hist * P_A
        incorrect_hist = incorrect_hist * P_B

        # Plot the histogram
        plt.bar(correct_bins[:-1], correct_hist, width=correct_bins[1] - correct_bins[0], alpha=0.5, label=f'Correct: {val_acc:.4f}')
        plt.bar(incorrect_bins[:-1], incorrect_hist, width=incorrect_bins[1] - incorrect_bins[0], alpha=0.5, label=f'Incorrect: {1-val_acc:.4f}')
        plt.axvline(threshold, color='r', linestyle='dashed', linewidth=1, label=f'Threshold: {threshold:.4f}')
        plt.legend(loc='upper right')
        plt.title(f'Confidence Histogram - {dataset}')
        plt.xlabel('Confidence')
        plt.ylabel('Frequency')
        plt.savefig(f'{log_dir_name}/histogram_{os.path.basename(dataset)}.png')
        plt.close()

        num_total_prob = len(correct_probs) + len(incorrect_probs)
        prob_tp = np.sum(correct_probs > threshold) / num_total_prob
        prob_fp = np.sum(incorrect_probs > threshold) / num_total_prob
        prob_tn = np.sum(incorrect_probs <= threshold) / num_total_prob
        prob_fn = np.sum(correct_probs <= threshold) / num_total_prob

        prob_confident = prob_tp + prob_fp
        prob_not_confident = prob_tn + prob_fn
        prob_correct = prob_tp + prob_fn
        prob_incorrect = prob_fp + prob_tn

        log_file.write(
            f"[Validation Result]\n"
            f" - DATASET  : {dataset}\n"
            f" - VAL LOSS : {val_loss:.4f}\n"
            f" - VAL ACC  : {val_acc:.4f}\n"
            f" - Threshold: {threshold:.4f} (lowest failure prob.)\n"
            f" - Confusion Matrix:\n"
            f"              |  Correct | Incorrect |    Sum |\n"
            f"    Confident |   {prob_tp:6.4f} |    {prob_fp:6.4f} | {prob_confident:6.4f} |\n"
            f"    Not Conf. |   {prob_fn:6.4f} |    {prob_tn:6.4f} | {prob_not_confident:6.4f} |\n"
            f"    Sum       |   {prob_correct:6.4f} |    {prob_incorrect:6.4f} | {prob_correct + prob_incorrect:6.4f} |\n"
            f" - F1 Score: {2 * np.sum(correct_probs > threshold) / (2 * np.sum(correct_probs > threshold) + np.sum(incorrect_probs > threshold) + np.sum(correct_probs <= threshold)):.4f}\n"
            f"\n"
        )

        log_file.flush()

    log_file.close()
