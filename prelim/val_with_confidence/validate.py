
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

DEVICE = 1

model_list = [
    'caformer_b36.sail_in22k_ft_in1k',
    'caformer_m36.sail_in22k_ft_in1k',
    'caformer_s36.sail_in22k_ft_in1k',

    # 'vit_huge_patch14_clip_224.laion2b_ft_in12k_in1k',
    # 'vit_large_patch14_clip_224.openai_ft_in12k_in1k',
    # 'vit_base_patch8_224.augreg2_in21k_ft_in1k',
    
    # 'convformer_b36.sail_in22k_ft_in1k',
    # 'convformer_m36.sail_in22k_ft_in1k',
    # 'convformer_s36.sail_in22k_ft_in1k',
    
    # 'deit3_huge_patch14_224.fb_in22k_ft_in1k',
    # 'deit3_large_patch16_224.fb_in22k_ft_in1k',
    # 'deit3_medium_patch16_224.fb_in22k_ft_in1k',
    # 'deit3_small_patch16_224.fb_in22k_ft_in1k',
    
    # 'nextvit_base.bd_ssld_6m_in1k',
    # 'resnet152.a1h_in1k',
    # 'rexnetr_200.sw_in12k_ft_in1k',
]

dataset_list = [
    '/data/ImageNet-1k/val',
    '/data/ImageNet-1k/val_sampled/gaussian_2',
    '/data/ImageNet-1k/val_sampled/gaussian_4',
    '/data/ImageNet-1k/val_sampled/gaussian_8',
    # '/data/ImageNet-1k/val_sampled/gaussian_16',
    '/data/ImageNet-1k/val_sampled/subsample_2',
    '/data/ImageNet-1k/val_sampled/subsample_4',
    '/data/ImageNet-1k/val_sampled/subsample_8',
    # '/data/ImageNet-1k/val_sampled/subsample_16',
    '/data/ImageNet-1k/val_sampled/avgsample_2',
    '/data/ImageNet-1k/val_sampled/avgsample_4',
    '/data/ImageNet-1k/val_sampled/avgsample_8',
    # '/data/ImageNet-1k/val_sampled/avgsample_16',
]

criterion = nn.CrossEntropyLoss().cuda(DEVICE)

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

    img = torch.tensor(img).cuda(DEVICE).transpose(1, 3).transpose(2, 3).float()

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
        data, target = data.cuda(DEVICE), target.cuda(DEVICE)
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

    model = timm.create_model(mname, pretrained=True, num_classes=1000).cuda(DEVICE)
    nparams = sum(p.numel() for p in model.parameters())
    
    # measure latency
    num_repeat = 100
    start_time = time.time()
    for _ in range(num_repeat):
        model(torch.randn(1, 3, 224, 224).cuda(DEVICE))
    latency = (time.time() - start_time) / num_repeat

    log_file.write(
        f"[Model Info]\n"
        f" - MODEL    : {mname}\n"
        f" - PARAMS   : {nparams:,d}\n"
        f" - LATENCY  : {latency:.4f}\n"
        f"\n"
    )
    log_file.flush()


    desired_acc = 0
    
    for dataset in dataset_list:

        # validation
        # pass if the probability data is saved
        if os.path.exists(f'{log_dir_name}/npy/correct_probs_{os.path.basename(dataset)}.npy') and os.path.exists(f'{log_dir_name}/npy/incorrect_probs_{os.path.basename(dataset)}.npy'):
            correct_probs = np.load(f'{log_dir_name}/npy/correct_probs_{os.path.basename(dataset)}.npy')
            incorrect_probs = np.load(f'{log_dir_name}/npy/incorrect_probs_{os.path.basename(dataset)}.npy')

            val_acc = (len(correct_probs) / (len(correct_probs) + len(incorrect_probs)))
            val_loss = -1

        else:
            resize_big = (dataset == '/data/ImageNet-1k/val')
            _, val_loader = get_imnet1k_dataloader(root=dataset, batch_size=BATCH_SIZE, resize_big=resize_big)
            val_loss, val_acc, additional_info = validate(val_loader, model, criterion)

            correct_probs = np.array(additional_info['correct_probs'])
            incorrect_probs = np.array(additional_info['incorrect_probs'])

            # save the probability data
            if not os.path.exists(f'{log_dir_name}/npy'):
                os.mkdir(f'{log_dir_name}/npy')
            np.save(f'{log_dir_name}/npy/correct_probs_{os.path.basename(dataset)}.npy', correct_probs)
            np.save(f'{log_dir_name}/npy/incorrect_probs_{os.path.basename(dataset)}.npy', incorrect_probs)

        correct_cdf = lambda x: np.sum(correct_probs > x) / len(correct_probs)
        incorrect_cdf = lambda x: np.sum(incorrect_probs > x) / len(incorrect_probs)

        if str(os.path.basename(dataset)) == 'val':
            desired_acc = val_acc
            print(f"Val Acc: {val_acc:.4f}")

        P_A = len(correct_probs) / (len(correct_probs) + len(incorrect_probs))
        P_B = 1 - P_A

        # Fine the threshold that minimizes the failure probability
        def prob_failure(x):
            return P_A * (1 - correct_cdf(x)) + P_B * incorrect_cdf(x)

        def accepted_acc(x):
            num_correct = np.sum(correct_probs > x)
            num_incorrect = np.sum(incorrect_probs > x)

            return num_correct / (num_correct + num_incorrect)
        
        thresholds = np.linspace(0, 1, 1000)
        failure_probs = [prob_failure(t) for t in thresholds]
        accepted_accs = [accepted_acc(t) for t in thresholds]
        
        
        threshold_minfail = thresholds[np.argmin(failure_probs)]
        if max(accepted_accs) < desired_acc or len(accepted_accs) == 0:
            threshold_desired = threshold_minfail
            print(f'Using threshold_minfail: {threshold_minfail:.4f}')
            log_file.write(
                f" ! No threshold found for desired accuracy {desired_acc:.4f}\n"
                f" ! Using threshold_minfail: {threshold_minfail:.4f}\n"
                f"\n"
            )
        else:
            for i in range(len(thresholds)):
                if accepted_accs[i] >= desired_acc:
                    threshold_desired = thresholds[i]
                    break
            print(f'Using threshold_desired: {threshold_desired:.4f}')
        

        # Create histogram data
        correct_hist, correct_bins = np.histogram(correct_probs, bins=50, density=True)
        incorrect_hist, incorrect_bins = np.histogram(incorrect_probs, bins=50, density=True)

        # Scale the histogram to match the density
        correct_hist = correct_hist * P_A
        incorrect_hist = incorrect_hist * P_B

        # Plot the histogram
        plt.bar(correct_bins[:-1], correct_hist, width=correct_bins[1] - correct_bins[0], alpha=0.5, label=f'Correct: {val_acc:.4f}')
        plt.bar(incorrect_bins[:-1], incorrect_hist, width=incorrect_bins[1] - incorrect_bins[0], alpha=0.5, label=f'Incorrect: {1-val_acc:.4f}')
        plt.axvline(threshold_minfail, color='r', linestyle='dashed', linewidth=1, label=f'Threshold (minfail): {threshold_minfail:.4f}')
        plt.axvline(threshold_desired, color='g', linestyle='dashed', linewidth=1, label=f'Threshold (desired): {threshold_desired:.4f}')
        plt.legend(loc='upper right')
        plt.title(f'Confidence Histogram - {dataset}')
        plt.xlabel('Confidence')
        plt.ylabel('Frequency')
        plt.savefig(f'{log_dir_name}/histogram_{os.path.basename(dataset)}.png')
        plt.close()

        log_file.write(
            f"[Validation Result]\n"
            f" - DATASET  : {dataset}\n"
            f" - VAL LOSS : {val_loss:.4f}\n"
            f" - VAL ACC  : {val_acc:.4f}\n"
            f"\n"
        )

        num_total_prob = len(correct_probs) + len(incorrect_probs)
        prob_tp = np.sum(correct_probs > threshold_minfail) / num_total_prob
        prob_fp = np.sum(incorrect_probs > threshold_minfail) / num_total_prob
        prob_tn = np.sum(incorrect_probs <= threshold_minfail) / num_total_prob
        prob_fn = np.sum(correct_probs <= threshold_minfail) / num_total_prob

        prob_confident = prob_tp + prob_fp
        prob_not_confident = prob_tn + prob_fn
        prob_correct = prob_tp + prob_fn
        prob_incorrect = prob_fp + prob_tn

        log_file.write(
            f" - Thres (Minfail) : {threshold_minfail:.4f}\n"
            f" - Confusion Matrix:\n"
            f"              |  Correct | Incorrect |    Sum |\n"
            f"    Confident |   {prob_tp:6.4f} |    {prob_fp:6.4f} | {prob_confident:6.4f} | (acc. accepted: {prob_tp / prob_confident * 100:.2f} %)\n"
            f"    Not Conf. |   {prob_fn:6.4f} |    {prob_tn:6.4f} | {prob_not_confident:6.4f} |\n"
            f"    Sum       |   {prob_correct:6.4f} |    {prob_incorrect:6.4f} | {prob_correct + prob_incorrect:6.4f} |\n"
            f"\n"
        )

        prob_tp = np.sum(correct_probs > threshold_desired) / num_total_prob
        prob_fp = np.sum(incorrect_probs > threshold_desired) / num_total_prob
        prob_tn = np.sum(incorrect_probs <= threshold_desired) / num_total_prob
        prob_fn = np.sum(correct_probs <= threshold_desired) / num_total_prob

        prob_confident = prob_tp + prob_fp
        prob_not_confident = prob_tn + prob_fn
        prob_correct = prob_tp + prob_fn
        prob_incorrect = prob_fp + prob_tn

        log_file.write(
            f" - Thres (Desired) : {threshold_desired:.4f}\n"
            f" - Confusion Matrix:\n"
            f"              |  Correct | Incorrect |    Sum |\n"
            f"    Confident |   {prob_tp:6.4f} |    {prob_fp:6.4f} | {prob_confident:6.4f} | (acc. accepted: {prob_tp / prob_confident * 100:.2f} %)\n"
            f"    Not Conf. |   {prob_fn:6.4f} |    {prob_tn:6.4f} | {prob_not_confident:6.4f} |\n"
            f"    Sum       |   {prob_correct:6.4f} |    {prob_incorrect:6.4f} | {prob_correct + prob_incorrect:6.4f} |\n"
            f"\n"
        )

        log_file.flush()

    log_file.close()
