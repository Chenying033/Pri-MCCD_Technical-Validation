import os
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
from utils import *
from solver import Solver
from config import get_args
from data_loader import get_loader


def extract_label_from_filename(filename):
    try:
        parts = filename.split('_')
        # Find the part that ends with a number before the extension
        for part in reversed(parts):
            if '.' in part:
                label_str = part.split('.')[0]
                if label_str.isdigit():
                    return int(label_str)
        # Fallback if the above doesn't work as expected
        label_str = parts[-1].split('.')[0]
        return int(label_str)
    except:
        # Handle cases where filename might not follow the expected format
        print(f"Warning: Could not extract label from filename: {filename}. Returning -1.")
        return -1 # Return a default value or handle error as appropriate

def plot_confusion_matrix(cm, classes, filename):
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    # plt.title("Normalized Confusion Matrix", fontsize=20)
    plt.colorbar()

    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, fontsize=20)
    plt.yticks(tick_marks, classes, fontsize=20)

    thresh = cm.max() / 2.0
    for i, j in np.ndindex(cm.shape):
        plt.text(j, i, f"{cm[i, j]:.2f}",
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black",
                 fontsize=20)

    plt.tight_layout()
    plt.ylabel("True Label", fontsize=20)
    plt.xlabel("Predicted Label", fontsize=20)


    plt.savefig(filename)
    plt.close()

if __name__ == '__main__':
    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    
    args = get_args()
    args.n_class = 3  
    args.d_ain = 768
    args.d_vin = 512
    args.d_audio_in = 768
    args.d_audio_hidden = 128
    args.d_pose = 272

    generator = torch.Generator(device='cpu')
    # test
    test_data_dir = '/data1'
    test_audio_dir = '/data1'
    test_pose_dir = '/data1/features_mtcnn_head_pose_aggregated'
    test_is09_dir = '/data1/IS09NPZ'
    test_loader = get_loader(
        data_dir='/data1',
        pose_dir='/data1/features_mtcnn_head_pose_aggregated',
        audio_dir='/data1',
        is09_dir = test_is09_dir,  
        batch_size=1,
        phase='test',
        shuffle=False,
        generator=torch.Generator(device='cpu')
    )
    solver = Solver(args, train_loader=None, dev_loader=None, test_loader=test_loader, is_train=False)
    # state_dict = torch.load('/data1/outputs/batch_size32_best_model.pth')
    state_dict = torch.load('/data1/outputs/LSTM_trans.pth')
    solver.model.load_state_dict(state_dict, strict=True)  
    solver.model.eval()
    solver.model.to(device)

    all_preds = []
    all_labels = []
    filenames = []
    all_probs = [] # Store probabilities


    with torch.no_grad():
        for i, batch in enumerate(test_loader): # Added enumerate to get index
           
            frames, pose_features, audio_features, is09_features, label = batch

            frames = frames.to(device)
            pose_features = pose_features.to(device)
            audio_features = audio_features.to(device)
            is09_features =is09_features.to(device)
            label = label.to(device)

            # Get model logits
            preds = solver.model(frames, pose_features, audio_features, is09_features, label)[0]  # 使用 solver.model

            # Ensure preds is [batch_size, n_class]
            if preds.dim() == 1:
                preds = preds.unsqueeze(0)

            # Calculate probabilities
            prob = torch.softmax(preds, dim=-1).cpu().detach().numpy()[0]

            # Get predicted class
            pred = torch.argmax(preds, dim=-1).cpu().detach().numpy()[0]

            all_preds.append(pred)
            all_labels.append(label.cpu().numpy()[0])
            all_probs.append(prob) # Append probabilities

            # Get filename using the index
            filenames.append(test_loader.dataset.data[i])


    # Calculate evaluation metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='macro')
    f1 = f1_score(all_labels, all_preds, average='macro')
    recall = recall_score(all_labels, all_preds, average='macro')

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"F1-score: {f1:.4f}")
    print(f"Recall: {recall:.4f}")

    # Generate normalized confusion matrix
    cm = confusion_matrix(all_labels, all_preds, normalize='true')

    # Plot normalized confusion matrix
    output_img_path = "/data1/LSTM_trans.jpg"
    plot_confusion_matrix(cm, classes=['Class 0', 'Class 1', 'Class 2'], filename=output_img_path)

    # Save results to file
    output_dir = '/data1/outputs'
    output_txt_path = os.path.join(output_dir, 'LSTM_trans.txt')
    with open(output_txt_path, 'w') as fw:
        # Optionally write metrics at the beginning
        # fw.write(f'Accuracy: {accuracy:.4f}\n')
        # fw.write(f'Precision: {precision:.4f}\n')
        # fw.write(f'F1-score: {f1:.4f}\n')
        # fw.write(f'Recall: {recall:.4f}\n')
        # fw.write('filename true_label pred_label prob_0 prob_1 prob_2\n') # Header line

        for i in range(len(all_preds)):
            file_name = os.path.basename(filenames[i])
            true_label = all_labels[i] # Use the stored true label from the batch
            pred_label = all_preds[i]
            probs = all_probs[i]
            # Ensure we have probabilities for all 3 classes
            if len(probs) == 3:
                 fw.write(f'{file_name} {true_label} {pred_label} {probs[0]:.6f} {probs[1]:.6f} {probs[2]:.6f}\n')
            elif len(probs) < 3:
                 # Handle cases with fewer than 3 probabilities if necessary, pad with zeros
                 padded_probs = np.pad(probs, (0, 3 - len(probs)), 'constant')
                 fw.write(f'{file_name} {true_label} {pred_label} {padded_probs[0]:.6f} {padded_probs[1]:.6f} {padded_probs[2]:.6f}\n')
            # No explicit handling for > 3 probabilities as n_class is set to 3

    print(f"Results have been written to {output_txt_path} in the format: filename true_label pred_label prob_0 prob_1 prob_2")