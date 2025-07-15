import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
import json
import os  


# EXTRACT_FREQUENCY = 10


def extract_frames(video_path):
    video = cv2.VideoCapture(video_path)
    frames_list = []
    while True:
        success, frame = video.read()
        if not success:
            break
     
        frame = cv2.resize(frame, (256, 256))
        
        # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames_list.append(frame)
    video.release()
  
    if not frames_list:
        return np.array([])  

    frames = np.stack(frames_list, axis=0)
    frames = np.transpose(frames, (0, 3, 1, 2))  
    return frames



class MSADataset(Dataset):
    def __init__(self, data_dir, pose_dir, audio_dir, is09_dir, batch_size, phase):
        self.data = []
        self.pose_dir = pose_dir  
        self.audio_dir = audio_dir
        self.is09_dir = is09_dir
        self.phase = phase
        self.batch_size = batch_size

        phase_dir = os.path.join(data_dir, phase)
        print(f"Loading data from: {phase_dir}")
        if not os.path.exists(phase_dir):
            raise ValueError(f"Data directory does not exist: {phase_dir}")

        for root, _, files in os.walk(phase_dir):
            for file in files:
                if file.endswith('.mp4'):
                    self.data.append(os.path.join(root, file))
        print(f"Number of samples in dataset: {len(self.data)}")
        if not self.data:
            print(f"Warning: No .mp4 files found in {phase_dir}")

    def __getitem__(self, index):
        video_path = self.data[index]
        frames = extract_frames(video_path)

        if frames.size == 0: 
            print(f"Warning: No frames extracted from video: {video_path}. Skipping sample.")
            
            return None  

        frames = torch.from_numpy(frames).float().cpu()
        if frames.ndim != 4:
            raise ValueError(
                f"Expected frames to have 4 dimensions, but got {frames.ndim} with shape {frames.shape} for video: {video_path}")


        base_name = os.path.basename(video_path)
        try:
            label_part = base_name.split('_')[-1].split('.')[0]
            label = int(label_part)
        except (IndexError, ValueError):
            print(f"Warning: Could not parse label from filename: {base_name}. Assigning label -1.")
            label = -1

       
        pose_base_name = base_name.split('.')[0]  # e.g., 'cuckoo10_2'
        pose_file = f"{pose_base_name}_head_pose_features_mtcnn_aggregated.json"

        
        pose_path = os.path.join(self.pose_dir, self.phase, pose_file)

        if not os.path.exists(pose_path):
            print(f"Aggregated pose file not found: {pose_path} for video {video_path}. Skipping sample.")
            return None  

        with open(pose_path, 'r') as f:
            pose_data_json = json.load(f)

        
        extracted_pose_features = []
        for frame_entry in pose_data_json:
            aggregated_pose = frame_entry.get("aggregated_pose", {})
            pitch = aggregated_pose.get("avg_pitch", 0.0)
            yaw = aggregated_pose.get("avg_yaw", 0.0)
            roll = aggregated_pose.get("avg_roll", 0.0)
            extracted_pose_features.append([pitch, yaw, roll])

        
        pose_features = np.array(extracted_pose_features, dtype=np.float32)
       
        pose_features = torch.from_numpy(pose_features).float().cpu()
        # ===================================================

        
        audio_file = f"{base_name.split('.')[0]}.npz"
        audio_path = os.path.join(self.audio_dir, self.phase, audio_file)
        if not os.path.exists(audio_path):
            print(f"Audio file not found: {audio_path} for video {video_path}. Skipping sample.")
            return None 
        audio_data = np.load(audio_path)
        audio_features = torch.from_numpy(audio_data['feat']).float().cpu()

       
        is09_file = f"IS09{base_name.split('.')[0]}.npy"
        is09_path = os.path.join(self.is09_dir, self.phase, is09_file)
        if not os.path.exists(is09_path):
            print(f"IS09 file not found: {is09_path} for video {video_path}. Skipping sample.")
            return None  
        is09_features = torch.from_numpy(np.load(is09_path)).float().cpu()

        return frames, pose_features, audio_features, is09_features, label

    def __len__(self):
        return len(self.data)


def collate_fn(batch):
    

    batch = [item for item in batch if item is not None]
    if not batch:
        

        raise RuntimeError(
            "Batch is empty after filtering out invalid samples. Check if files exist or videos have frames.")

    frames, poses, audios, is09s, labels = zip(*batch)

    target_sequence_length = 150  

   

    frames_padded = []
    for frame_seq in frames:  # frame_seq 是 (num_frames, C, H, W)
        frame_seq = frame_seq.cpu()
        if frame_seq.size(0) < target_sequence_length:
            padding_size = target_sequence_length - frame_seq.size(0)
            padding_tensor = torch.zeros(padding_size, *frame_seq.size()[1:], device='cpu')
            frame_seq = torch.cat((frame_seq, padding_tensor), dim=0)
        else:
            frame_seq = frame_seq[:target_sequence_length]
        frames_padded.append(frame_seq)
    frames_padded = torch.stack(frames_padded).cpu()

   

    processed_poses = []
    for pose_seq in poses:  # pose_seq 是 (seq_len, 3)
        pose_seq = pose_seq.cpu()
        if pose_seq.size(0) < target_sequence_length:  

            padding_size = target_sequence_length - pose_seq.size(0)
           

            padding_tensor = torch.zeros(padding_size, pose_seq.size(1), device='cpu')
            pose_seq = torch.cat((pose_seq, padding_tensor), dim=0)
        else:
            pose_seq = pose_seq[:target_sequence_length]  

        processed_poses.append(pose_seq)
    poses_padded = torch.stack(processed_poses).cpu()  # 堆叠成 (batch_size, seq_len, 3)

   
    audios_padded = []
    for audio_seq in audios:  # audio_seq 预期是 (seq_len, feature_dim)
        audio_seq = audio_seq.cpu()

        
        if audio_seq.ndim == 3 and audio_seq.size(1) == 1:
            audio_seq = audio_seq.squeeze(1)  

        if audio_seq.size(0) < target_sequence_length:
            padding_size = target_sequence_length - audio_seq.size(0)
            padding_tensor = torch.zeros(padding_size, audio_seq.size(1), device='cpu')
            audio_seq = torch.cat((audio_seq, padding_tensor), dim=0)
        else:
            audio_seq = audio_seq[:target_sequence_length, :]
        audios_padded.append(audio_seq)
    audios_padded = torch.stack(audios_padded).cpu()

    is09s_stacked = torch.stack(is09s).cpu()

    labels_tensor = torch.tensor(labels, dtype=torch.long, device='cpu')

    return frames_padded, poses_padded, audios_padded, is09s_stacked, labels_tensor



def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def worker_init_fn(worker_id):
    worker_seed = torch.initial_seed() % (2 ** 32) + worker_id
    np.random.seed(worker_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(worker_seed)


# 获取 DataLoader
def get_loader(data_dir, pose_dir, audio_dir, is09_dir, batch_size, phase='train', shuffle=True, generator=None):
    dataset = MSADataset(data_dir, pose_dir, audio_dir, is09_dir, batch_size, phase)
    print(f"Data loader phase: {phase}, number of samples: {len(dataset)}")

    if generator is None:
        generator_device = 'cuda' if torch.cuda.is_available() else 'cpu'
        generator = torch.Generator(device=generator_device)

    data_loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
        worker_init_fn=worker_init_fn,
        num_workers=0,  
        pin_memory=torch.cuda.is_available(),
        generator=generator
    )

    return data_loader