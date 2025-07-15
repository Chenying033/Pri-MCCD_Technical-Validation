# import os
# import random
# import shutil
#

# raw_data_folder = '/data1/cy/raw_data/positive/simt'
# data_folder = '/data1/cy/Multimodal-Infomax/data'
# train_folder = os.path.join(data_folder, 'train')
# validation_folder = os.path.join(data_folder, 'valid')
# test_folder = os.path.join(data_folder, 'test')

# folders_to_create = [data_folder, train_folder, validation_folder, test_folder]
# for folder in folders_to_create:
#     if not os.path.exists(folder):
#         os.makedirs(folder)

# video_files = [f for f in os.listdir(raw_data_folder) if f.endswith(('.mp4', '.avi', '.mkv', '.mpg'))]
#

# random.shuffle(video_files)

# total_files = len(video_files)
# train_size = int(total_files * 0.7)
# validation_size = int(total_files * 0.2)
# test_size = total_files - train_size - validation_size

# train_files = video_files[:train_size]
# validation_files = video_files[train_size:train_size + validation_size]
# test_files = video_files[train_size + validation_size:]
#

# for file in train_files:
#     shutil.move(os.path.join(raw_data_folder, file), os.path.join(train_folder, file))
# for file in validation_files:
#     shutil.move(os.path.join(raw_data_folder, file), os.path.join(validation_folder, file))
# for file in test_files:
#     shutil.move(os.path.join(raw_data_folder, file), os.path.join(test_folder, file))
#
# # 输出每个集合的文件数量
# print(f"train: {len(train_files)}files")
# print(f"valid: {len(validation_files)}files")
# print(f"test: {len(test_files)}files")



