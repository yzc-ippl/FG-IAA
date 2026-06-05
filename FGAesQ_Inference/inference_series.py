import os
import platform
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from itertools import combinations
from pathlib import Path
from collections import defaultdict
import numpy as np

from utils.FGAesQ import FGAesQ
from utils.DiffToken import DiffToken


def is_ubuntu_system():
    if platform.system().lower() != 'linux':
        return False

    try:
        with open('/etc/os-release', 'r', encoding='utf-8') as f:
            for line in f:
                normalized = line.strip().lower().replace('"', '')
                if normalized == 'id=ubuntu':
                    return True
                if normalized.startswith('id_like=') and 'ubuntu' in normalized:
                    return True
    except OSError:
        return False

    return False


def get_num_workers_for_system(default_num_workers):
    system_name = platform.platform()
    num_workers = default_num_workers if is_ubuntu_system() else 0
    print(f"System: {system_name}")
    return num_workers


def load_model(model_path, device):
    model = FGAesQ(pretrained_path=None).to(device)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model


class SeriesInferenceDataset(Dataset):
    def __init__(self, pairs, series_data, clip_model):
        self.pairs = pairs
        self.series_data = series_data
        self.preprocessor = DiffToken(
            clip_model=clip_model,
            patch_selection='similarity'
        )

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        pair = self.pairs[idx]
        series_id = pair['series_id']
        img1_idx = pair['img1_idx']
        img2_idx = pair['img2_idx']
        img1_name = pair['img1_name']
        img2_name = pair['img2_name']
        
        series_folder = self.series_data[series_id]['folder']
        img1_path = os.path.join(series_folder, img1_name)
        img2_path = os.path.join(series_folder, img2_name)
        
        try:
            img1 = Image.open(img1_path).convert('RGB')
            img2 = Image.open(img2_path).convert('RGB')
        except Exception:
            return self.__getitem__(0)
        
        pair_identifier = f"{series_id}_{img1_idx}_{img2_idx}"
        patches1, pos_embed1, mask1, patches2, pos_embed2, mask2 = \
            self.preprocessor.process_similarity_pair(img1, img2, is_train=False,pair_id=pair_identifier)
        
        return {
            'series_id': series_id,
            'img1_idx': img1_idx,
            'img2_idx': img2_idx,
            'patches1': patches1,
            'pos_embed1': pos_embed1,
            'mask1': mask1,
            'patches2': patches2,
            'pos_embed2': pos_embed2,
            'mask2': mask2
        }


def get_image_files(folder_path):
    valid_extensions = {'.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'}
    image_files = [f for f in os.listdir(folder_path) 
                   if os.path.isfile(os.path.join(folder_path, f)) and Path(f).suffix in valid_extensions]
    return sorted(image_files)


def collect_series_info(input_folder):
    series_data = {}
    subdirs = [d for d in os.listdir(input_folder) if os.path.isdir(os.path.join(input_folder, d))]
    
    if subdirs:
        series_folders = [(subdir, os.path.join(input_folder, subdir)) for subdir in sorted(subdirs)]
    else:
        series_name = os.path.basename(input_folder)
        series_folders = [(series_name, input_folder)]
    
    for series_name, series_folder in series_folders:
        try:
            normalized_name = str(int(series_name))
        except ValueError:
            normalized_name = series_name
        
        image_files = get_image_files(series_folder)
        
        if len(image_files) < 2:
            continue
        
        image_mapping = {}
        for img_name in image_files:
            try:
                img_idx = int(img_name.split('-')[1].split('.')[0])
                image_mapping[img_idx] = img_name
            except (IndexError, ValueError):
                continue
        
        if len(image_mapping) < 2:
            continue

        series_data[normalized_name] = {
            'folder': series_folder,
            'images': image_files,
            'image_mapping': image_mapping
        }
    
    return series_data


def create_pairs(series_data):
    pairs = []
    
    for series_name, data in series_data.items():
        image_indices = sorted(data['image_mapping'].keys())
        image_pairs = list(combinations(image_indices, 2))
        
        for img1_idx, img2_idx in image_pairs:
            img1_name = data['image_mapping'][img1_idx]
            img2_name = data['image_mapping'][img2_idx]

            pairs.append({
                'series_id': series_name,
                'img1_idx': img1_idx,
                'img2_idx': img2_idx,
                'img1_name': img1_name,
                'img2_name': img2_name
            })
    
    return pairs


def evaluate_pairs(model, pairs, series_data, device, batch_size=64, num_workers=16):
    dataset = SeriesInferenceDataset(
        pairs=pairs,
        series_data=series_data,
        clip_model=model.clip_model
    )
    
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    model.eval()
    image_scores_all = defaultdict(list)
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Processing"):
            patches1 = batch['patches1'].to(device)
            pos_embed1 = batch['pos_embed1'].to(device)
            mask1 = batch['mask1'].to(device)
            patches2 = batch['patches2'].to(device)
            pos_embed2 = batch['pos_embed2'].to(device)
            mask2 = batch['mask2'].to(device)
            
            series_ids = batch['series_id']
            img1_indices = batch['img1_idx']
            img2_indices = batch['img2_idx']
            
            scores1 = model(patches1, pos_embed1, mask1)
            scores2 = model(patches2, pos_embed2, mask2)
            
            for series_id, img1_idx, img2_idx, score1, score2 in zip(
                series_ids, img1_indices, img2_indices, scores1, scores2
            ):
                image_scores_all[(series_id, int(img1_idx))].append(float(score1))
                image_scores_all[(series_id, int(img2_idx))].append(float(score2))
    
    return image_scores_all


def apply_confidence_adjustment(image_scores_all, series_data):
    large_series = {series_id for series_id, data in series_data.items() 
                    if len(data['image_mapping']) >= 4}
    
    series_means = {}
    for series_id in large_series:
        series_scores = []
        for img_idx in series_data[series_id]['image_mapping'].keys():
            if (series_id, img_idx) in image_scores_all:
                series_scores.extend(image_scores_all[(series_id, img_idx)])
        if series_scores:
            series_means[series_id] = np.mean(series_scores)
    
    adjusted_image_scores = {}
    for (series_id, img_idx), scores in image_scores_all.items():
        count = len(scores)
        raw_avg = np.mean(scores)
        
        if series_id in large_series and series_id in series_means:
            series_mean = series_means[series_id]
            if count == 1:
                adjusted_score = 0.3 * raw_avg + 0.7 * series_mean
            elif count == 2:
                adjusted_score = 0.7 * raw_avg + 0.3 * series_mean
            else:
                adjusted_score = 0.9 * raw_avg + 0.1 * series_mean
        else:
            adjusted_score = raw_avg
        
        adjusted_image_scores[(series_id, img_idx)] = adjusted_score
    
    return adjusted_image_scores


def save_series_result(series_name, image_scores, image_mapping, output_folder):
    series_scores = {}
    for (sid, idx), score in image_scores.items():
        if sid == series_name:
            img_name = image_mapping[idx]
            series_scores[img_name] = score
    
    if not series_scores:
        return
    
    sorted_images = sorted(series_scores.items(), key=lambda x: x[1], reverse=True)
    
    output_file = os.path.join(output_folder, f"{series_name}_result.txt")
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(f"Series: {series_name}\n")
        f.write(f"Count: {len(series_scores)}\n")
        f.write(f"{'='*60}\n\n")
        f.write("Ranking: ")
        f.write("  ".join([img for img, _ in sorted_images]))
        f.write("\n\n")
        f.write("Scores:  ")
        f.write("  ".join([f"{score:.4f}" for _, score in sorted_images]))
        f.write("\n\n")
        f.write("Order: ")
        f.write(" > ".join([img for img, _ in sorted_images]))
        f.write("\n")


def inference_series_auto(model_path, input_folder, output_folder, device='cuda', 
                          batch_size=64, weighted=True, num_workers=16):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not os.path.exists(input_folder):
        raise FileNotFoundError(f"Folder not found: {input_folder}")
    
    os.makedirs(output_folder, exist_ok=True)
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    
    model = load_model(model_path, device)
    series_data = collect_series_info(input_folder)
    
    if len(series_data) == 0:
        print("No valid series found")
        return
    
    pairs = create_pairs(series_data)
    image_scores_all = evaluate_pairs(model, pairs, series_data, device, batch_size, num_workers)
    image_scores = apply_confidence_adjustment(image_scores_all, series_data)

    for series_name, data in series_data.items():
        save_series_result(series_name, image_scores, data['image_mapping'], output_folder)

    print(f"Completed: {output_folder}")


def main():
    MODEL_PATH = "/home/wyy2/Sdd_D/WJJ/wjj/MLLMS/FGAesQ/utils/best.pt" # Path to trained model
    INPUT_FOLDER = "/home/wyy2/Sdd_D/WJJ/wjj/MLLMS/FGAesQ1/huawei2026/top_10" # Path to input image series
    OUTPUT_FOLDER = "/home/wyy2/Sdd_D/WJJ/wjj/MLLMS/FGAesQ/DATA/series_result2"  # Path to save results
    DEVICE = "cuda:3"
    BATCH_SIZE = 64
    WEIGHTED = True
    NUM_WORKERS = 16  # non-Ubuntu systems auto force it to 0.

    NUM_WORKERS = get_num_workers_for_system(NUM_WORKERS)

    inference_series_auto(
        model_path=MODEL_PATH,
        input_folder=INPUT_FOLDER,
        output_folder=OUTPUT_FOLDER,
        device=DEVICE,
        batch_size=BATCH_SIZE,
        weighted=WEIGHTED,
        num_workers=NUM_WORKERS
    )


if __name__ == "__main__":
    main()
