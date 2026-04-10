import os
import torch
from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

from utils.FGAesQ import FGAesQ
from utils.DiffToken import DiffToken


def load_model(model_path, device):
    model = FGAesQ(pretrained_path=None).to(device)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model


class AestheticInferenceDataset(Dataset):
    def __init__(self, image_files, image_dir, clip_model):
        self.image_files = image_files
        self.image_dir = image_dir
        self.preprocessor = DiffToken(
            clip_model=clip_model,
            patch_selection='random'
        )

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_name = self.image_files[idx]
        image_path = os.path.join(self.image_dir, image_name)
        
        try:
            image_pil = Image.open(image_path).convert('RGB')
        except Exception:
            return self.__getitem__(0)
        
        patches, pos_embed, mask_tokens = self.preprocessor.process_image(image_pil, is_train=False)
        
        return {
            'image_name': image_name,
            'patches': patches,
            'pos_embed': pos_embed,
            'mask': mask_tokens
        }


def get_image_files(folder_path):
    supported_formats = ('.jpg', '.jpeg', '.png', '.bmp', '.webp', '.JPG', '.JPEG', '.PNG')
    image_files = [f for f in os.listdir(folder_path) if f.endswith(supported_formats)]
    return sorted(image_files)


def predict_single_image(model, image_path, device):
    try:
        preprocessor = DiffToken(
            clip_model=model.clip_model,
            patch_selection='random'
        )
        
        image = Image.open(image_path).convert('RGB')
        patches, pos_embed, mask = preprocessor.process_image(image, is_train=False)
        
        patches = patches.unsqueeze(0).to(device)
        pos_embed = pos_embed.unsqueeze(0).to(device)
        mask = mask.unsqueeze(0).to(device)
        
        with torch.no_grad():
            output = model(patches, pos_embed, mask)
            score = output.item()
            return score
    
    except Exception:
        return None


def inference_aesthetic_single(model_path, input_path, output_txt=None, device='cuda'):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input not found: {input_path}")
    
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    
    model = load_model(model_path, device)
    score = predict_single_image(model, input_path, device)
    
    if score is not None:
        print(f"Image: {os.path.basename(input_path)}")
        print(f"Score: {score:.4f}")
    else:
        print("Inference failed")


def inference_aesthetic_batch(model_path, input_folder, output_txt=None, batch_size=128, device='cuda'):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not os.path.exists(input_folder):
        raise FileNotFoundError(f"Folder not found: {input_folder}")
    
    if output_txt:
        os.makedirs(os.path.dirname(output_txt) if os.path.dirname(output_txt) else '.', exist_ok=True)
    
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    
    model = load_model(model_path, device)
    image_files = get_image_files(input_folder)
    
    if not image_files:
        print("No images found")
        return
    
    dataset = AestheticInferenceDataset(
        image_files=image_files,
        image_dir=input_folder,
        clip_model=model.clip_model
    )
    
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    all_scores = []
    all_image_names = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Processing"):
            patches = batch['patches'].to(device)
            pos_embeds = batch['pos_embed'].to(device)
            masks = batch['mask'].to(device)
            image_names = batch['image_name']
            
            outputs = model(patches, pos_embeds, masks)
            scores = outputs.squeeze().cpu().numpy()
            
            if len(image_names) == 1:
                scores = [scores.item()]
            
            all_scores.extend(scores)
            all_image_names.extend(image_names)
    
    results = sorted(zip(all_image_names, all_scores), key=lambda x: x[1], reverse=True)
    
    for i, (img_name, score) in enumerate(results, 1):
        print(f"{i:3d}. {img_name:50s}  {score:.4f}")
    
    if output_txt:
        with open(output_txt, 'w', encoding='utf-8') as f:
            f.write(f"Total: {len(results)}\n")
            f.write(f"{'='*60}\n\n")
            for i, (img_name, score) in enumerate(results, 1):
                f.write(f"{i:3d}. {img_name:50s}  {score:.6f}\n")
        print(f"Results saved: {output_txt}")


def main():
    MODEL_PATH = "/home/wyy2/Sdd_D/WJJ/wjj/MLLMS/FGAesQ/utils/best.pt"  # Path to trained model
    INPUT_PATH = "/home/wyy2/Sdd_D/WJJ/wjj/MLLMS/FGAesQ/DATA/test_data_Aesthetic" # Input: single image path or folder path
    OUTPUT_TXT = "/home/wyy2/Sdd_D/WJJ/wjj/MLLMS/FGAesQ/DATA/single_result.txt" # Output txt path (only for folder mode, None to auto-generate)
    DEVICE = "cuda"
    BATCH_SIZE = 128
    
    output_txt = OUTPUT_TXT
    if output_txt is None and os.path.isdir(INPUT_PATH):
        folder_name = os.path.basename(os.path.normpath(INPUT_PATH))
        output_txt = f"aesthetic_scores_{folder_name}.txt"
    
    if os.path.isfile(INPUT_PATH):
        inference_aesthetic_single(
            model_path=MODEL_PATH,
            input_path=INPUT_PATH,
            output_txt=output_txt,
            device=DEVICE
        )
    elif os.path.isdir(INPUT_PATH):
        inference_aesthetic_batch(
            model_path=MODEL_PATH,
            input_folder=INPUT_PATH,
            output_txt=output_txt,
            batch_size=BATCH_SIZE,
            device=DEVICE
        )
    else:
        print(f"Invalid path: {INPUT_PATH}")


if __name__ == "__main__":
    main()