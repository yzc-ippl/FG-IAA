# ==================== DiffToken.py (Production Version) ====================
import os
import torch
from torchvision import transforms
import random
import numpy as np
import math
import torch.nn.functional as F
from PIL import Image
from skimage.metrics import structural_similarity as ssim
import warnings
from utils.data_utils import (
    add_padding, image_to_patches, interpolate_positional_embedding, 
    padding, random_flip, random_rotate, pad_or_crop, lcm, create_binary_mask,
)

warnings.filterwarnings("ignore", category=UserWarning)

class DiffToken:
    def __init__(self, clip_model=None, 
                 patch_size=16, 
                 patch_stride=16, 
                 max_seq_len_from_original_res=511, 
                 initial_hidden_size=112,
                 patch_selection='similarity',
                 augmentation=None,
                 importance_threshold=0.5,
                 min_large_image_size=384):
        
        self.patch_size = patch_size
        self.patch_stride = patch_stride
        self.hidden_size = max_seq_len_from_original_res + 1
        self.num_scales = 2
        self.initial_hidden_size = initial_hidden_size
        self.patch_selection_strategy = patch_selection
        self.augmentation = augmentation
        self.importance_threshold = importance_threshold
        self.min_large_image_size = min_large_image_size
        
        self.scale_factor = [0.5, 1.0]
        self.scaled_patchsizes = [16, 32]
        
        self.clip_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), 
                                (0.26862954, 0.26130258, 0.27577711))
        ])
        self.original_transform = transforms.ToTensor()
        
        self.pos_embeds = {}
        self.clip_model = clip_model
        self.load_pos_embed_from_clip()
        self.interpolate_offset = 0.1

    def load_pos_embed_from_clip(self):
        if self.clip_model is not None:
            try:
                pos_embed = self.clip_model.visual.positional_embedding.clone().detach().cpu()
                if pos_embed.dim() == 2:
                    pos_embed = pos_embed.unsqueeze(0)
                self.pos_embed = pos_embed
                
                os.makedirs('utils', exist_ok=True)
                torch.save(self.pos_embed, 'utils/clip_vit_base_16_224.pt')
            except Exception as e:
                print(f"Failed to extract CLIP positional embedding: {e}")
                self.pos_embed = torch.randn(1, 197, 768) * 0.02
        else:
            self.pos_embed = torch.randn(1, 197, 768) * 0.02

    def augment(self, image, mask=None):
        if self.augmentation == 'HF' or self.augmentation == 'all':
            if random.random() < 0.5:
                image, mask = random_flip(image, mask)
        
        if self.augmentation == 'RR' or self.augmentation == 'all':
            if random.random() < 0.5:
                image, mask = random_rotate(image, mask)
        
        return image, mask

    def process_small_image(self, original_image):
        original_image = add_padding(original_image, self.patch_size)
        channels, H, W = original_image.size()
        n_crops_w = math.ceil(W / self.patch_size)
        n_crops_H = math.ceil(H / self.patch_size)
        size = (n_crops_H, n_crops_w)

        image_patches = image_to_patches(original_image, self.patch_size, self.patch_stride)
        input = torch.stack(image_patches)
        input = torch.cat((torch.zeros(1, input.shape[1], input.shape[2], input.shape[3]), input), dim=0)

        if size not in self.pos_embeds:
            patch_pos_embed, class_pos_embed = interpolate_positional_embedding(
                self.pos_embed, size, self.interpolate_offset, self.pos_embed.shape[-1])
            pos_embed = torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)
            self.pos_embeds[size] = pos_embed
        else:
            pos_embed = self.pos_embeds[size]

        pos_embed = pos_embed.squeeze(0)
        
        if input.shape[0] != pos_embed.shape[0]:
            raise ValueError("Position embedding length doesn't match token length")
        
        if input.shape[0] < self.hidden_size:
            mask = torch.zeros(input.shape[0])
            padded_area = self.hidden_size - input.shape[0]
            pad = [9] * padded_area
            mask = torch.cat((mask, torch.Tensor(pad)), 0)
            input = padding(input, self.hidden_size)
            pos_embed = padding(pos_embed.unsqueeze(-1).unsqueeze(-1), self.hidden_size).squeeze(-1).squeeze(-1)
        elif input.shape[0] > self.hidden_size:
            input, pos_embed = self.drop_tokens(input, pos_embed)
            mask = torch.zeros(self.hidden_size)
        else:
            mask = torch.zeros(self.hidden_size)

        return input, pos_embed, mask
    
    def process_large_image_multiscale(self, image):
        image = pad_or_crop(image, lcm(self.scaled_patchsizes))
        
        patch_sizes = self.scaled_patchsizes
        patch_strides = self.scaled_patchsizes
        
        image_patches = image_to_patches(image, patch_sizes[-1], patch_strides[-1])
        importance = self.compute_patch_importance(image_patches)
        
        n_patch_per_col = image.size()[-1] // patch_sizes[-1]
        n_patch_per_row = image.size()[-2] // patch_sizes[-1]
        
        n_patches = int(self.initial_hidden_size * self.importance_threshold / 4)
        
        selected_indices_fine = self.select_patches(importance, n_patches, 1, range(len(image_patches)))
        
        patches_fine = []
        for i in sorted(selected_indices_fine):
            patches = image_to_patches(image_patches[i], self.patch_size, self.patch_stride)
            patches_fine.extend(patches)
        
        mask_fine = [1] * len(patches_fine)
        
        selected_indices_coarse = [x for x in range(len(image_patches)) if x not in selected_indices_fine]
        remaining_patches = []
        for index in sorted(selected_indices_coarse):
            a = F.interpolate(image_patches[index].unsqueeze(0),
                            size=(self.scaled_patchsizes[0], self.scaled_patchsizes[0]),
                            mode='bicubic').squeeze(0)
            patch = image_to_patches(a, self.patch_size, self.patch_stride)
            remaining_patches.extend(patch)
        
        mask_coarse = [0] * len(remaining_patches)
        
        final = remaining_patches + patches_fine
        mask_ms = mask_coarse + mask_fine
        final = torch.stack(final)
        
        masks = []
        for i in range(2):
            p = patch_sizes[i] // self.patch_size
            if i == 0:
                mask = create_binary_mask((3, p * n_patch_per_row, p * n_patch_per_col), p, selected_indices_coarse)
            else:
                mask = create_binary_mask((3, p * n_patch_per_row, p * n_patch_per_col), p, selected_indices_fine)
            masks.append(mask)
        
        pos_embeds = self.prepare_multiscale_pos_embed(masks, self.pos_embed.shape[-1]).squeeze(0)
        
        final_tensor = torch.cat((torch.zeros(1, final.shape[1], final.shape[2], final.shape[3]), final), dim=0)
        mask_ms.insert(0, 0)
        
        if final_tensor.shape[0] != pos_embeds.shape[0]:
            raise ValueError("Position embedding length doesn't match token length")
        
        if final_tensor.shape[0] < self.hidden_size:
            input = padding(final_tensor, self.hidden_size)
            pos_embeds = padding(pos_embeds.unsqueeze(-1).unsqueeze(-1), self.hidden_size).squeeze(-1).squeeze(-1)
            padded_area = self.hidden_size - final_tensor.shape[0]
            pad = [9] * padded_area
            mask_ms = mask_ms + pad
            mask = torch.Tensor(mask_ms)
        elif final_tensor.shape[0] > self.hidden_size:
            input, pos_embeds, mask = self.drop_tokens(final_tensor, pos_embeds, torch.Tensor(mask_ms))
        else:
            input = final_tensor
            mask = torch.Tensor(mask_ms)
        
        return input, pos_embeds, mask

    def prepare_patches(self, original_image):
        if hasattr(self, '_current_importance') and hasattr(self, '_current_patch_types'):
            self._current_drop_importance = self._current_importance.copy()
            self._current_drop_types = self._current_patch_types.copy()
        
        channels, H, W = original_image.size()
        n_crops_w = math.ceil(W / self.patch_size)
        n_crops_H = math.ceil(H / self.patch_size)
        
        if n_crops_H * n_crops_w <= self.hidden_size:
            input, pos_embed, mask = self.process_small_image(original_image)
        else:
            input, pos_embed, mask = self.process_large_image_multiscale(original_image)
        
        return input, pos_embed, mask

    def prepare_multiscale_pos_embed(self, masks, dim):
        fine_pos_embeds = []
        for fine_mask in masks:
            fine_size = fine_mask.size()
            if fine_size not in self.pos_embeds:
                patch_pos_embed, class_pos_embed = interpolate_positional_embedding(
                    self.pos_embed, fine_size, self.interpolate_offset, dim)
                self.pos_embeds[fine_size] = torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)
                fine_pos_embed = patch_pos_embed[:, fine_mask.type(torch.bool).flatten(), :]
            else:
                patch_pos_embed = self.pos_embeds[fine_size][:, 1:, :]
                class_pos_embed = self.pos_embeds[fine_size][:, 0, :]
                fine_pos_embed = patch_pos_embed[:, fine_mask.type(torch.bool).flatten(), :]
            fine_pos_embeds.append(fine_pos_embed)

        fine_pos_embeds = torch.cat(fine_pos_embeds, dim=1)
        final_pos_embed = torch.cat([class_pos_embed.unsqueeze(0), fine_pos_embeds], dim=1)
        return final_pos_embed
    
    def compute_patch_importance(self, image_patches):
        if self.patch_selection_strategy == 'similarity':
            if hasattr(self, '_current_importance') and self._current_importance:
                importance_scores = self._current_importance
                adjusted_scores = importance_scores[:len(image_patches)]
                if len(adjusted_scores) < len(image_patches):
                    adjusted_scores.extend([random.random() for _ in range(len(image_patches) - len(adjusted_scores))])
                return [(score if score is not None else random.random(), i) for i, score in enumerate(adjusted_scores)]
            else:
                return [(0.5, i) for i in range(len(image_patches))]
        else:
            return [0] * len(image_patches)
    
    def select_patches(self, importance, n_patches, scale, total_patches):
        if self.patch_selection_strategy == 'similarity':
            if isinstance(importance[0], tuple):
                scores = [score for score, _ in importance]
                indices = [idx for _, idx in importance]
            else:
                scores = importance
                indices = list(range(len(importance)))
            
            indexed_scores = list(zip(indices, scores))
            indexed_scores.sort(key=lambda x: x[1], reverse=True)
            
            if scale == 1:
                total_available = len(indexed_scores)
                top_percent = max(1, int(total_available * self.importance_threshold))
                candidates = indexed_scores[:top_percent]
                
                if n_patches <= len(candidates):
                    selected_candidates = random.sample(candidates, n_patches)
                else:
                    selected_candidates = candidates
                
                selected_indices = [idx for idx, _ in selected_candidates]
                self._current_selected_indices = selected_indices
            else:
                selected_indices = random.sample(list(total_patches), n_patches)
            
            return selected_indices
        
        elif self.patch_selection_strategy == 'random':
            return random.sample(list(total_patches), n_patches)
        
        else:
            raise ValueError(f'Patch selection strategy "{self.patch_selection_strategy}" is not supported')
    
    def drop_tokens(self, input, pos_embed, mask_ms=None):
        elements_to_remove = input.shape[0] - self.hidden_size
        
        if elements_to_remove <= 0:
            if mask_ms is not None:
                return input, pos_embed, mask_ms
            else:
                return input, pos_embed
        
        if (self.patch_selection_strategy == 'similarity' and
            hasattr(self, '_current_drop_importance') and self._current_drop_importance and 
            hasattr(self, '_current_drop_types') and self._current_drop_types):
            return self.drop_tokens_with_priority(input, pos_embed, mask_ms, 
                                                 self._current_drop_importance, self._current_drop_types)
        
        indices_to_remove = random.sample(range(1, input.shape[0]), elements_to_remove)
        mask = torch.ones(input.shape[0], dtype=torch.bool)
        mask[indices_to_remove] = False
        input = input[mask, :, :, :]
        pos_embed = pos_embed[mask, :]
        
        if mask_ms is not None:
            mask_ms = mask_ms[mask]
            return input, pos_embed, mask_ms
        else:
            return input, pos_embed
    
    def drop_tokens_with_priority(self, input, pos_embed, mask_ms, importance_scores, patch_types):
        elements_to_remove = input.shape[0] - self.hidden_size
        actual_patches = input.shape[0] - 1
        
        if len(importance_scores) > actual_patches:
            importance_scores = importance_scores[:actual_patches]
            patch_types = patch_types[:actual_patches]
        elif len(importance_scores) < actual_patches:
            deficit = actual_patches - len(importance_scores)
            avg_importance = np.mean([s for s in importance_scores if s is not None]) if importance_scores else 0.5
            importance_scores.extend([avg_importance] * deficit)
            patch_types.extend(['unknown'] * deficit)
        
        all_patches = []
        important_patches = set()
        if hasattr(self, '_current_selected_indices'):
            important_patches = set(self._current_selected_indices)
        
        for i in range(actual_patches):
            token_idx = i + 1
            score = importance_scores[i] if importance_scores[i] is not None else random.random()
            is_important = i in important_patches
            
            if not is_important:
                all_patches.append((token_idx, score))
        
        if len(all_patches) >= elements_to_remove:
            patches_to_drop_data = random.sample(all_patches, elements_to_remove)
        else:
            patches_to_drop_data = all_patches
        
        indices_to_remove = [idx for idx, _ in patches_to_drop_data]
        
        remaining_to_remove = elements_to_remove - len(indices_to_remove)
        if remaining_to_remove > 0:
            important_patch_indices = [i + 1 for i in important_patches if i < actual_patches]
            if len(important_patch_indices) >= remaining_to_remove:
                additional_drops = random.sample(important_patch_indices, remaining_to_remove)
            else:
                additional_drops = important_patch_indices
            indices_to_remove.extend(additional_drops)
        
        mask = torch.ones(input.shape[0], dtype=torch.bool)
        mask[indices_to_remove] = False
        
        if mask_ms is not None:
            return input[mask], pos_embed[mask], mask_ms[mask]
        else:
            return input[mask], pos_embed[mask]

    def process_image(self, image_pil, is_train=False):
        if self.patch_selection_strategy == 'similarity':
            image_pil = self.ensure_large_image_size(image_pil)
        
        if is_train and self.augmentation is not None:
            image_pil, _ = self.augment(image_pil, None)
        
        image_tensor = self.clip_transform(image_pil)
        patches, pos_embed, mask_tokens = self.prepare_patches(image_tensor)
        
        return patches, pos_embed, mask_tokens

    def _setup_processing_state(self, pair_id=None):
        from utils.data_utils import _derive_patch_alignment_seed
    
        if pair_id is not None:
            state = _derive_patch_alignment_seed(pair_id)
            torch.manual_seed(state)
            np.random.seed(state)
            random.seed(state)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(state)

    def ensure_large_image_size(self, image_pil):
        width, height = image_pil.size
        n_crops_w = math.ceil(width / self.patch_size)
        n_crops_h = math.ceil(height / self.patch_size)
        current_patches = n_crops_w * n_crops_h
        
        if current_patches <= self.hidden_size:
            target_patches = self.hidden_size * 1.2
            scale_factor = math.sqrt(target_patches / current_patches)
            
            min_dimension = min(width, height)
            if min_dimension < self.min_large_image_size:
                size_scale = self.min_large_image_size / min_dimension
                scale_factor = max(scale_factor, size_scale)
            
            new_width = int(width * scale_factor)
            new_height = int(height * scale_factor)
            image_pil = image_pil.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        return image_pil

    def align_image_pair(self, img1_pil, img2_pil):
        w1, h1 = img1_pil.size
        w2, h2 = img2_pil.size
        
        dimension_choice = random.choice(['width', 'height'])
        
        if dimension_choice == 'width':
            target_width = max(w1, w2)
            if w1 != target_width:
                new_h1 = int(h1 * target_width / w1)
                img1_pil = img1_pil.resize((target_width, new_h1), Image.Resampling.LANCZOS)
            if w2 != target_width:
                new_h2 = int(h2 * target_width / w2)
                img2_pil = img2_pil.resize((target_width, new_h2), Image.Resampling.LANCZOS)
        else:
            target_height = max(h1, h2)
            if h1 != target_height:
                new_w1 = int(w1 * target_height / h1)
                img1_pil = img1_pil.resize((new_w1, target_height), Image.Resampling.LANCZOS)
            if h2 != target_height:
                new_w2 = int(w2 * target_height / h2)
                img2_pil = img2_pil.resize((new_w2, target_height), Image.Resampling.LANCZOS)
        
        img1_large = self.ensure_large_image_size(img1_pil)
        img2_large = self.ensure_large_image_size(img2_pil)
        
        w1_after, h1_after = img1_large.size
        w2_after, h2_after = img2_large.size
        w1_aligned, h1_aligned = img1_pil.size
        w2_aligned, h2_aligned = img2_pil.size
        
        scale1 = max(w1_after / w1_aligned, h1_after / h1_aligned) if w1_aligned > 0 and h1_aligned > 0 else 1.0
        scale2 = max(w2_after / w2_aligned, h2_after / h2_aligned) if w2_aligned > 0 and h2_aligned > 0 else 1.0
        
        if abs(scale1 - scale2) > 0.01:
            final_scale = max(scale1, scale2)
            final_w1 = int(w1_aligned * final_scale)
            final_h1 = int(h1_aligned * final_scale)
            final_w2 = int(w2_aligned * final_scale)
            final_h2 = int(h2_aligned * final_scale)
            img1_large = img1_pil.resize((final_w1, final_h1), Image.Resampling.LANCZOS)
            img2_large = img2_pil.resize((final_w2, final_h2), Image.Resampling.LANCZOS)
        
        return img1_large, img2_large

    def prepare_dual_tensors(self, img1_pil, img2_pil):
        img1_aligned, img2_aligned = self.align_image_pair(img1_pil, img2_pil)
        
        original1 = self.original_transform(img1_aligned)
        original2 = self.original_transform(img2_aligned)
        clip1 = self.clip_transform(img1_aligned)
        clip2 = self.clip_transform(img2_aligned)
        
        return clip1, clip2, original1, original2

    def calculate_pair_importance(self, original1, original2):
        patch_size = self.scaled_patchsizes[-1]
        
        try:
            img1_aligned = pad_or_crop(original1, lcm(self.scaled_patchsizes))
            img2_aligned = pad_or_crop(original2, lcm(self.scaled_patchsizes))
            
            overlap_h = min(img1_aligned.shape[1], img2_aligned.shape[1])
            img1_overlap = img1_aligned[:, :overlap_h, :]
            img2_overlap = img2_aligned[:, :overlap_h, :]
            
            img1_patches = image_to_patches(img1_overlap, patch_size, patch_size)
            img2_patches = image_to_patches(img2_overlap, patch_size, patch_size)
            
            if not img1_patches or not img2_patches:
                return {
                    'overlap_importance': [0.5],
                    'img1_unique_count': 0,
                    'img2_unique_count': 0
                }
            
            overlap_importance = []
            for p1, p2 in zip(img1_patches, img2_patches):
                diff_score = self.calculate_ssim_difference(p1, p2)
                overlap_importance.append(diff_score)
            
            img1_unique_count = 0
            img2_unique_count = 0
            
            if img1_aligned.shape[1] > overlap_h:
                img1_extra = img1_aligned[:, overlap_h:, :]
                img1_extra_patches = image_to_patches(img1_extra, patch_size, patch_size)
                img1_unique_count = len(img1_extra_patches)
            
            if img2_aligned.shape[1] > overlap_h:
                img2_extra = img2_aligned[:, overlap_h:, :]
                img2_extra_patches = image_to_patches(img2_extra, patch_size, patch_size)
                img2_unique_count = len(img2_extra_patches)
            
            return {
                'overlap_importance': overlap_importance,
                'img1_unique_count': img1_unique_count,
                'img2_unique_count': img2_unique_count
            }
        except Exception:
            return {
                'overlap_importance': [0.5] * 10,
                'img1_unique_count': 0,
                'img2_unique_count': 0
            }

    def calculate_ssim_difference(self, patch1, patch2):
        try:
            p1_np = patch1.permute(1, 2, 0).cpu().numpy()
            p2_np = patch2.permute(1, 2, 0).cpu().numpy()
            
            ssim_scores = []
            for c in range(3):
                channel_ssim = ssim(p1_np[:,:,c], p2_np[:,:,c], data_range=1.0)
                ssim_scores.append(channel_ssim)
            
            avg_ssim = np.mean(ssim_scores)
            diff_score = 1 - avg_ssim
            return max(0, min(1, diff_score))
        except Exception:
            return 0.5

    def apply_pair_importance_to_image(self, pair_importance, is_img1=True):
        importance_scores = []
        patch_types = []
        
        overlap_scores = pair_importance['overlap_importance']
        importance_scores.extend(overlap_scores)
        patch_types.extend(['overlap'] * len(overlap_scores))
        
        unique_count = pair_importance['img1_unique_count'] if is_img1 else pair_importance['img2_unique_count']
        importance_scores.extend([None] * unique_count)
        patch_types.extend(['unique'] * unique_count)
        
        return importance_scores, patch_types

    def process_similarity_pair(self, img1_pil, img2_pil, is_train=False,pair_id=None):
        try:
            if is_train and self.augmentation is not None:
                img1_pil, _ = self.augment(img1_pil, None)
                img2_pil, _ = self.augment(img2_pil, None)

            self._setup_processing_state(pair_id) 
            clip1, clip2, original1, original2 = self.prepare_dual_tensors(img1_pil, img2_pil)
            
            pair_importance = self.calculate_pair_importance(original1, original2)
            
            img1_importance, img1_types = self.apply_pair_importance_to_image(pair_importance, is_img1=True)
            self._current_importance = img1_importance
            self._current_patch_types = img1_types
            patches1, pos_embed1, mask1 = self.prepare_patches(clip1)
            
            img2_importance, img2_types = self.apply_pair_importance_to_image(pair_importance, is_img1=False)
            self._current_importance = img2_importance
            self._current_patch_types = img2_types
            patches2, pos_embed2, mask2 = self.prepare_patches(clip2)
            
            return patches1, pos_embed1, mask1, patches2, pos_embed2, mask2
            
        except Exception as e:
            print(f"Similarity pair processing failed: {e}")
            
            patches1, pos_embed1, mask1 = self.process_image(img1_pil, is_train)
            patches2, pos_embed2, mask2 = self.process_image(img2_pil, is_train)
            return patches1, pos_embed1, mask1, patches2, pos_embed2, mask2