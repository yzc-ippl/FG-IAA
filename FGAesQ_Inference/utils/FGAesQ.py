import torch
import torch.nn as nn
import clip
import numpy as np
import torch.nn.functional as F

class FGAesQ(nn.Module):
    def __init__(self, 
                 scales=2,
                 max_seq_len=512,
                 factor=0.5,
                 initial_hidden_size=112,
                 pretrained_path=None):
        super().__init__()
        
        print("Loading CLIP model...")
        self.clip_model, _ = clip.load("ViT-B/16", device='cpu')
        self.clip_model = self.clip_model.float()
        
        self.scales = scales
        self.max_seq_len = max_seq_len
        self.factor = factor
        self.initial_hidden_size = initial_hidden_size
        
        for param in self.clip_model.parameters():
            param.requires_grad = False
        
        for param in self.clip_model.visual.parameters():
            param.requires_grad = True
        
        self.ln_pre = self.clip_model.visual.ln_pre
        self.transformer = self.clip_model.visual.transformer
        self.ln_post = self.clip_model.visual.ln_post
        self.proj = self.clip_model.visual.proj
        
        self.feature_dim = 512
        
        self.scale_embedding = nn.Parameter(torch.randn(1, scales, 768) * 0.02)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, 768))
        
        self.iqa_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 10)
        )
        
        if pretrained_path and pretrained_path.endswith('.pt'):
            try:
                state_dict = torch.load(pretrained_path, map_location='cpu')
                self.load_state_dict(state_dict, strict=False)
                print(f"Loaded pretrained model: {pretrained_path}")
            except Exception as e:
                print(f"Failed to load pretrained model: {e}")
    
    def forward(self, patches, pos_embeds, masks, flag=False):
        tokens = self.patches_to_tokens(patches, masks)
        features = self.extract_features(tokens, pos_embeds, masks)
        dist_logits = self.iqa_head(features)
        
        score_weights = torch.arange(1, 11, dtype=torch.float32, device=dist_logits.device)
        dist_probs = F.softmax(dist_logits, dim=1)
        scores = torch.sum(dist_probs * score_weights.unsqueeze(0), dim=1, keepdim=True)
        
        if flag:
            return features, (dist_logits, scores)
        else:
            return scores
    
    def patches_to_tokens(self, patches, masks):
        batch_size, seq_len, c, h, w = patches.shape
        device = patches.device
        
        if self.ln_pre.weight.device != device:
            self.clip_model = self.clip_model.to(device)
            self.scale_embedding = self.scale_embedding.to(device)
            self.mask_token = self.mask_token.to(device)
        
        actual_patches = patches[:, 1:, :, :, :]
        actual_seq_len = actual_patches.shape[1]
        
        reshaped_patches = actual_patches.reshape(-1, c, h, w)
        embeddings = self.clip_model.visual.conv1(reshaped_patches)
        embeddings = embeddings.flatten(2).transpose(1, 2).squeeze(1)
        embeddings = embeddings.reshape(batch_size, actual_seq_len, 768)
        
        if masks is not None:
            padding_mask = (masks == 9).int()
            masks[masks == 9] = 0
            seq_length = embeddings.shape[1]
            mask_tokens = self.mask_token.expand(batch_size, seq_length, -1)
            mask = padding_mask[:, 1:].unsqueeze(-1).type_as(mask_tokens)
            embeddings = embeddings * (1.0 - mask) + mask_tokens * mask
        
        embeddings = self.add_scale_embed(embeddings, masks)
        
        cls_tokens = self.clip_model.visual.class_embedding.expand(batch_size, 1, -1)
        tokens = torch.cat((cls_tokens, embeddings), dim=1)
        
        return tokens
    
    def add_scale_embed(self, embeddings, masks):
        np_se = self.scale_embedding.detach().cpu().numpy()[0]
        mask = masks[:, 1:].cpu()
        scale_embed = torch.tensor(np.take(np_se, mask, axis=0)).to(embeddings.device)
        embeddings = embeddings + scale_embed
        return embeddings
    
    def extract_features(self, tokens, pos_embeds, masks):
        x = tokens + pos_embeds
        x = self.ln_pre(x)
        x = x.permute(1, 0, 2)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)
        
        cls_tokens = x[:, 0, :]
        cls_tokens = self.ln_post(cls_tokens)
        features = cls_tokens @ self.proj
        
        return features