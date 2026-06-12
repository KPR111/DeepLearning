# Building VLM from Scratch


# The main problem we were facing in Multi Modal System is 
# We trained the Image Models seperately i.e Image Embeddings
# We trained the Text Models seperately i.e Text Embeddings

# But when take a image represeting apple and a text 'apple' and when we find the text and image embedding of these two
# As these two represet same thing , then it's similarity should be high 
# But it's obviously that without trainig it explicitly ,how can it have the similarity high??

# So this is our objective function 
# Find the Image embedding and Text embedding and try to maximze the similarity between them 

# So there are 3 types of VLM's architecture in use.


# 1. Dual-Stream Contrastive Models:
# “Some will find the image context vector and text context vector and maximize similarity between them.”
# Instead of heavy fusion, this architecture uses two independent, parallel encoders (one for images, one for text). 
# It processes the inputs separately and projects them into a shared vector embedding space. The model is trained using contrastive learning (e.g., InfoNCE loss)
# to maximize the cosine similarity of true image-text pairs while minimizing it for mismatched pairs

# [ Image Input ] ---> [ Image Encoder ] ---> [ Image Vector ]  \  Maximize Cosine
#                                                                 ===> Similarity
# [ Text Input  ] ---> [ Text Encoder  ] ---> [ Text Vector ]   /  (Contrastive Loss


# Examples:
# CLIP (OpenAI): The foundational model utilizing this design.
# SigLIP (Google): An updated version replacing the softmax loss with a more efficient pairwise sigmoid loss. 



# 2. Single-Stream / Unified Embedding Models (Concatenation):
# “Some will just concat image and text embedding and run attention on it.”
# Often referred to as Early Fusion or Unified Architecture, this approach treats visual components exactly like text tokens. An image is cut into patches, processed
# through a vision backbone, projected via a linear adapter to match the LLM's dimensions, and then literally concatenated sequentially with the text token embeddings. 
# The combined sequence is fed directly into a standard transformer where Self-Attention processes both modalities simultaneously.

# [ Image patches ] -> [ Projector ] -> [ Image Tokens ] \
#                                                         ===> [ Concatenated Tokens ] ---> [ Transformer Layer (Self-Attention) ]
# [ Text string   ] ------------------> [ Text Tokens  ] /

# Primary Examples:
# LLaVA: Projects vision patches seamlessly into the Llama token space.
# Gemma 3 / Qwen2-VL: Modern natively multimodal auto-regressive models processing combined inputs.



# 3.  Cross-Attention Fusion Models:
# “Some will use cross attention between images and text.”
# Often referred to as Intermediate Fusion, this structure keeps the processing pathways mostly separate but introduces specialized Cross-Attention layers intercalated 
# between standard text transformer blocks. Instead of the text attending to itself, the text features serve as the Queries (Qcap Q𝑄), while the visual features from 
# an independent vision encoder act as the Keys (Kcap K𝐾) and Values (Vcap V𝑉). This allows the language model to dynamically extract specific visual cues precisely
# when generating certain words. 

# 					[ Image Input ] ---> [ Vision Encoder ] 
#                                                    │
#                                                    │ (Keys/Values)
# 												   ▼
# [ Text Input ] ---> [ Text Layer ] --------> [ Cross-Attention Layer ] ---> [ Next Text Layer ]
#                                (Queries)
							   

# Primary Examples:
# Flamingo (DeepMind): Pioneered "Gated Cross-Attention" to inject visual context into a frozen language model.
# BLIP-2 / InstructBLIP: Uses a querying transformer (Q-Former) to bridge modalities using cross-attention mechanisms.



# So First let's talk about the CLIP Model 
# CLIP Model First finds the context vector of images and texts individually and tries to maximize the similarity between the context vector of images and texts 
# between the correct pair using the Cosine similarity

# What they OpenAI told is that
# We consider N Images and N captions

# so they have created a NxN Matrix representing the Data , where N Images represented in row wise and N captions (text) are represented in column wise

# they calculated loss twice (HOW??)
# First time i2t -> image to text
# In i2t , they first pick one Image and pick all N captions 
# now we will try to maximize the sim(Image_1,Text_1) and minize other pairs like sim(Image_1,Text_i) where i not equal to 1
# so they are simiply maximizing the softmax(sim(Image_1,Text_1)) -> which can be represented as  exp(sim(Image_1,Text_1))/sigma(exp(Image_1,Text_i)) 
# this sigma is taken for all N texts wrt to Image one only
# Like that we have N Images 
# so this is i2t Loss 

# In similar way we have t2i loss function where 
# i.e we pick one text and consider N images where one correct pair (Text_1 , Image_1) maximize it and minimize other pairs with Text_1
# in similar way we are having N Texts
# this is t2i Loss

# so L clip = (L(i2t) + L(t2i)) /2
# This is called bi direcitonal cross entropy losses

# If you understand it properly our CLIP follows VLM Type1
# so this is want we will try to Implement now.

# CLIP - Contrastive Language-Image Pretraining











import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# =====================================================================
# 1. VISION ENCODER COMPONENTS (Vision Transformer from Scratch)
# =====================================================================

class PatchEmbedding(nn.Module):
    """Splits an image into patches and projects them into an embedding space."""
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768):
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        
        # Convolutions easily extract and linearly project patches in one step
        self.proj = nn.Conv2d(
            in_channels, embed_dim, 
            kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x):
        # Input: (Batch, Channels, Height, Width)
        x = self.proj(x)  # (Batch, Embed_Dim, H_patches, W_patches)
        x = x.flatten(2)  # (Batch, Embed_Dim, Num_Patches)
        x = x.transpose(1, 2)  # (Batch, Num_Patches, Embed_Dim)
        return x


class TransformerEncoderLayer(nn.Module):
    """Standard Transformer Encoder block with Multi-Head Attention and MLP."""
    def __init__(self, embed_dim, num_heads, mlp_dim, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.mha = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.ln2 = nn.LayerNorm(embed_dim)
        
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        # Multi-Head Attention with Residual Connection
        x_ln = self.ln1(x)
        attn_out, _ = self.mha(x_ln, x_ln, x_ln)
        x = x + attn_out
        
        # MLP with Residual Connection
        x = x + self.mlp(self.ln2(x))
        return x


class VisionTransformer(nn.Module):
    """Full ViT backbone parsing images to class tokens."""
    def __init__(self, img_size=224, patch_size=16, in_channels=3, num_layers=6, 
                 embed_dim=768, num_heads=12, mlp_dim=2048, dropout=0.1):
        super().__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        num_patches = self.patch_embed.num_patches
        
        # Learnable Class [CLS] token and Position Embeddings
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.dropout = nn.Dropout(dropout)
        
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(embed_dim, num_heads, mlp_dim, dropout)
            for _ in range(num_layers)
        ])
        self.ln = nn.LayerNorm(embed_dim)

    def forward(self, x):
        b = x.shape[0]
        x = self.patch_embed(x)
        
        # Expand CLS token to match batch size and append to patches
        cls_tokens = self.cls_token.expand(b, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        
        x = x + self.pos_embed
        x = self.dropout(x)
        
        for layer in self.layers:
            x = layer(x)
            
        x = self.ln(x)
        return x[:, 0]  # Return only the [CLS] token representation


# =====================================================================
# 2. TEXT ENCODER COMPONENTS (Text Transformer from Scratch)
# =====================================================================

class TextTransformer(nn.Module):
    """Text encoder based on a standard non-causal Transformer Encoder."""
    def __init__(self, vocab_size=49408, max_seq_len=77, num_layers=6, 
                 embed_dim=512, num_heads=8, mlp_dim=2048, dropout=0.1):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_embedding = nn.Parameter(torch.zeros(1, max_seq_len, embed_dim))
        self.dropout = nn.Dropout(dropout)
        
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(embed_dim, num_heads, mlp_dim, dropout)
            for _ in range(num_layers)
        ])
        self.ln = nn.LayerNorm(embed_dim)

    def forward(self, text, eos_indices):
        # text shape: (Batch, Seq_Len)
        x = self.token_embedding(text) + self.pos_embedding[:, :text.size(1), :]
        x = self.dropout(x)
        
        for layer in self.layers:
            x = layer(x)
            
        x = self.ln(x)
        
        # Extract features corresponding specifically to the [EOS] token index for each batch
        b = x.shape[0]
        eos_features = x[torch.arange(b), eos_indices]
        return eos_features


# =====================================================================
# 3. CORE CLIP MODULE
# =====================================================================

class CLIP(nn.Module):
    """The master CLIP framework binding text and vision with projection maps."""
    def __init__(self, embed_dim=512, 
                 vocab_size=49408, max_seq_len=77, text_layers=6, text_heads=8, text_mlp=2048,
                 img_size=224, patch_size=16, vision_layers=6, vision_dim=768, vision_heads=12, vision_mlp=2048):
        super().__init__()
        
        # Encoders
        self.vision_encoder = VisionTransformer(img_size, patch_size, 3, vision_layers, vision_dim, vision_heads, vision_mlp)
        self.text_encoder = TextTransformer(vocab_size, max_seq_len, text_layers, embed_dim, text_heads, text_mlp)
        
        # Projection Heads to map to shared multi-modal embed_dim
        self.visual_projection = nn.Linear(vision_dim, embed_dim, bias=False)
        self.text_projection = nn.Linear(embed_dim, embed_dim, bias=False)
        
        # Learnable Temperature Scale (initialized to CLIP's baseline 0.07 value)
        self.logit_scale = nn.Parameter(torch.ones([]) * math.log(1 / 0.07))

    def forward(self, images, text, eos_indices):
        # 1. Compute features
        image_features = self.vision_encoder(images)
        text_features = self.text_encoder(text, eos_indices)
        
        # 2. Project into shared embedding space
        image_embeddings = self.visual_projection(image_features)
        text_embeddings = self.text_projection(text_features)
        
        # 3. Normalize embeddings to unit L2 norms
        image_embeddings = F.normalize(image_embeddings, p=2, dim=-1)
        text_embeddings = F.normalize(text_embeddings, p=2, dim=-1)
        
        return image_embeddings, text_embeddings, self.logit_scale.exp()


# =====================================================================
# 4. SYMMETRIC LOSS FUNCTION
# =====================================================================

def contrastive_loss(image_embeddings, text_embeddings, temperature):
    """Calculates bidirectional symmetric cross-entropy loss over matched pairs."""
    batch_size = image_embeddings.shape[0]
    
    # Similarity matrix (Batch, Batch)
    logits = torch.matmul(image_embeddings, text_embeddings.T) * temperature
    
    # Ground truth: targets are on the main diagonal
    targets = torch.arange(batch_size, device=image_embeddings.device)
    
    # Bidirectional cross entropy losses
    loss_images = F.cross_entropy(logits, targets)
    loss_texts = F.cross_entropy(logits.T, targets)
    
    return (loss_images + loss_texts) / 2


# =====================================================================
# 5. EXECUTION & SANITY CHECK PIPELINE
# =====================================================================

if __name__ == "__main__":
    # Setup Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running CLIP implementation on device: {device}")
    
    # Hyperparameters & Dimensions
    batch_size = 4
    seq_length = 77
    vocab_size = 49408
    shared_dim = 512
    
    # Initialize the CLIP Model
    clip_model = CLIP(
        embed_dim=shared_dim, 
        vocab_size=vocab_size, 
        max_seq_len=seq_length,
        vision_dim=768  # Mapping ViT base outputs
    ).to(device)
    
    # Mock Data Pipeline Setup
    # Create random images: (Batch, Channels, Height, Width)
    mock_images = torch.randn(batch_size, 3, 224, 224).to(device)
    
    # Create random token IDs for captions
    mock_text = torch.randint(10, vocab_size - 1, (batch_size, seq_length)).to(device)
    
    # Force [SOS] and [EOS] tokens for realistic setup
    mock_text[:, 0] = 1  # Standard start token index mock
    eos_indices = torch.randint(10, seq_length - 1, (batch_size,)).to(device)
    for i, eos_idx in enumerate(eos_indices):
        mock_text[i, eos_idx] = 2  # Standard end token index mock
        mock_text[i, eos_idx + 1:] = 0  # Pad remaining indices with 0s
        
    print("\n--- Pipeline Inputs Asserted ---")
    print(f"Images tensor shape: {mock_images.shape}")
    print(f"Text tensor shape:   {mock_text.shape}")
    
    # Forward Pass through Model
    img_embeds, txt_embeds, current_temp = clip_model(mock_images, mock_text, eos_indices)
    
    print("\n--- Forward Pass Output Matrix Dimensions ---")
    print(f"Projected Image Embeddings: {img_embeds.shape}")
    print(f"Projected Text Embeddings:  {txt_embeds.shape}")
    print(f"Learned Scaled Temperature: {current_temp.item():.4f}")
    
    # Loss Calculation Step
    loss = contrastive_loss(img_embeds, txt_embeds, current_temp)
    print(f"\nCalculated Contrastive Loss: {loss.item():.4f}")
    
    # Backward Pass Verification
    loss.backward()
    print("Backward pass completed successfully! Gradients initialized.")
