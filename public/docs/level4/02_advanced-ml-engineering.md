# 高度な機械学習エンジニアリング

![機械学習エンジニアリング](/images/illustrations/level4-ml-engineering.jpg)

## 学習目標
- 最先端の深層学習アーキテクチャを実装する
- 大規模言語モデル（LLM）のファインチューニングと最適化を行う
- 強化学習システムを設計・実装する
- AutoMLとニューラルアーキテクチャサーチを活用する

## 想定学習時間
約12-14時間（実装演習・実験含む）

---

## 1. 最先端深層学習アーキテクチャ

### Transformer アーキテクチャの実装

#### 1. マルチヘッドアテンション機構
```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        
        # 1. Linear projections in batch from d_model => h x d_k
        Q = self.W_q(query).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        
        # 2. Scaled Dot-Product Attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        context = torch.matmul(attention_weights, V)
        
        # 3. Concatenate heads and put through final linear layer
        context = context.transpose(1, 2).contiguous().view(
            batch_size, -1, self.n_heads * self.d_k
        )
        
        output = self.W_o(context)
        
        # Residual connection and layer normalization
        output = self.layer_norm(output + query)
        
        return output, attention_weights

class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super(TransformerBlock, self).__init__()
        self.attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        # Multi-head attention
        attn_output, _ = self.attention(x, x, x, mask)
        
        # Feed forward
        ff_output = self.feed_forward(attn_output)
        ff_output = self.dropout(ff_output)
        output = self.layer_norm2(ff_output + attn_output)
        
        return output

class TransformerModel(nn.Module):
    def __init__(self, vocab_size, d_model=512, n_heads=8, n_layers=6, 
                 d_ff=2048, max_seq_length=512, dropout=0.1):
        super(TransformerModel, self).__init__()
        
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_length)
        
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        
        self.ln_f = nn.LayerNorm(d_model)
        self.fc_out = nn.Linear(d_model, vocab_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        seq_length = x.size(1)
        
        # Token embeddings and positional encoding
        token_embeddings = self.embedding(x)
        token_embeddings = token_embeddings * math.sqrt(self.d_model)
        token_embeddings = self.positional_encoding(token_embeddings)
        embeddings = self.dropout(token_embeddings)
        
        # Transformer blocks
        for transformer in self.transformer_blocks:
            embeddings = transformer(embeddings, mask)
        
        # Final layer norm and output projection
        embeddings = self.ln_f(embeddings)
        logits = self.fc_out(embeddings)
        
        return logits

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length=512):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                            -(math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))
        
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]
```

#### 2. Vision Transformer (ViT) 実装
```python
class VisionTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, num_classes=1000, 
                 d_model=768, n_heads=12, n_layers=12, d_ff=3072, dropout=0.1):
        super(VisionTransformer, self).__init__()
        
        assert img_size % patch_size == 0
        
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        self.d_model = d_model
        
        # Patch embedding
        self.patch_embedding = nn.Conv2d(
            in_channels=3,
            out_channels=d_model,
            kernel_size=patch_size,
            stride=patch_size
        )
        
        # Positional embedding
        self.positional_embedding = nn.Parameter(
            torch.randn(1, self.n_patches + 1, d_model)
        )
        
        # CLS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        
        # Transformer encoder
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        
        self.ln = nn.LayerNorm(d_model)
        self.fc = nn.Linear(d_model, num_classes)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        batch_size = x.size(0)
        
        # Create patch embeddings
        x = self.patch_embedding(x)  # (B, d_model, H', W')
        x = x.flatten(2).transpose(1, 2)  # (B, n_patches, d_model)
        
        # Add CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        
        # Add positional embedding
        x = x + self.positional_embedding
        x = self.dropout(x)
        
        # Transformer blocks
        for transformer in self.transformer_blocks:
            x = transformer(x)
        
        # Classification head
        x = self.ln(x)
        cls_output = x[:, 0]  # CLS token output
        logits = self.fc(cls_output)
        
        return logits

# 学習用の追加機能
class VisionTransformerWithAugmentation(VisionTransformer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mixup_alpha = 1.0
        self.cutmix_alpha = 1.0
        
    def mixup_data(self, x, y, alpha=1.0):
        """Mixup augmentation"""
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1
            
        batch_size = x.size(0)
        index = torch.randperm(batch_size).to(x.device)
        
        mixed_x = lam * x + (1 - lam) * x[index]
        y_a, y_b = y, y[index]
        
        return mixed_x, y_a, y_b, lam
    
    def cutmix_data(self, x, y, alpha=1.0):
        """CutMix augmentation"""
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1
            
        batch_size = x.size(0)
        index = torch.randperm(batch_size).to(x.device)
        
        # Generate random bounding box
        W = x.size(2)
        H = x.size(3)
        cut_rat = np.sqrt(1. - lam)
        cut_w = np.int(W * cut_rat)
        cut_h = np.int(H * cut_rat)
        
        cx = np.random.randint(W)
        cy = np.random.randint(H)
        
        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)
        
        # Apply CutMix
        x[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]
        
        # Adjust lambda based on actual box area
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (W * H))
        
        y_a, y_b = y, y[index]
        
        return x, y_a, y_b, lam
```

### 先進的な最適化手法

#### 1. Sharpness-Aware Minimization (SAM)
```python
class SAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho=0.05, adaptive=False, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"
        
        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super(SAM, self).__init__(params, defaults)
        
        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)
        
    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)
            
            for p in group["params"]:
                if p.grad is None:
                    continue
                self.state[p]["old_p"] = p.data.clone()
                e_w = (torch.pow(p, 2) if group["adaptive"] else 1.0) * p.grad * scale.to(p)
                p.add_(e_w)  # climb to the local maximum "w + e(w)"
                
        if zero_grad:
            self.zero_grad()
    
    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                p.data = self.state[p]["old_p"]  # get back to "w" from "w + e(w)"
                
        self.base_optimizer.step()  # do the actual "sharpness-aware" update
        
        if zero_grad:
            self.zero_grad()
    
    @torch.no_grad()
    def step(self, closure=None):
        assert closure is not None, "Sharpness Aware Minimization requires closure"
        closure = torch.enable_grad()(closure)  # the closure should do a full forward-backward pass
        
        self.first_step(zero_grad=True)
        closure()
        self.second_step()
        
    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device  # assume all on same device
        norm = torch.norm(
            torch.stack([
                ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad).norm(p=2).to(shared_device)
                for group in self.param_groups for p in group["params"]
                if p.grad is not None
            ]),
            p=2
        )
        return norm
    
    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups
```

---

## 2. 大規模言語モデル（LLM）の最適化

### LoRA（Low-Rank Adaptation）実装

#### 1. LoRA レイヤーの実装
```python
class LoRALayer(nn.Module):
    def __init__(self, in_features, out_features, rank=4, alpha=1):
        super(LoRALayer, self).__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        
        # Frozen pre-trained weight
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.weight.requires_grad = False
        
        # LoRA weights
        self.lora_A = nn.Parameter(torch.randn(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        
        # Initialize LoRA weights
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
        
    def forward(self, x):
        # Original forward pass
        result = F.linear(x, self.weight)
        
        # LoRA forward pass
        lora_output = F.linear(F.linear(x, self.lora_A), self.lora_B) * self.scaling
        
        return result + lora_output
    
    def merge_weights(self):
        """Merge LoRA weights into the original weights for inference"""
        self.weight.data += (self.lora_B @ self.lora_A) * self.scaling
        self.merged = True

class LoRATransformer(nn.Module):
    def __init__(self, base_model, lora_rank=8, lora_alpha=16, target_modules=None):
        super(LoRATransformer, self).__init__()
        self.base_model = base_model
        
        # Default target modules for LoRA
        if target_modules is None:
            target_modules = ['q_proj', 'v_proj', 'k_proj', 'o_proj']
        
        # Replace linear layers with LoRA layers
        self._replace_with_lora(target_modules, lora_rank, lora_alpha)
        
        # Freeze base model
        for param in base_model.parameters():
            param.requires_grad = False
            
    def _replace_with_lora(self, target_modules, rank, alpha):
        for name, module in self.base_model.named_modules():
            if any(target in name for target in target_modules):
                if isinstance(module, nn.Linear):
                    # Get parent module
                    parent_name = '.'.join(name.split('.')[:-1])
                    child_name = name.split('.')[-1]
                    parent_module = self.base_model
                    
                    if parent_name:
                        for part in parent_name.split('.'):
                            parent_module = getattr(parent_module, part)
                    
                    # Replace with LoRA layer
                    lora_layer = LoRALayer(
                        module.in_features,
                        module.out_features,
                        rank=rank,
                        alpha=alpha
                    )
                    
                    # Copy original weights
                    lora_layer.weight.data = module.weight.data.clone()
                    
                    # Set the new module
                    setattr(parent_module, child_name, lora_layer)
    
    def forward(self, *args, **kwargs):
        return self.base_model(*args, **kwargs)
    
    def save_lora_weights(self, path):
        """Save only LoRA weights"""
        lora_state_dict = {}
        for name, param in self.named_parameters():
            if param.requires_grad:  # Only LoRA parameters
                lora_state_dict[name] = param.data
        torch.save(lora_state_dict, path)
    
    def load_lora_weights(self, path):
        """Load LoRA weights"""
        lora_state_dict = torch.load(path)
        self.load_state_dict(lora_state_dict, strict=False)
```

#### 2. 量子化とメモリ最適化
```python
import bitsandbytes as bnb
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch.nn as nn

class QuantizedLLM:
    def __init__(self, model_name, load_in_8bit=True, device_map="auto"):
        self.model_name = model_name
        self.device_map = device_map
        
        # 8-bit quantization configuration
        self.quantization_config = {
            "load_in_8bit": load_in_8bit,
            "device_map": device_map,
            "torch_dtype": torch.float16,
            "low_cpu_mem_usage": True,
        }
        
    def load_model(self):
        """Load quantized model"""
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            **self.quantization_config
        )
        
        # Prepare for k-bit training
        model = self.prepare_model_for_kbit_training(model)
        
        return model
    
    def prepare_model_for_kbit_training(self, model):
        """Prepare the model for k-bit training"""
        model.gradient_checkpointing_enable()
        
        # Cast non-int8 layers to fp32
        for param in model.parameters():
            param.requires_grad = False
            if param.ndim == 1:
                param.data = param.data.to(torch.float32)
        
        # Enable input require grads
        model.enable_input_require_grads()
        
        class CastOutputToFloat(nn.Sequential):
            def forward(self, x):
                return super().forward(x).to(torch.float32)
        
        model.lm_head = CastOutputToFloat(model.lm_head)
        
        return model
    
    def add_lora_layers(self, model, lora_config):
        """Add LoRA layers to the quantized model"""
        from peft import LoraConfig, get_peft_model, TaskType
        
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=lora_config.get("r", 8),
            lora_alpha=lora_config.get("alpha", 32),
            lora_dropout=lora_config.get("dropout", 0.1),
            target_modules=lora_config.get("target_modules", ["q_proj", "v_proj"])
        )
        
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
        
        return model

# メモリ効率的なトレーニング
class EfficientTrainer:
    def __init__(self, model, optimizer_type="adamw_8bit"):
        self.model = model
        self.optimizer_type = optimizer_type
        
    def create_optimizer(self, learning_rate=1e-4):
        """Create memory-efficient optimizer"""
        if self.optimizer_type == "adamw_8bit":
            optimizer = bnb.optim.AdamW8bit(
                self.model.parameters(),
                lr=learning_rate,
                betas=(0.9, 0.999),
                eps=1e-8,
                weight_decay=0.01
            )
        elif self.optimizer_type == "lion_8bit":
            optimizer = bnb.optim.Lion8bit(
                self.model.parameters(),
                lr=learning_rate,
                betas=(0.9, 0.99),
                weight_decay=0.01
            )
        else:
            optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=learning_rate
            )
        
        return optimizer
    
    def gradient_accumulation_step(self, batch, accumulation_steps=4):
        """Gradient accumulation for large batch training"""
        loss_total = 0
        
        for i in range(accumulation_steps):
            # Get mini-batch
            mini_batch = {
                k: v[i::accumulation_steps] for k, v in batch.items()
            }
            
            # Forward pass
            outputs = self.model(**mini_batch)
            loss = outputs.loss / accumulation_steps
            
            # Backward pass
            loss.backward()
            loss_total += loss.item()
        
        return loss_total
```

---

## 3. 強化学習システム実装

### Deep Q-Network (DQN) と改良版

#### 1. Rainbow DQN 実装
```python
import numpy as np
from collections import deque, namedtuple
import random

class PrioritizedReplayBuffer:
    """優先度付き経験再生バッファ"""
    def __init__(self, capacity, alpha=0.6):
        self.capacity = capacity
        self.alpha = alpha
        self.buffer = []
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.pos = 0
        
    def push(self, state, action, reward, next_state, done):
        max_prio = self.priorities.max() if self.buffer else 1.0
        
        if len(self.buffer) < self.capacity:
            self.buffer.append((state, action, reward, next_state, done))
        else:
            self.buffer[self.pos] = (state, action, reward, next_state, done)
        
        self.priorities[self.pos] = max_prio
        self.pos = (self.pos + 1) % self.capacity
        
    def sample(self, batch_size, beta=0.4):
        if len(self.buffer) == self.capacity:
            prios = self.priorities
        else:
            prios = self.priorities[:self.pos]
        
        probs = prios ** self.alpha
        probs /= probs.sum()
        
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]
        
        # Importance sampling weights
        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-beta)
        weights /= weights.max()
        weights = np.array(weights, dtype=np.float32)
        
        return samples, indices, weights
    
    def update_priorities(self, batch_indices, batch_priorities):
        for idx, prio in zip(batch_indices, batch_priorities):
            self.priorities[idx] = prio

class NoisyLinear(nn.Module):
    """Noisy Networks for Exploration"""
    def __init__(self, in_features, out_features, std_init=0.5):
        super(NoisyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init
        
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.register_buffer('weight_epsilon', torch.empty(out_features, in_features))
        
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        self.register_buffer('bias_epsilon', torch.empty(out_features))
        
        self.reset_parameters()
        self.reset_noise()
        
    def reset_parameters(self):
        mu_range = 1 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.out_features))
        
    def _scale_noise(self, size):
        x = torch.randn(size, device=self.weight_mu.device)
        return x.sign().mul_(x.abs().sqrt_())
        
    def reset_noise(self):
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)
        
    def forward(self, input):
        if self.training:
            return F.linear(input, 
                          self.weight_mu + self.weight_sigma * self.weight_epsilon,
                          self.bias_mu + self.bias_sigma * self.bias_epsilon)
        else:
            return F.linear(input, self.weight_mu, self.bias_mu)

class RainbowDQN(nn.Module):
    """Rainbow: Combining Improvements in Deep Reinforcement Learning"""
    def __init__(self, state_size, action_size, num_atoms=51, v_min=-10, v_max=10):
        super(RainbowDQN, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.num_atoms = num_atoms
        self.v_min = v_min
        self.v_max = v_max
        
        self.register_buffer('support', torch.linspace(v_min, v_max, num_atoms))
        self.delta_z = (v_max - v_min) / (num_atoms - 1)
        
        # Dueling network architecture
        self.feature = nn.Sequential(
            nn.Linear(state_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU()
        )
        
        # Value stream
        self.value_stream = nn.Sequential(
            NoisyLinear(128, 128),
            nn.ReLU(),
            NoisyLinear(128, num_atoms)
        )
        
        # Advantage stream
        self.advantage_stream = nn.Sequential(
            NoisyLinear(128, 128),
            nn.ReLU(),
            NoisyLinear(128, action_size * num_atoms)
        )
        
    def forward(self, state):
        features = self.feature(state)
        value = self.value_stream(features).view(-1, 1, self.num_atoms)
        advantage = self.advantage_stream(features).view(-1, self.action_size, self.num_atoms)
        
        # Combine value and advantage (Dueling DQN)
        q_atoms = value + advantage - advantage.mean(dim=1, keepdim=True)
        
        # Apply softmax to get probabilities
        q_dist = F.softmax(q_atoms, dim=-1)
        q_values = (q_dist * self.support).sum(dim=-1)
        
        return q_values, q_dist
    
    def reset_noise(self):
        """Reset noise for NoisyNet layers"""
        for module in self.modules():
            if isinstance(module, NoisyLinear):
                module.reset_noise()
```

### Proximal Policy Optimization (PPO)

#### 1. PPO実装
```python
class PPOAgent:
    def __init__(self, state_dim, action_dim, lr=3e-4, gamma=0.99, 
                 eps_clip=0.2, k_epochs=4, device='cuda'):
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.k_epochs = k_epochs
        self.device = device
        
        self.policy = ActorCritic(state_dim, action_dim).to(device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        self.policy_old = ActorCritic(state_dim, action_dim).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        self.MseLoss = nn.MSELoss()
        
    def select_action(self, state, memory):
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        with torch.no_grad():
            action, action_logprob = self.policy_old.act(state)
        
        memory.states.append(state)
        memory.actions.append(action)
        memory.logprobs.append(action_logprob)
        
        return action.cpu().numpy().flatten()
    
    def update(self, memory):
        # Monte Carlo estimate of rewards
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(memory.rewards), 
                                       reversed(memory.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
        
        # Normalizing rewards
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)
        
        # Convert list to tensor
        old_states = torch.stack(memory.states).to(self.device).detach()
        old_actions = torch.stack(memory.actions).to(self.device).detach()
        old_logprobs = torch.stack(memory.logprobs).to(self.device).detach()
        
        # Optimize policy for K epochs
        for _ in range(self.k_epochs):
            # Evaluating old actions and values
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)
            
            # Finding the ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(logprobs - old_logprobs.detach())
            
            # Finding Surrogate Loss
            advantages = rewards - state_values.detach()
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages
            
            # Final loss of PPO
            loss = -torch.min(surr1, surr2) + 0.5*self.MseLoss(state_values, rewards) - 0.01*dist_entropy
            
            # Take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
            self.optimizer.step()
        
        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super(ActorCritic, self).__init__()
        
        # Actor
        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1)
        )
        
        # Critic
        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        
    def act(self, state):
        action_probs = self.actor(state)
        dist = torch.distributions.Categorical(action_probs)
        action = dist.sample()
        action_logprob = dist.log_prob(action)
        
        return action.detach(), action_logprob.detach()
    
    def evaluate(self, state, action):
        action_probs = self.actor(state)
        dist = torch.distributions.Categorical(action_probs)
        
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = self.critic(state)
        
        return action_logprobs, torch.squeeze(state_values), dist_entropy
```

---

## 4. AutoMLとニューラルアーキテクチャサーチ

### Neural Architecture Search (NAS)

#### 1. DARTS実装
```python
class DARTSCell(nn.Module):
    """Differentiable Architecture Search Cell"""
    def __init__(self, steps, C_prev_prev, C_prev, C, reduction):
        super(DARTSCell, self).__init__()
        self.reduction = reduction
        
        if reduction:
            self.preprocess0 = FactorizedReduce(C_prev_prev, C)
            self.preprocess1 = FactorizedReduce(C_prev, C)
        else:
            self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0)
            self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0)
        
        self._steps = steps
        self._ops = nn.ModuleList()
        
        for i in range(self._steps):
            for j in range(2+i):
                stride = 2 if reduction and j < 2 else 1
                op = MixedOp(C, stride)
                self._ops.append(op)
    
    def forward(self, s0, s1, weights):
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)
        
        states = [s0, s1]
        offset = 0
        
        for i in range(self._steps):
            s = sum(self._ops[offset+j](h, weights[offset+j]) 
                   for j, h in enumerate(states))
            offset += len(states)
            states.append(s)
        
        return torch.cat(states[-self._steps:], dim=1)

class MixedOp(nn.Module):
    """Mixed operation for architecture search"""
    def __init__(self, C, stride):
        super(MixedOp, self).__init__()
        self._ops = nn.ModuleList()
        
        # Candidate operations
        PRIMITIVES = [
            'none',
            'max_pool_3x3',
            'avg_pool_3x3',
            'skip_connect',
            'sep_conv_3x3',
            'sep_conv_5x5',
            'dil_conv_3x3',
            'dil_conv_5x5'
        ]
        
        for primitive in PRIMITIVES:
            op = OPS[primitive](C, stride, False)
            if 'pool' in primitive:
                op = nn.Sequential(op, nn.BatchNorm2d(C, affine=False))
            self._ops.append(op)
    
    def forward(self, x, weights):
        return sum(w * op(x) for w, op in zip(weights, self._ops))

class DARTSNetwork(nn.Module):
    """DARTS Network for Architecture Search"""
    def __init__(self, C, num_classes, layers, steps=4):
        super(DARTSNetwork, self).__init__()
        self._C = C
        self._num_classes = num_classes
        self._layers = layers
        self._steps = steps
        
        C_curr = C
        self.stem = nn.Sequential(
            nn.Conv2d(3, C_curr, 3, padding=1, bias=False),
            nn.BatchNorm2d(C_curr)
        )
        
        C_prev_prev, C_prev, C_curr = C_curr, C_curr, C
        self.cells = nn.ModuleList()
        
        for i in range(layers):
            if i in [layers//3, 2*layers//3]:
                C_curr *= 2
                reduction = True
            else:
                reduction = False
            
            cell = DARTSCell(steps, C_prev_prev, C_prev, C_curr, reduction)
            self.cells.append(cell)
            C_prev_prev, C_prev = C_prev, C_curr
        
        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(C_prev, num_classes)
        
        self._initialize_alphas()
    
    def _initialize_alphas(self):
        k = sum(1 for i in range(self._steps) for n in range(2+i))
        num_ops = len(PRIMITIVES)
        
        self.alphas_normal = nn.Parameter(1e-3*torch.randn(k, num_ops))
        self.alphas_reduce = nn.Parameter(1e-3*torch.randn(k, num_ops))
        
        self._arch_parameters = [
            self.alphas_normal,
            self.alphas_reduce,
        ]
    
    def forward(self, input):
        s0 = s1 = self.stem(input)
        
        for i, cell in enumerate(self.cells):
            if cell.reduction:
                weights = F.softmax(self.alphas_reduce, dim=-1)
            else:
                weights = F.softmax(self.alphas_normal, dim=-1)
            
            s0, s1 = s1, cell(s0, s1, weights)
        
        out = self.global_pooling(s1)
        logits = self.classifier(out.view(out.size(0), -1))
        
        return logits
    
    def arch_parameters(self):
        return self._arch_parameters
    
    def genotype(self):
        """Extract discrete architecture from continuous weights"""
        def _parse(weights):
            gene = []
            n = 2
            start = 0
            
            for i in range(self._steps):
                end = start + n
                W = weights[start:end].copy()
                edges = sorted(range(i + 2), 
                             key=lambda x: -max(W[x][k] for k in range(len(W[x])) 
                                              if k != PRIMITIVES.index('none')))[:2]
                
                for j in edges:
                    k_best = None
                    for k in range(len(W[j])):
                        if k != PRIMITIVES.index('none'):
                            if k_best is None or W[j][k] > W[j][k_best]:
                                k_best = k
                    gene.append((PRIMITIVES[k_best], j))
                
                start = end
                n += 1
            
            return gene
        
        gene_normal = _parse(F.softmax(self.alphas_normal, dim=-1).data.cpu().numpy())
        gene_reduce = _parse(F.softmax(self.alphas_reduce, dim=-1).data.cpu().numpy())
        
        genotype = Genotype(
            normal=gene_normal,
            normal_concat=range(2+self._steps-4, self._steps+2),
            reduce=gene_reduce,
            reduce_concat=range(2+self._steps-4, self._steps+2)
        )
        
        return genotype
```

### Hyperparameter Optimization

#### 1. Bayesian Optimization
```python
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner

class HyperparameterOptimizer:
    def __init__(self, model_class, train_loader, val_loader, device='cuda'):
        self.model_class = model_class
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        
    def objective(self, trial):
        """Optuna objective function"""
        # Hyperparameter suggestions
        config = {
            'learning_rate': trial.suggest_loguniform('learning_rate', 1e-5, 1e-1),
            'batch_size': trial.suggest_categorical('batch_size', [16, 32, 64, 128]),
            'optimizer': trial.suggest_categorical('optimizer', ['adam', 'sgd', 'adamw']),
            'weight_decay': trial.suggest_loguniform('weight_decay', 1e-6, 1e-2),
            'dropout': trial.suggest_uniform('dropout', 0.1, 0.5),
            'hidden_dim': trial.suggest_int('hidden_dim', 64, 512, step=64),
            'num_layers': trial.suggest_int('num_layers', 2, 8),
        }
        
        # Create model with suggested hyperparameters
        model = self.model_class(**config).to(self.device)
        
        # Create optimizer
        if config['optimizer'] == 'adam':
            optimizer = torch.optim.Adam(
                model.parameters(), 
                lr=config['learning_rate'],
                weight_decay=config['weight_decay']
            )
        elif config['optimizer'] == 'sgd':
            optimizer = torch.optim.SGD(
                model.parameters(),
                lr=config['learning_rate'],
                momentum=0.9,
                weight_decay=config['weight_decay']
            )
        else:
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=config['learning_rate'],
                weight_decay=config['weight_decay']
            )
        
        # Training loop with early stopping
        best_val_loss = float('inf')
        patience = 10
        patience_counter = 0
        
        for epoch in range(100):
            # Training
            model.train()
            train_loss = 0
            for batch_idx, (data, target) in enumerate(self.train_loader):
                data, target = data.to(self.device), target.to(self.device)
                
                optimizer.zero_grad()
                output = model(data)
                loss = F.cross_entropy(output, target)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            # Validation
            model.eval()
            val_loss = 0
            correct = 0
            
            with torch.no_grad():
                for data, target in self.val_loader:
                    data, target = data.to(self.device), target.to(self.device)
                    output = model(data)
                    val_loss += F.cross_entropy(output, target, reduction='sum').item()
                    pred = output.argmax(dim=1, keepdim=True)
                    correct += pred.eq(target.view_as(pred)).sum().item()
            
            val_loss /= len(self.val_loader.dataset)
            accuracy = correct / len(self.val_loader.dataset)
            
            # Report intermediate value for pruning
            trial.report(accuracy, epoch)
            
            # Handle pruning
            if trial.should_prune():
                raise optuna.TrialPruned()
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break
        
        return accuracy
    
    def optimize(self, n_trials=100):
        """Run hyperparameter optimization"""
        study = optuna.create_study(
            direction='maximize',
            sampler=TPESampler(seed=42),
            pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=10)
        )
        
        study.optimize(self.objective, n_trials=n_trials)
        
        print("Best trial:")
        print(f"  Value: {study.best_value}")
        print("  Params: ")
        for key, value in study.best_params.items():
            print(f"    {key}: {value}")
        
        return study.best_params
```

---

## まとめ

### 本章で習得した技術

1. **最先端アーキテクチャ**: Transformer、Vision Transformer の実装
2. **LLM最適化**: LoRA、量子化、メモリ効率的な学習
3. **強化学習**: DQN、PPO の実装と応用
4. **AutoML/NAS**: アーキテクチャ探索とハイパーパラメータ最適化

### 実装のポイント

#### パフォーマンス最適化
- **Mixed Precision Training**: FP16/BF16 活用
- **Gradient Checkpointing**: メモリ効率向上
- **Model Parallelism**: 大規模モデル学習
- **Efficient Attention**: Flash Attention、Sparse Attention

#### デバッグ・監視
- **Gradient Flow**: 勾配の可視化
- **Activation Statistics**: 活性化分布の監視
- **Loss Landscape**: 損失関数の可視化
- **Attention Weights**: アテンション重みの分析

### 今後の学習方向

1. **最新論文の実装**: arXiv の論文を実装
2. **コンペティション参加**: Kaggle、NeurIPS challenges
3. **オープンソース貢献**: PyTorch、Transformers への貢献
4. **独自研究**: 新しいアーキテクチャの提案

---

## 参考資料
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
- [Rainbow: Combining Improvements in Deep RL](https://arxiv.org/abs/1710.02298)
- [DARTS: Differentiable Architecture Search](https://arxiv.org/abs/1806.09055)