# generate_v3.py
import torch
import torch.nn as nn
from torch.nn import functional as F
import math
from dataclasses import dataclass, field
from contextlib import nullcontext
import os
import tiktoken
import argparse 
import time

# --- NECESSARY CLASS DEFINITIONS ---
# (Must match the NEW 124M BPE model architecture)

@dataclass
class SLMConfig:
    # --- V3 Model Architecture (GPT-2 Small, ~124M) ---
    block_size: int = 256
    vocab_size: int = 50257
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.1 # Dropout is 0 during eval, so this value doesn't matter
    bias: bool = False
    # Add other fields with defaults only if needed
    device: str = 'cuda'
    dtype: str = 'bfloat16'
    compile: bool = True # Set to True to compile for inference

class LayerNorm(nn.Module):
    def __init__(self, ndim, bias): super().__init__(); self.weight = nn.Parameter(torch.ones(ndim)); self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None
    def forward(self, x): return F.layer_norm(x, self.weight.shape, self.weight, self.bias, 1e-5)

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__(); assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias); self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout); self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head; self.n_embd = config.n_embd; self.dropout = config.dropout
        self.flash = hasattr(F, 'scaled_dot_product_attention')
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size)).view(1, 1, config.block_size, config.block_size))
    def forward(self, x):
        B, T, C = x.size(); q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2); q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2); v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        is_training = self.training; dropout_val = self.dropout if is_training else 0.0
        if self.flash and not is_training:
             try: y = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=True)
             except Exception: self.flash = False; # Fallthrough
        if not self.flash or is_training:
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            current_T = min(T, self.bias.size(-1)); att = att.masked_fill(self.bias[:,:,:current_T,:current_T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1);
            if dropout_val > 0.0: att = self.attn_dropout(att)
            y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C); proj_output = self.c_proj(y)
        if dropout_val > 0.0: y = self.resid_dropout(proj_output)
        else: y = proj_output
        return y

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__(); self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU(approximate='tanh'); self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)
    def forward(self, x):
        x = self.c_fc(x); x = self.gelu(x); x = self.c_proj(x)
        if self.training: x = self.dropout(x)
        return x

class Block(nn.Module):
    def __init__(self, config): super().__init__(); self.ln_1 = LayerNorm(config.n_embd, bias=config.bias); self.attn = CausalSelfAttention(config); self.ln_2 = LayerNorm(config.n_embd, bias=config.bias); self.mlp = MLP(config)
    def forward(self, x): x = x + self.attn(self.ln_1(x)); x = x + self.mlp(self.ln_2(x)); return x

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__(); assert config.vocab_size == 50257; assert config.block_size is not None; self.config = config
        self.transformer = nn.ModuleDict(dict(wte = nn.Embedding(config.vocab_size, config.n_embd), wpe = nn.Embedding(config.block_size, config.n_embd), drop = nn.Dropout(config.dropout), h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]), ln_f = LayerNorm(config.n_embd, bias=config.bias)))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False); self.transformer.wte.weight = self.lm_head.weight
    def get_num_params(self, non_embedding=True): return sum(p.numel() for p in self.parameters())
    def _init_weights(self, module): # Keep for reference
        if isinstance(module, nn.Linear): torch.nn.init.normal_(module.weight, mean=0.0, std=0.02);
        if module.bias is not None: torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding): torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    def forward(self, idx, targets=None):
        device = idx.device; b, t = idx.size();
        if t > self.config.block_size: idx = idx[:, -self.config.block_size:]; t = self.config.block_size
        pos = torch.arange(0, t, dtype=torch.long, device=device); tok_emb = self.transformer.wte(idx); pos_emb = self.transformer.wpe(pos)
        x = self.transformer.drop(tok_emb + pos_emb) if self.training else tok_emb + pos_emb
        for block in self.transformer.h: x = block(x)
        x = self.transformer.ln_f(x)
        if targets is not None: logits = self.lm_head(x); loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else: logits = self.lm_head(x[:, [-1], :]); loss = None
        return logits, loss
    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        self.eval()
        enc = tiktoken.get_encoding("gpt2")
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None: v, _ = torch.topk(logits, min(top_k, logits.size(-1))); logits[logits < v[:, [-1]]] = -float('Inf')
            probs = F.softmax(logits, dim=-1); idx_next = torch.multinomial(probs, num_samples=1)
            if idx_next == enc.eot_token: break
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

# --- END OF MODEL DEFINITIONS ---

def load_model(checkpoint_path, device, compile_model=True):
    """Loads the model from a checkpoint."""
    print(f"Loading checkpoint from: {checkpoint_path}")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    # --- Load Config ---
    config_to_load = None
    if 'config' in checkpoint:
        saved_config_data = checkpoint['config']
        if isinstance(saved_config_data, dict):
            config_fields = {f.name for f in SLMConfig.__dataclass_fields__.values()}
            filtered_args = {k: v for k, v in saved_config_data.items() if k in config_fields}
            for f_name, f_field in SLMConfig.__dataclass_fields__.items():
                has_default = hasattr(f_field, 'default') and f_field.default is not None
                if f_name not in filtered_args and has_default:
                      filtered_args[f_name] = f_field.default
            try: config_to_load = SLMConfig(**filtered_args)
            except Exception as e: print(f"Warning: Error creating config from checkpoint dict: {e}")
        elif isinstance(saved_config_data, SLMConfig):
             config_to_load = saved_config_data
    
    # Fallback to default 124M config if load fails
    if config_to_load is None:
        print("Warning: Could not load config from checkpoint. Using default 124M SLMConfig.")
        config_to_load = SLMConfig() 

    # --- Instantiate Model ---
    model = GPT(config_to_load)
    print(f"Instantiated model with {model.get_num_params()/1e6:.2f}M parameters.")

    # --- Load State Dict ---
    if 'model' in checkpoint and isinstance(checkpoint['model'], dict): state_dict = checkpoint['model']
    else: state_dict = checkpoint; print("Loading state_dict directly.")
    uncompiled_state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
    missing_keys, unexpected_keys = model.load_state_dict(uncompiled_state_dict, strict=False)
    if missing_keys: print(f"Warning: Missing keys: {missing_keys}")
    if unexpected_keys: print(f"Warning: Unexpected keys: {unexpected_keys}")
    print("Loaded model weights.")

    model.to(device)
    model.eval()
    print(f"Moved model to {device} and set to eval mode.")

    if compile_model:
         print("Compiling model...")
         try: model = torch.compile(model); print("Model compiled.")
         except Exception as e: print(f"Compile failed: {e}")

    return model, config_to_load

# --- Main Execution ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate text using the 124M V3 SLM model.")
    parser.add_argument("--prompt", type=str, default="Once upon a time", help="Input prompt for the model.")
    # --- DEFAULT CHECKPOINT PATH UPDATED ---
    parser.add_argument("--checkpoint_path", type=str, default="out_v3/best_model_v3.pt", help="Path to the V3 model checkpoint (.pt file).")
    parser.add_argument("--max_new_tokens", type=int, default=150, help="Maximum number of new tokens to generate.")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature.")
    parser.add_argument("--top_k", type=int, default=50, help="Top-k sampling. Use 0 to disable.")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"], help="Device to run inference on.")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["float32", "float16", "bfloat16"], help="Datatype for inference.")
    parser.add_argument("--no_compile", action="store_true", help="Disable torch.compile for this run.")
    
    args = parser.parse_args()
    
    device = args.device
    if device == 'cuda' and not torch.cuda.is_available(): device = 'cpu'; print("CUDA not found, using CPU.")
    dtype = args.dtype
    if dtype == 'bfloat16' and (device == 'cpu' or not torch.cuda.is_bf16_supported()): dtype = 'float16'; print("bfloat16 not supported, using float16.")
    if dtype == 'float16' and device == 'cpu': dtype = 'float32'; print("float16 not supported on CPU, using float32.")
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
    ctx = nullcontext() if device == 'cpu' else torch.amp.autocast(device_type=device, dtype=ptdtype)
    
    # Set default dtype based on device for generation
    torch.set_default_dtype(ptdtype if device=='cuda' else torch.float32) 
    
    # Load model
    model, loaded_config = load_model(args.checkpoint_path, device, compile_model=(not args.no_compile))
    
    # Initialize tokenizer
    enc = tiktoken.get_encoding("gpt2")
    
    # Tokenize prompt
    start_ids = enc.encode(args.prompt, allowed_special={"<|endoftext|>"})
    x = torch.tensor(start_ids, dtype=torch.long, device=device).unsqueeze(0)
    
    print(f"\n--- Generating response ---")
    print(f"Prompt: \"{args.prompt}\"")
    print("-" * 30)
    start_gen_time = time.time()
    
    with torch.inference_mode():
        with ctx:
            y = model.generate(x,
                               args.max_new_tokens,
                               temperature=args.temperature,
                               top_k=args.top_k if args.top_k > 0 else None)
    end_gen_time = time.time()

    generated_ids = y[0].tolist()
    generated_text = enc.decode(generated_ids)
    
    print("Generated Text:")
    print(generated_text)
    print("-" * 30)
    print(f"Generated {len(generated_ids) - len(start_ids)} tokens in {end_gen_time - start_gen_time:.3f} seconds.")
    
    print("Project Complete")