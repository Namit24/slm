
# Small Language Model

A Small Language Model, as name suggests is a small version of LLM, The model here is trained (and still training) on datasets library which can be viewed from: https://huggingface.co/datasets/Amod/mental_health_counseling_conversations/viewer/default/train?p=1&views%5B%5D=train

It is a GPT based Transformer with configurable parameters:

6 layers

6 attention heads

384 embedding dimensions

~29.92M parameters

For Data: Currently it's trained on Tiny Stories Dataset and was tokenized on GPT Tokenizer using Tiktoken

Training: Supports mixed precision training (bfloat16/float16), gradient clipping, AdamW optimizer with cosine learning rate decay, and model compilation with torch.compile.


Inference: Provides standard and streaming text generation with temperature and top-k sampling.


Visualizations: Includes plots for training/validation loss, learning rate schedule, weight distributions, and attention heatmaps.




Performance: Achieves ~714.87 tokens/second on an NVIDIA GeForce RTX 3050 Laptop GPU with bfloat16 and model compilation.

Here are different visuals from the model:

<img width="842" height="470" alt="image" src="https://github.com/user-attachments/assets/4437afb6-e4fb-469f-ab4a-1527bcb754ad" />
<img width="833" height="731" alt="image" src="https://github.com/user-attachments/assets/4acbf35a-8d26-425f-b5ef-c0e04b3b18bd" />
<img width="1189" height="955" alt="image" src="https://github.com/user-attachments/assets/c74b91fb-80c3-43cc-9173-d825f1dcff33" />
<img width="881" height="470" alt="image" src="https://github.com/user-attachments/assets/35a5ca14-6f41-42b1-a2e8-60c30d3cec3d" />

Requirements:
Python 3.13
CUDA 12.9+

1. Data Preparation
The script downloads and tokenizes the TinyStories dataset, saving it as memory-mapped binary files (train.bin and validation.bin) in the data/ directory. To preprocess the data:
python# This is handled automatically in the script
from datasets import load_dataset
import tiktoken
import numpy as np

df = load_dataset("roneneldan/TinyStories")
encoding = tiktoken.get_encoding("gpt2")
# Tokenization and saving logic is included in the script
2. Training
The model is trained for 10,000 iterations with a batch size of 32, using the AdamW optimizer, cosine learning rate decay, and mixed precision training. Checkpoints are saved in the out/ directory when validation loss improves.
python# Training is initiated in the script
model = GPT(config)
optimizer = torch.optim.AdamW(param_groups, lr=config.learning_rate, betas=(config.beta1, config.beta2))
# Training loop with loss tracking and checkpointing
To run the training:
bashpython <script_name>.py
3. Inference
The script supports both standard and streaming text generation. Example prompts are included:
pythonprompt = "Once upon a time"
context = torch.tensor(encoding.encode_ordinary(prompt), dtype=torch.long, device=config.device).unsqueeze(0)
with torch.inference_mode():
    with ctx:
        y = model.generate(context, max_new_tokens=200, temperature=0.8, top_k=20)
print(encoding.decode(y.squeeze().tolist()))
For streaming generation:
pythonprompt = "A little girl went to the woods"
context = torch.tensor(encoding.encode_ordinary(prompt), dtype=torch.long, device=config.device).unsqueeze(0)
with torch.inference_mode():
    with ctx:
        for token_tensor in generate_streaming(model, context, max_new_tokens=200, temperature=0.8, top_k=20):
            print(encoding.decode(token_tensor[0].tolist()), end='', flush=True)
4. Visualizations
The script generates the following plots:

Training vs. Validation Loss: Tracks loss over training steps.
Learning Rate Schedule: Visualizes the linear warmup and cosine decay.
Weight Distributions: Histograms of model weights (token embeddings, attention, MLP, and output layer).
Attention Heatmap: Displays attention weights for a specific head in the final layer.

To generate these plots, ensure matplotlib is installed and run the script. The plots are displayed automatically during training and evaluation.
5. Benchmarking
The script includes a performance benchmark for text generation:
pythonprompt = "Once upon a time"
context = torch.tensor(encoding.encode_ordinary(prompt), dtype=torch.long, device=config.device).unsqueeze(0)
with torch.inference_mode():
    with ctx:
        y = model.generate(context, max_new_tokens=200, temperature=0.8, top_k=20)