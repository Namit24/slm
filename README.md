
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

