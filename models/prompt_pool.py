import torch
import torch.nn as nn


# --------------------------- L2P Implementation -------------------------------
# Adapted from Google's L2P

def expand_to_batch(x: torch.Tensor, batch_size: int, dim = 0):
    shape = list(x.shape)
    shape.insert(dim, batch_size)
    return x.unsqueeze(dim = dim).expand(*shape)


class PromptPool(nn.Module):

    batchwise_prompt: bool = False

    def __init__(
        self, 
        embed_dim: int,
        length: int,
        embedding_key: str = "mean", 
        pool_size: int = 0,
        num_layers: int = 1, 
        num_heads: int = 1,
        top_k: int = 1
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.length = length
        self.embedding_key = embedding_key
        self.pool_size = pool_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.top_k = top_k

        self.prompt = nn.Parameter(torch.zeros(
            self.num_layers, self.pool_size, self.length, self.embed_dim
        ))

        self.key_shape = (self.pool_size, self.embed_dim)
        self.prompt_key = nn.Parameter(torch.zeros(
            self.key_shape[0], self.key_shape[1]
        ))
    
    def extract_query(self, x_embed):
         return torch.mean(x_embed, dim = 1)

    def select_prompts(self, prompt_key, x_embed_mean, batch_size, res):
        prompt_key_norm = torch.linalg.norm(prompt_key, dim = -1)
        x_embed_norm = torch.linalg.norm(x_embed_mean, dim = -1)

        csim = prompt_key_norm @ torch.transpose(x_embed_norm, 0, 1)
        csim = torch.transpose(csim, 0, 2)

        top_k_tensor = torch.topk(csim, self.top_k)
        csim_top_k = top_k_tensor.values
        idx = top_k_tensor.indices

        prompt_id, id_counts = torch.unique(idx, return_counts = True)
        
        major = torch.topk(id_counts, self.top_k)
        major_prompt_id = prompt_id[major.indices]
        idx = expand_to_batch(major_prompt_id, batch_size)

        self.res["prompt_idx"] = idx

        batched_key_norm = torch.take(
            prompt_key_norm, idx, dim = 0
        )
        self.res["selected_key"] = batched_key_norm

        batched_prompt_raw = torch.take(
            self.prompt, idx, dim = 1
        )
        num_layers, bs, allowed_size, prompt_len, embed_dim = batched_prompt_raw.shape

        batched_prompt = torch.reshape(
            batched_prompt_raw,
            (num_layers, bs, allowed_size * prompt_len, embed_dim)
        )

        res["batched_prompt"] = batched_prompt
        res["prompt_key_norm"] = prompt_key_norm
        res["x_embed_norm"] = x_embed_norm
        res["sim"] = csim


