import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------- L2P Implementation -------------------------------
# Adapted from Google's L2P

def expand_to_batch(x: torch.Tensor, batch_size: int, dim = 0):
    shape = list(x.shape)
    shape.insert(dim, batch_size)
    return x.unsqueeze(dim = dim).expand(*shape)


class PromptPool(nn.Module):

    def __init__(
        self, 
        embed_dim: int,
        length: int,
        embedding_key: str = "mean", 
        pool_size: int = 1,
        num_layers: int = 1, 
        top_k: int = 1,
        batchwise_prompt: bool = False
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.length = length
        self.embedding_key = embedding_key
        self.pool_size = pool_size
        self.num_layers = num_layers
        self.top_k = top_k
        self.batchwise_prompt = batchwise_prompt

        self.prompt = nn.Parameter(torch.empty(
            self.num_layers, self.pool_size, self.length, self.embed_dim
        ))

        nn.init.uniform_(self.prompt, a = 0.0, b = 0.01)

        self.key_shape = (self.pool_size, self.embed_dim)
        self.prompt_key = nn.Parameter(torch.empty(
            self.key_shape[0], self.key_shape[1]
        ))

        nn.init.uniform_(self.prompt_key, a = -0.01, b = 0.01)
    
    def extract_query(self, x_embed):
         return torch.mean(x_embed, dim = 1)

    def select_prompts(self, prompt_key, x_embed_mean, res):
        batch_size = x_embed_mean.shape[0]
        prompt_key_norm = F.normalize(prompt_key, p = 2, dim = -1)
        x_embed_norm = F.normalize(x_embed_mean, p = 2, dim = -1)

        csim = x_embed_norm @ prompt_key_norm.transpose(0, 1)

        top_k_tensor = torch.topk(csim, self.top_k)
        csim_top_k = top_k_tensor.values
        idx = top_k_tensor.indices

        if self.batchwise_prompt:
            id_counts = torch.bincount(
                idx.flatten(),
                minlength = self.pool_size,
            )

            major_prompt_id = torch.topk(
                id_counts,
                self.top_k,
            ).indices
            idx = expand_to_batch(major_prompt_id, batch_size)


        res["prompt_idx"] = idx

        batched_key_norm = prompt_key_norm[idx]
        res["selected_key"] = batched_key_norm

        batched_prompt_raw = self.prompt[:, idx]
        num_layers, bs, allowed_size, prompt_len, embed_dim = batched_prompt_raw.shape

        batched_prompt = torch.reshape(
            batched_prompt_raw,
            (num_layers, bs, allowed_size * prompt_len, self.embed_dim)
        )

        res["batched_prompt"] = batched_prompt
        res["prompt_key_norm"] = prompt_key_norm
        res["x_embed_norm"] = x_embed_norm
        res["sim"] = csim
        res["similarity_top_k"] = csim_top_k

        x_embed_norm = x_embed_norm.unsqueeze(dim = 1)
        sim_pull = batched_key_norm * x_embed_norm
        reduce_sim = torch.sum(sim_pull) / batch_size
        res["reduce_sim"] = reduce_sim

        return res    

    def forward(self, x):
        res = {}

        x_embed_mean = self.extract_query(x)

        res = self.select_prompts(
            prompt_key = self.prompt_key,
            x_embed_mean = x_embed_mean,
            res = res
        )

        return res
