import math

import torch
from torch import nn
import torch.nn.functional as F


def causal_mask(length, device):
    return torch.triu(torch.ones((length, length), dtype=torch.bool, device=device), diagonal=1)


def initialize_weights(module):
    if isinstance(module, nn.Linear):
        nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, mean=0.0, std=0.02)
    elif isinstance(module, nn.LayerNorm):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)


class TokenBackbone(nn.Module):
    def __init__(self, vocab_size, max_length, d_model=256, n_layers=4, n_heads=8, d_ff=1024,
                 dropout=0.1, condition_dim=0):
        super().__init__()
        self.max_length = max_length
        self.d_model = d_model
        self.token = nn.Embedding(vocab_size, d_model)
        self.position = nn.Embedding(max_length, d_model)
        self.condition = nn.Linear(condition_dim, d_model) if condition_dim else None
        layer = nn.TransformerEncoderLayer(
            d_model, n_heads, d_ff, dropout, batch_first=True, norm_first=True, activation="gelu"
        )
        self.encoder = nn.TransformerEncoder(layer, n_layers)
        self.norm = nn.LayerNorm(d_model)

    def embed(self, ids, conditions=None):
        positions = torch.arange(ids.size(1), device=ids.device).unsqueeze(0)
        hidden = self.token(ids) * math.sqrt(self.d_model) + self.position(positions)
        if self.condition is not None:
            if conditions is None:
                conditions = torch.zeros(ids.size(0), self.condition.in_features, device=ids.device)
            hidden = hidden + self.condition(conditions).unsqueeze(1)
        return hidden

    def forward(self, ids, pad_id, causal=False, conditions=None):
        mask = causal_mask(ids.size(1), ids.device) if causal else None
        hidden = self.encoder(self.embed(ids, conditions), mask=mask, src_key_padding_mask=ids.eq(pad_id))
        return self.norm(hidden)


class AutoregressiveTransformer(nn.Module):
    def __init__(self, vocab_size, max_length, pad_id, **kwargs):
        super().__init__()
        self.pad_id = pad_id
        self.backbone = TokenBackbone(vocab_size, max_length, **kwargs)
        self.head = nn.Linear(self.backbone.d_model, vocab_size, bias=False)
        self.head.weight = self.backbone.token.weight

    def logits(self, ids, conditions=None):
        return self.head(self.backbone(ids, self.pad_id, causal=True, conditions=conditions))

    def loss(self, ids, conditions=None, **_):
        logits = self.logits(ids[:, :-1], conditions)
        return {"total": F.cross_entropy(logits.transpose(1, 2), ids[:, 1:], ignore_index=self.pad_id)}

    @torch.no_grad()
    def sample(self, prefix, eos_id, max_length, temperature=1.0, top_k=32, conditions=None):
        ids = prefix
        for _ in range(max_length - ids.size(1)):
            logits = self.logits(ids, conditions)[:, -1] / max(temperature, 1e-5)
            if top_k:
                values, indices = logits.topk(min(top_k, logits.size(-1)), dim=-1)
                probs = torch.softmax(values, dim=-1)
                next_token = indices.gather(1, torch.multinomial(probs, 1))
            else:
                next_token = torch.multinomial(torch.softmax(logits, dim=-1), 1)
            ids = torch.cat([ids, next_token], dim=1)
            if ids[:, -1].eq(eos_id).all():
                break
        return ids


class TransformerVAE(nn.Module):
    def __init__(self, vocab_size, max_length, pad_id, latent_dim=128, **kwargs):
        super().__init__()
        self.pad_id = pad_id
        self.max_length = max_length
        self.latent_dim = latent_dim
        self.backbone = TokenBackbone(vocab_size, max_length, **kwargs)
        d_model = self.backbone.d_model
        self.mu = nn.Linear(d_model, latent_dim)
        self.logvar = nn.Linear(d_model, latent_dim)
        self.latent_to_memory = nn.Linear(latent_dim, d_model)
        layer = nn.TransformerDecoderLayer(
            d_model, kwargs.get("n_heads", 8), kwargs.get("d_ff", 1024),
            kwargs.get("dropout", 0.1), batch_first=True, norm_first=True, activation="gelu"
        )
        self.decoder = nn.TransformerDecoder(layer, kwargs.get("n_layers", 4))
        self.head = nn.Linear(d_model, vocab_size)

    def encode(self, ids, conditions=None):
        hidden = self.backbone(ids, self.pad_id, conditions=conditions)
        valid = (~ids.eq(self.pad_id)).float().unsqueeze(-1)
        pooled = (hidden * valid).sum(1) / valid.sum(1).clamp_min(1)
        return self.mu(pooled), self.logvar(pooled).clamp(-12, 12)

    def decode_logits(self, decoder_ids, z, conditions=None):
        target = self.backbone.embed(decoder_ids, conditions)
        memory = self.latent_to_memory(z).unsqueeze(1)
        hidden = self.decoder(
            target, memory, tgt_mask=causal_mask(decoder_ids.size(1), decoder_ids.device),
            tgt_key_padding_mask=decoder_ids.eq(self.pad_id)
        )
        return self.head(hidden)

    def loss(self, ids, conditions=None, kl_weight=1.0, free_bits=0.02, deterministic=False, **_):
        mu, logvar = self.encode(ids, conditions)
        z = mu if deterministic else mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
        logits = self.decode_logits(ids[:, :-1], z, conditions)
        reconstruction = F.cross_entropy(logits.transpose(1, 2), ids[:, 1:], ignore_index=self.pad_id)
        kl_each = -0.5 * (1 + logvar - mu.square() - logvar.exp())
        kl = torch.clamp(kl_each, min=free_bits).sum(-1).mean()
        return {"total": reconstruction + kl_weight * kl, "reconstruction": reconstruction, "kl": kl}

    @torch.no_grad()
    def sample(self, prefix, eos_id, max_length, temperature=1.0, top_k=32, conditions=None):
        z = torch.randn(prefix.size(0), self.latent_dim, device=prefix.device)
        ids = prefix
        for _ in range(max_length - ids.size(1)):
            logits = self.decode_logits(ids, z, conditions)[:, -1] / max(temperature, 1e-5)
            values, indices = logits.topk(min(top_k, logits.size(-1)), dim=-1)
            next_token = indices.gather(1, torch.multinomial(torch.softmax(values, -1), 1))
            ids = torch.cat([ids, next_token], dim=1)
            if ids[:, -1].eq(eos_id).all():
                break
        return ids


class DiscreteDiffusion(nn.Module):
    def __init__(self, vocab_size, max_length, pad_id, mask_id, protected_ids, diffusion_steps=200, **kwargs):
        super().__init__()
        self.pad_id = pad_id
        self.mask_id = mask_id
        self.protected_ids = tuple(protected_ids)
        self.steps = diffusion_steps
        self.backbone = TokenBackbone(vocab_size, max_length, **kwargs)
        self.time = nn.Embedding(diffusion_steps + 1, self.backbone.d_model)
        self.head = nn.Linear(self.backbone.d_model, vocab_size)

    def logits(self, noisy, timesteps, conditions=None):
        hidden = self.backbone.embed(noisy, conditions) + self.time(timesteps).unsqueeze(1)
        hidden = self.backbone.encoder(hidden, src_key_padding_mask=noisy.eq(self.pad_id))
        return self.head(self.backbone.norm(hidden))

    def loss(self, ids, conditions=None, **_):
        batch = ids.size(0)
        t = torch.randint(1, self.steps + 1, (batch,), device=ids.device)
        probability = t.float() / self.steps
        protected = torch.zeros_like(ids, dtype=torch.bool)
        for token_id in self.protected_ids:
            protected |= ids.eq(token_id)
        corruptible = ~protected
        mask = torch.rand_like(ids.float()).lt(probability.unsqueeze(1)) & corruptible
        empty_rows = ~mask.any(dim=1)
        if empty_rows.any():
            for row in empty_rows.nonzero(as_tuple=False).flatten().tolist():
                candidates = corruptible[row].nonzero(as_tuple=False).flatten()
                if len(candidates):
                    choice = candidates[torch.randint(len(candidates), (1,), device=ids.device)]
                    mask[row, choice] = True
        noisy = ids.masked_fill(mask, self.mask_id)
        logits = self.logits(noisy, t, conditions)
        targets = ids.masked_fill(~mask, self.pad_id)
        denoising = F.cross_entropy(logits.transpose(1, 2), targets, ignore_index=self.pad_id)
        return {"total": denoising, "denoising": denoising}

    @torch.no_grad()
    def sample(self, prefix, eos_id, max_length, temperature=1.0, top_k=32, conditions=None,
               lengths=None, fixed_positions=None, fixed_token_ids=None):
        batch = prefix.size(0)
        ids = torch.full((batch, max_length), self.mask_id, device=prefix.device, dtype=torch.long)
        ids[:, :prefix.size(1)] = prefix
        fixed = torch.zeros_like(ids, dtype=torch.bool)
        fixed[:, :prefix.size(1)] = True
        if lengths is not None:
            lengths = lengths.to(device=ids.device, dtype=torch.long).clamp(prefix.size(1) + 2, max_length)
            positions = torch.arange(max_length, device=ids.device).unsqueeze(0)
            eos_positions = lengths - 1
            ids.scatter_(1, eos_positions.unsqueeze(1), eos_id)
            after_eos = positions > eos_positions.unsqueeze(1)
            ids = torch.where(after_eos, torch.full_like(ids, self.pad_id), ids)
            fixed |= after_eos | positions.eq(eos_positions.unsqueeze(1))
        if fixed_positions is not None and fixed_token_ids is not None:
            fixed_positions = fixed_positions.to(device=ids.device, dtype=torch.long)
            fixed_token_ids = fixed_token_ids.to(device=ids.device, dtype=torch.long)
            ids.scatter_(1, fixed_positions, fixed_token_ids)
            fixed.scatter_(1, fixed_positions, True)
        schedule = torch.linspace(self.steps, 1, min(50, self.steps)).round().long().unique_consecutive().tolist()
        for step_index, t in enumerate(schedule):
            times = torch.full((batch,), t, device=ids.device, dtype=torch.long)
            logits = self.logits(ids, times, conditions) / max(temperature, 1e-5)
            forbidden = set(self.protected_ids) | {self.mask_id}
            logits[..., list(forbidden)] = torch.finfo(logits.dtype).min
            values, indices = logits.topk(min(top_k, logits.size(-1) - len(forbidden)), dim=-1)
            local_probs = torch.softmax(values, -1)
            sampled = torch.multinomial(local_probs.reshape(-1, local_probs.size(-1)), 1)
            sampled = sampled.view(batch, max_length, 1)
            predicted = indices.gather(-1, sampled).squeeze(-1)
            confidence = local_probs.gather(-1, sampled).squeeze(-1)
            masked = ids.eq(self.mask_id) & ~fixed
            fraction_left = max(0.0, 1 - (step_index + 1) / len(schedule))
            remaining = math.ceil(fraction_left * (max_length - prefix.size(1)))
            fill_count = max(1, masked.sum(1).max().item() - remaining)
            scores = confidence.masked_fill(~masked, -1)
            positions = scores.topk(min(fill_count, scores.size(1)), dim=1).indices
            chosen = torch.zeros_like(masked)
            chosen.scatter_(1, positions, True)
            ids = torch.where(chosen & masked, predicted, ids)
        ids = torch.where(ids.eq(self.mask_id), torch.full_like(ids, eos_id), ids)
        return ids


class SequenceDiscriminator(nn.Module):
    def __init__(self, vocab_size, max_length, pad_id, **kwargs):
        super().__init__()
        self.pad_id = pad_id
        self.backbone = TokenBackbone(vocab_size, max_length, **kwargs)
        self.head = nn.Sequential(nn.Linear(self.backbone.d_model, self.backbone.d_model), nn.GELU(), nn.Linear(self.backbone.d_model, 1))

    def forward(self, ids):
        hidden = self.backbone(ids, self.pad_id)
        valid = (~ids.eq(self.pad_id)).float().unsqueeze(-1)
        pooled = (hidden * valid).sum(1) / valid.sum(1).clamp_min(1)
        return self.head(pooled).squeeze(-1)


class SeqGAN(AutoregressiveTransformer):
    def __init__(self, vocab_size, max_length, pad_id, **kwargs):
        super().__init__(vocab_size, max_length, pad_id, **kwargs)
        self.discriminator = SequenceDiscriminator(vocab_size, max_length, pad_id, **kwargs)


def create_model(name, vocab_size, max_length, tokenizer, model_config, condition_dim=0):
    common = dict(
        d_model=model_config["d_model"], n_layers=model_config["n_layers"],
        n_heads=model_config["n_heads"], d_ff=model_config["d_ff"], dropout=model_config["dropout"],
        condition_dim=condition_dim,
    )
    if name == "transformer":
        model = AutoregressiveTransformer(vocab_size, max_length, tokenizer.pad_id, **common)
    elif name == "vae":
        model = TransformerVAE(vocab_size, max_length, tokenizer.pad_id, latent_dim=model_config["latent_dim"], **common)
    elif name == "diffusion":
        protected = [tokenizer.pad_id, tokenizer.bos_id, tokenizer.eos_id,
                     tokenizer.vocab["<CAT>"], tokenizer.vocab["<ANI>"], tokenizer.vocab["<PAIR>"]]
        model = DiscreteDiffusion(vocab_size, max_length, tokenizer.pad_id, tokenizer.mask_id,
                                  protected, model_config["diffusion_steps"], **common)
    elif name == "seqgan":
        model = SeqGAN(vocab_size, max_length, tokenizer.pad_id, **common)
    else:
        raise ValueError(name)
    model.apply(initialize_weights)
    return model
