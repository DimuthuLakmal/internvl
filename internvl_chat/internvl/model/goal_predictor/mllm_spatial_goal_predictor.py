import torch
import torch.nn as nn
import torch.nn.functional as F

from internvl.train.constants import GOAL_TOKEN


class SoftSectionAttentionEncoder(nn.Module):
    """
    Soft section-aware encoder for MLLM hidden states.

    Optional section-level transformer is used after extracting section tokens.
    This is cheaper than applying a transformer over the full MLLM token sequence.
    """

    def __init__(
        self,
        input_dim,
        feat_dim=128,
        attn_dim=128,
        num_sections=4,
        dropout=0.1,
        use_section_transformer=True,
        section_transformer_layers=1,
        section_transformer_heads=4,
        section_ff_dim=256,
    ):
        super().__init__()

        self.num_sections = num_sections
        self.use_section_transformer = use_section_transformer

        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, attn_dim),
            nn.LayerNorm(attn_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        self.section_queries = nn.Parameter(
            torch.randn(1, num_sections, attn_dim)
        )

        self.key_proj = nn.Linear(attn_dim, attn_dim)
        self.value_proj = nn.Linear(attn_dim, attn_dim)

        self.section_type_embedding = nn.Parameter(
            torch.randn(1, num_sections, attn_dim)
        )

        if use_section_transformer:
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=attn_dim,
                nhead=section_transformer_heads,
                dim_feedforward=section_ff_dim,
                dropout=dropout,
                batch_first=True,
                activation="gelu",
                norm_first=True,
            )

            self.section_transformer = nn.TransformerEncoder(
                encoder_layer,
                num_layers=section_transformer_layers,
            )
        else:
            self.section_transformer = nn.Identity()

        self.section_gate = nn.Sequential(
            nn.Linear(attn_dim * num_sections, num_sections),
            nn.Softmax(dim=-1),
        )

        self.output_proj = nn.Sequential(
            nn.Linear(attn_dim, feat_dim),
            nn.LayerNorm(feat_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

    def forward(self, hidden_states, attention_mask=None, return_details=False):
        """
        hidden_states: [B, T, D]
        attention_mask: optional [B, T], 1 = valid token, 0 = padding
        """
        b, t, _ = hidden_states.shape

        x = self.input_proj(hidden_states)  # [B, T, A]

        q = self.section_queries.expand(b, -1, -1)  # [B, S, A]
        k = self.key_proj(x)                        # [B, T, A]
        v = self.value_proj(x)                      # [B, T, A]

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (k.shape[-1] ** 0.5)

        if attention_mask is not None:
            mask = attention_mask[:, None, :].bool()
            attn_scores = attn_scores.masked_fill(~mask, -1e9)

        token_attn = torch.softmax(attn_scores, dim=-1)  # [B, S, T]

        section_feats = torch.matmul(token_attn, v)  # [B, S, A]

        section_feats = section_feats + self.section_type_embedding

        # Important addition:
        # Let the four semantic section features interact.
        section_feats = self.section_transformer(section_feats)  # [B, S, A]

        flat = section_feats.flatten(start_dim=1)  # [B, S * A]
        section_weights = self.section_gate(flat)  # [B, S]

        fused = torch.sum(
            section_feats * section_weights.unsqueeze(-1),
            dim=1,
        )  # [B, A]

        final_feat = self.output_proj(fused)  # [B, feat_dim]

        if return_details:
            return {
                "final_feat": final_feat,
                "section_feats": section_feats,
                "token_attn": token_attn,
                "section_weights": section_weights,
            }

        return final_feat


class CoordConvHead(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv2d(in_ch + 2, in_ch, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(in_ch, out_ch, kernel_size=1),
        )

    def forward(self, x):
        B, C, H, W = x.shape
        device = x.device

        yy, xx = torch.meshgrid(
            torch.linspace(-1, 1, H, device=device),
            torch.linspace(-1, 1, W, device=device),
            indexing="ij",
        )

        coords = torch.stack([xx, yy], dim=0)
        coords = coords.unsqueeze(0).expand(B, -1, -1, -1)

        x = torch.cat([x, coords], dim=1)

        return self.head(x)


class MLLMRasterGoalDecoder(nn.Module):
    def __init__(
        self,
        feat_dim=128,
        map_size=224,
        base_channels=128,
        output_channels=4,
        init_grid_size=7,
    ):
        super().__init__()

        self.map_size = map_size
        self.init_grid_size = init_grid_size
        self.base_channels = base_channels

        self.fc = nn.Sequential(
            nn.Linear(feat_dim, base_channels * init_grid_size * init_grid_size),
            nn.LayerNorm(base_channels * init_grid_size * init_grid_size),
            nn.GELU(),
        )

        # Each block doubles the grid: map_size = init_grid_size * 2**n lands exactly on map_size,
        # anything else uses all 5 blocks and is interpolated in forward()
        num_up = 5
        for n in range(1, 6):
            if init_grid_size * 2 ** n == map_size:
                num_up = n
                break

        out_channels = [base_channels, base_channels // 2, base_channels // 4,
                        base_channels // 8, base_channels // 16][:num_up]
        in_channels = [base_channels] + out_channels[:-1]
        self.up_blocks = nn.Sequential(*[
            self._up_block(in_ch, out_ch) for in_ch, out_ch in zip(in_channels, out_channels)
        ])

        self.head = CoordConvHead(out_channels[-1], output_channels)

    @staticmethod
    def _up_block(in_ch, out_ch):
        return nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, 4, stride=2, padding=1),
            nn.GroupNorm(num_groups=min(8, out_ch), num_channels=out_ch),
            nn.GELU(),
        )

    def forward(self, feat):
        B = feat.shape[0]

        x = self.fc(feat)
        x = x.reshape(
            B,
            self.base_channels,
            self.init_grid_size,
            self.init_grid_size,
        )

        x = self.up_blocks(x)

        pred_map = self.head(x)

        if pred_map.shape[-1] != self.map_size:
            pred_map = F.interpolate(
                pred_map,
                size=(self.map_size, self.map_size),
                mode="bilinear",
                align_corners=False,
            )

        return pred_map


class LatentTransformerEncoder(nn.Module):
    def __init__(
        self,
        input_dim,
        latent_dim=256,
        num_latents=64,
        num_heads=8,
        num_layers=2,
        ff_dim=512,
        dropout=0.1,
        feat_dim=256,
        input_norm=False,
    ):
        super().__init__()

        # Off by default so MLLMGoalPredictor checkpoints (no norm weights) still load strictly
        self.input_norm = nn.LayerNorm(input_dim) if input_norm else nn.Identity()
        self.input_proj = nn.Linear(input_dim, latent_dim)

        self.latents = nn.Parameter(
            torch.randn(1, num_latents, latent_dim)
        )

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=latent_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=latent_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        self.norm = nn.LayerNorm(latent_dim)

        self.pool_gate = nn.Sequential(
            nn.Linear(latent_dim, 1),
        )

        self.output_proj = nn.Sequential(
            nn.Linear(latent_dim, feat_dim),
            nn.LayerNorm(feat_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, hidden_states, valid_token_mask=None):
        """
        hidden_states: [B, T, H]
        valid_token_mask: [B, T], True = usable token
        """
        B = hidden_states.shape[0]

        x = self.input_proj(self.input_norm(hidden_states))  # [B, T, D]
        q = self.latents.expand(B, -1, -1)          # [B, L, D]

        key_padding_mask = None
        if valid_token_mask is not None:
            key_padding_mask = ~valid_token_mask.bool()

        z, _ = self.cross_attn(
            query=q,
            key=x,
            value=x,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )                                           # [B, L, D]

        z = self.transformer(z)                     # [B, L, D]
        z = self.norm(z)

        weights = torch.softmax(self.pool_gate(z), dim=1)
        feat = torch.sum(z * weights, dim=1)        # [B, D]

        # feat = self.output_proj(feat)               # [B, feat_dim]

        return feat


class MLLMGoalPredictor(nn.Module):
    """
    SpatialPlanner-style model that uses MLLM hidden states instead of images.

    It predicts a raster-space goal map:
        pred_map: [B, 4, H, W]

    Channels:
        0: location logits
        1: raster x residual
        2: raster y residual
        3: yaw
    """

    def __init__(
        self,
        mllm_hidden_dim,
        feat_dim=256,
        attn_dim=128,
        num_sections=4,
        map_size=224,
        dropout=0.1,
        ablate_mllm=False,
        llm_model=None,
        llm_tokenizer=None,
    ):
        super().__init__()

        self.init_config = {
            "mllm_hidden_dim": mllm_hidden_dim,
            "feat_dim": feat_dim,
            "attn_dim": attn_dim,
            "num_sections": num_sections,
            "map_size": map_size,
            "dropout": dropout,
            "ablate_mllm": ablate_mllm,
        }

        self.map_size = map_size
        # Diagnostic: heads see only ego_hist, to measure what the MLLM features add
        self.ablate_mllm = ablate_mllm

        self.llm_model = llm_model
        self.llm_tokenizer = llm_tokenizer
        self.llm_config = dict(
            num_beams=1,
            max_new_tokens=1024,
            do_sample=False,
            temperature=0,
            return_dict_in_generate=True,
            output_logits=False,
            output_scores=False,
            output_hidden_states=True
        )


        self.sequence_encoder = LatentTransformerEncoder(
            input_dim=mllm_hidden_dim,
            latent_dim=256,
            num_latents=256,
            num_heads=8,
            num_layers=2,
            ff_dim=512,
            dropout=dropout,
            feat_dim=feat_dim,
        )

        # x (forward progress) is mostly set by ego speed, which the LLM can't read reliably
        # from the coordinate text, so the x head also gets the numeric ego history
        self.x_hist_encoder = nn.Sequential(
            nn.Linear(40, 64),
            nn.GELU(),
            nn.Linear(64, 64),
        )

        self.goal_head = nn.Sequential(
            nn.Linear(256 + 64, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1),
        )

        # Linear kinematic prior (e.g. constant-velocity extrapolation); goal_head learns the residual.
        # Zero-init so training starts from the plain head.
        self.x_prior = nn.Linear(40, 1)
        nn.init.zeros_(self.x_prior.weight)
        nn.init.zeros_(self.x_prior.bias)

        # y has no signal in the coordinate text, so it gets its own encoder over image tokens only
        self.y_sequence_encoder = LatentTransformerEncoder(
            input_dim=mllm_hidden_dim,
            latent_dim=256,
            num_latents=64,
            num_heads=8,
            num_layers=2,
            ff_dim=512,
            dropout=dropout,
            feat_dim=feat_dim,
        )

        # The ego's lateral drift (a few px) carries the sign of y but is too fine for image tokens
        self.hist_encoder = nn.Sequential(
            nn.Linear(40, 64),
            nn.GELU(),
            nn.Linear(64, 64),
        )

        self.y_head = nn.Sequential(
            nn.Linear(256 + 64, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1),
        )

        # self.raster_decoder = MLLMRasterGoalDecoder(
        #     feat_dim=feat_dim,
        #     map_size=map_size,
        #     base_channels=128,
        #     output_channels=4,
        #     init_grid_size=7,
        # )

    def _bundle_tensors(self, tensor_list, max_seq_len=1024, pad_value=0.0):
        batch_size = len(tensor_list)
        feature_dim = tensor_list[0].shape[1]

        # Create output tensor
        batch_tensor = torch.full(
            (batch_size, max_seq_len, feature_dim),
            fill_value=pad_value,
            dtype=tensor_list[0].dtype,
            device=tensor_list[0].device,
        )

        mask = torch.zeros(
                (batch_size, max_seq_len),
                dtype=torch.bool,
                device=tensor_list[0].device,
            )

        for i, x in enumerate(tensor_list):
            seq_len = min(x.shape[0], max_seq_len)
            batch_tensor[i, :seq_len] = x[:seq_len]
            mask[i, :seq_len] = True

        return batch_tensor, mask

    def forward(
        self,
        inputs,
    ):
        """
        obs_dict should contain:
            mllm_hidden_states: [B, L, D]
            mllm_attention_mask: [B, L], optional

        Optional:
            drivable_map: [B, H, W]
            agent_from_raster: [B, 3, 3], only needed if convert_to_agent=True

        Returns:
            predictions["positions_raster"]: [B, N, 2]
            predictions["yaws"]:             [B, N, 1]
            predictions["positions"]:        [B, N, 2], if convert_to_agent=True
        """

        # all_hidden_states = generate_llm_output(obs_dict['llm_text'], obs_dict['llm_images'], self.llm_config, self.llm_model, self.llm_tokenizer)
        ego_hist = inputs.pop("ego_hist")  # [B, 40], consumed by the y branch, not the LLM

        if self.ablate_mllm:
            ego_hist = ego_hist.float()
            zeros = ego_hist.new_zeros(ego_hist.shape[0], 256)  # stands in for feat / y_feat
            with torch.autocast(device_type=ego_hist.device.type, enabled=False):
                x_hist = self._fp32_call(self.x_hist_encoder, ego_hist)
                x_pred = (self._fp32_call(self.goal_head, torch.cat([zeros, x_hist], dim=-1))
                          + self._fp32_call(self.x_prior, ego_hist))
                hist_feat = self._fp32_call(self.hist_encoder, ego_hist)
                y_pred = self._fp32_call(self.y_head, torch.cat([zeros, hist_feat], dim=-1))
            return torch.cat([x_pred, y_pred], dim=-1)

        inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)
        outputs = self.llm_model(
            **inputs,
            output_hidden_states=True,
            return_dict=True,
            use_cache=False)

        # Predictor runs in fp32: bf16 outputs quantize coordinates to ~0.5-1px
        hidden_states = outputs.hidden_states[-1].float()  # [B, 8192, H]
        attention_mask = inputs.get("attention_mask", None)
        ego_hist = ego_hist.float()

        # y reads road geometry from the map image tokens only (keeps x's text shortcut out),
        # plus the ego's numeric history for the direction of lateral drift
        image_mask = inputs["input_ids"] == self.llm_model.img_context_token_id
        if attention_mask is not None:
            image_mask = image_mask & attention_mask.bool()

        with torch.autocast(device_type=hidden_states.device.type, enabled=False):
            feat = self._fp32_call(self.sequence_encoder, hidden_states, valid_token_mask=attention_mask)
            x_hist = self._fp32_call(self.x_hist_encoder, ego_hist)
            x_pred = (self._fp32_call(self.goal_head, torch.cat([feat, x_hist], dim=-1))
                      + self._fp32_call(self.x_prior, ego_hist))

            y_feat = self._fp32_call(self.y_sequence_encoder, hidden_states, valid_token_mask=image_mask)
            hist_feat = self._fp32_call(self.hist_encoder, ego_hist)
            y_pred = self._fp32_call(self.y_head, torch.cat([y_feat, hist_feat], dim=-1))

        pred_xy_norm = torch.cat([x_pred, y_pred], dim=-1)

        return pred_xy_norm

    @staticmethod
    def _fp32_call(module, *args, **kwargs):
        """Run module with fp32 copies of its params; grads still flow to the (bf16) originals."""
        params_and_buffers = {k: v.float() for k, v in module.named_parameters()}
        params_and_buffers.update({k: v.float() for k, v in module.named_buffers()})
        return torch.func.functional_call(module, params_and_buffers, args, kwargs)

class LayerMix(nn.Module):
    """Learned softmax-weighted sum of per-layer LayerNormed hidden states (ELMo-style scalar mix)."""

    def __init__(self, hidden_dim, num_layers):
        super().__init__()
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])
        self.layer_logits = nn.Parameter(torch.zeros(num_layers))

    def forward(self, layer_states):
        weights = torch.softmax(self.layer_logits, dim=0)
        return sum(w * norm(h) for w, norm, h in zip(weights, self.norms, layer_states))


class MLLMHeatmapGoalPredictor(nn.Module):
    """
    Predicts the goal as a classification over a coarse raster grid plus an in-cell offset,
    so distinct manoeuvres (straight / turn) stay separate modes instead of being averaged
    by a scalar regression.

    The LLM is read through K learnable query tokens appended to the prompt (their
    embeddings are injected by a hook on the input embedding, so they train with a frozen
    LLM) and through the map image tokens, both from a learned mix of several layers.

    forward() returns a dict:
        logits:       [B, G*G] goal-cell logits (row = y, col = x)
        offsets:      [B, 2, G, G] in-cell (dx, dy) offset, in cell units
        pred_xy_norm: [B, 2] decoded argmax goal, in the trainer's [-1, 1] convention
    """

    PIXEL_SIZE = 224

    def __init__(
        self,
        mllm_hidden_dim,
        num_goal_queries=8,
        llm_tap_layers=(-1, -5, -9, -13),
        heatmap_size=56,
        heatmap_sigma=1.0,
        offset_loss_weight=1.0,
        hist_dropout=0.3,
        dropout=0.1,
        ablate_mllm=False,
        llm_model=None,
        llm_tokenizer=None,
    ):
        super().__init__()

        llm_tap_layers = [int(i) for i in llm_tap_layers]

        self.init_config = {
            "mllm_hidden_dim": mllm_hidden_dim,
            "num_goal_queries": num_goal_queries,
            "llm_tap_layers": llm_tap_layers,
            "heatmap_size": heatmap_size,
            "heatmap_sigma": heatmap_sigma,
            "offset_loss_weight": offset_loss_weight,
            "hist_dropout": hist_dropout,
            "dropout": dropout,
            "ablate_mllm": ablate_mllm,
        }

        self.num_goal_queries = num_goal_queries
        self.llm_tap_layers = llm_tap_layers
        self.heatmap_size = heatmap_size
        self.heatmap_sigma = heatmap_sigma
        self.offset_loss_weight = offset_loss_weight
        self.hist_dropout = hist_dropout
        self.ablate_mllm = ablate_mllm

        self.llm_model = llm_model
        self.llm_tokenizer = llm_tokenizer

        self.goal_token_id = None
        self._embed_hook = None
        if llm_tokenizer is not None:
            self.goal_token_id = llm_tokenizer.convert_tokens_to_ids(GOAL_TOKEN)

        # Learnable query embeddings, written over the <GOAL> placeholders by _inject_goal_queries
        self.goal_query_embed = nn.Parameter(torch.randn(num_goal_queries, mllm_hidden_dim) * 0.02)
        if llm_model is not None and not ablate_mllm:
            embedding = llm_model.language_model.get_input_embeddings()
            with torch.no_grad():
                # Match the scale of real token embeddings (subsampled rows: the full matrix is large)
                embed_std = embedding.weight[::97].float().std().item()
                self.goal_query_embed.normal_(0.0, embed_std)
            self._embed_hook = embedding.register_forward_hook(self._inject_goal_queries)

        self.layer_mix = LayerMix(mllm_hidden_dim, len(llm_tap_layers))

        # Query tokens have seen the whole prompt (causal LLM), so they carry the fused scene summary
        self.query_encoder = LatentTransformerEncoder(
            input_dim=mllm_hidden_dim,
            latent_dim=256,
            num_latents=num_goal_queries,
            num_heads=8,
            num_layers=2,
            ff_dim=512,
            dropout=dropout,
            feat_dim=256,
            input_norm=True,
        )

        # Map image tokens: road geometry
        self.img_encoder = LatentTransformerEncoder(
            input_dim=mllm_hidden_dim,
            latent_dim=256,
            num_latents=64,
            num_heads=8,
            num_layers=2,
            ff_dim=512,
            dropout=dropout,
            feat_dim=256,
            input_norm=True,
        )

        self.hist_encoder = nn.Sequential(
            nn.Linear(40, 64),
            nn.GELU(),
            nn.Linear(64, 64),
        )

        # Stands in for hist_feat when history is dropped; zeros would mean "stopped at the anchor"
        self.hist_missing_embed = nn.Parameter(torch.randn(64) * 0.02)

        self.raster_decoder = MLLMRasterGoalDecoder(
            feat_dim=256 + 256 + 64,
            map_size=heatmap_size,
            base_channels=128,
            output_channels=3,
            init_grid_size=7,
        )

    def _inject_goal_queries(self, module, args, output):
        input_ids = args[0]
        goal_mask = input_ids == self.goal_token_id
        if not goal_mask.any():
            return output

        counts = goal_mask.sum(dim=-1)
        assert (counts == self.num_goal_queries).all(), \
            f"expected {self.num_goal_queries} {GOAL_TOKEN} tokens per sample, got {counts.tolist()}"

        # Row-major masking keeps each sample's K positions in order, matching the repeated queries
        output = output.clone()
        output[goal_mask] = self.goal_query_embed.to(output.dtype).repeat(input_ids.shape[0], 1)
        return output

    def _cell_size(self):
        return self.PIXEL_SIZE / self.heatmap_size

    def _decode(self, logits, offsets):
        g = self.heatmap_size
        idx = logits.argmax(dim=-1)  # [B]
        row = torch.div(idx, g, rounding_mode="floor")
        col = idx % g

        off = offsets.flatten(start_dim=2)  # [B, 2, G*G]
        off = off.gather(2, idx[:, None, None].expand(-1, 2, 1)).squeeze(-1)  # [B, 2]

        s = self._cell_size()
        pred_px = torch.stack([(col + 0.5 + off[:, 0]) * s,
                               (row + 0.5 + off[:, 1]) * s], dim=-1)
        return pred_px / (self.PIXEL_SIZE - 1) * 2.0 - 1.0

    def heatmap_loss(self, outputs, target_xy_px):
        """
        Soft-label CE against a Gaussian (heatmap_sigma, in cells) at the GT position,
        plus SmoothL1 on the offset of the GT cell. Returns (loss, {name: value}).
        """
        g = self.heatmap_size
        s = self._cell_size()
        logits = outputs["logits"].float()
        offsets = outputs["offsets"].float()

        t = target_xy_px.float().clamp(0, self.PIXEL_SIZE - 1) / s  # [B, 2] in cell units
        cell = t.floor().clamp(max=g - 1)

        centers = torch.arange(g, device=t.device, dtype=t.dtype) + 0.5
        gauss = torch.exp(-(centers[None, None, :] - t[:, :, None]) ** 2 / (2 * self.heatmap_sigma ** 2))  # [B, 2, G]
        soft_target = (gauss[:, 1, :, None] * gauss[:, 0, None, :]).flatten(start_dim=1)  # [B, G*G]
        soft_target = soft_target / soft_target.sum(dim=-1, keepdim=True)

        ce = -(soft_target * F.log_softmax(logits, dim=-1)).sum(dim=-1).mean()

        idx = (cell[:, 1] * g + cell[:, 0]).long()
        off = offsets.flatten(start_dim=2).gather(2, idx[:, None, None].expand(-1, 2, 1)).squeeze(-1)
        offset_loss = F.smooth_l1_loss(off, t - (cell + 0.5))

        loss = ce + self.offset_loss_weight * offset_loss
        return loss, {"ce": ce.detach(), "offset": offset_loss.detach()}

    def forward(self, inputs):
        ego_hist = inputs.pop("ego_hist").float()  # [B, 40]
        b = ego_hist.shape[0]

        if self.ablate_mllm:
            query_feat = ego_hist.new_zeros(b, 256)
            img_feat = ego_hist.new_zeros(b, 256)
        else:
            inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)
            outputs = self.llm_model(
                **inputs,
                output_hidden_states=True,
                return_dict=True,
                use_cache=False)

            input_ids = inputs["input_ids"]
            attention_mask = inputs.get("attention_mask", None)
            goal_mask = input_ids == self.goal_token_id
            image_mask = input_ids == self.llm_model.img_context_token_id
            if attention_mask is not None:
                image_mask = image_mask & attention_mask.bool()

            # Predictor runs in fp32: bf16 outputs quantize coordinates to ~0.5-1px
            layer_states = [outputs.hidden_states[i].float() for i in self.llm_tap_layers]

            with torch.autocast(device_type=ego_hist.device.type, enabled=False):
                mixed = self._fp32_call(self.layer_mix, layer_states)  # [B, T, H]
                query_tokens = mixed[goal_mask].view(b, self.num_goal_queries, -1)  # [B, K, H]
                query_feat = self._fp32_call(self.query_encoder, query_tokens)
                img_feat = self._fp32_call(self.img_encoder, mixed, valid_token_mask=image_mask)

        with torch.autocast(device_type=ego_hist.device.type, enabled=False):
            hist_feat = self._fp32_call(self.hist_encoder, ego_hist)
            if self.training and self.hist_dropout > 0:
                # Forces the decoder to localize the goal from the MLLM features alone on some samples
                drop = torch.rand(b, 1, device=hist_feat.device) < self.hist_dropout
                hist_feat = torch.where(drop, self.hist_missing_embed.float().expand_as(hist_feat), hist_feat)

            cond = torch.cat([query_feat, img_feat, hist_feat], dim=-1)
            pred_map = self._fp32_call(self.raster_decoder, cond)  # [B, 3, G, G]

        logits = pred_map[:, 0].flatten(start_dim=1)
        offsets = pred_map[:, 1:]

        return {
            "logits": logits,
            "offsets": offsets,
            "pred_xy_norm": self._decode(logits.detach(), offsets.detach()),
        }

    _fp32_call = staticmethod(MLLMGoalPredictor._fp32_call)
