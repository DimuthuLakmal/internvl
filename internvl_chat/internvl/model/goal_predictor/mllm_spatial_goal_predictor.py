import torch
import torch.nn as nn
import torch.nn.functional as F


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

        self.up1 = self._up_block(base_channels, base_channels)
        self.up2 = self._up_block(base_channels, base_channels // 2)
        self.up3 = self._up_block(base_channels // 2, base_channels // 4)
        self.up4 = self._up_block(base_channels // 4, base_channels // 8)
        self.up5 = self._up_block(base_channels // 8, base_channels // 16)

        self.head = CoordConvHead(base_channels // 16, output_channels)

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

        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        x = self.up4(x)
        x = self.up5(x)

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
    ):
        super().__init__()

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

        x = self.input_proj(hidden_states)          # [B, T, D]
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
        }

        self.map_size = map_size

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
        inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)
        outputs = self.llm_model(
            **inputs,
            output_hidden_states=True,
            return_dict=True,
            use_cache=False)

        hidden_states = outputs.hidden_states[-1]  # [B, 8192, H]
        hidden_states = hidden_states.to(self.sequence_encoder.input_proj.weight.dtype)
        attention_mask = inputs.get("attention_mask", None)

        feat = self.sequence_encoder(
            hidden_states,
            valid_token_mask=attention_mask,
        )

        ego_hist = ego_hist.to(feat.dtype)
        x_hist = self.x_hist_encoder(ego_hist)
        x_pred = self.goal_head(torch.cat([feat, x_hist], dim=-1)) + self.x_prior(ego_hist)

        # y reads road geometry from the map image tokens only (keeps x's text shortcut out),
        # plus the ego's numeric history for the direction of lateral drift
        image_mask = inputs["input_ids"] == self.llm_model.img_context_token_id
        if attention_mask is not None:
            image_mask = image_mask & attention_mask.bool()
        y_feat = self.y_sequence_encoder(hidden_states, valid_token_mask=image_mask)
        hist_feat = self.hist_encoder(ego_hist.to(y_feat.dtype))
        y_pred = self.y_head(torch.cat([y_feat, hist_feat], dim=-1))
        pred_xy_norm = torch.cat([x_pred, y_pred], dim=-1)

        return pred_xy_norm