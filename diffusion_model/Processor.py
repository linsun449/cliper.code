import torch
import math

DIFFUSION_LAYERS = [
    'down_blocks[0].attentions[0].transformer_blocks[0].attn1',  # 0
    'down_blocks[0].attentions[0].transformer_blocks[0].attn2',  # 1
    'down_blocks[0].attentions[1].transformer_blocks[0].attn1',  # 2
    'down_blocks[0].attentions[1].transformer_blocks[0].attn2',  # 3
    'down_blocks[1].attentions[0].transformer_blocks[0].attn1',  # 4
    'down_blocks[1].attentions[0].transformer_blocks[0].attn2',  # 5
    'down_blocks[1].attentions[1].transformer_blocks[0].attn1',  # 6
    'down_blocks[1].attentions[1].transformer_blocks[0].attn2',  # 7
    'down_blocks[2].attentions[0].transformer_blocks[0].attn1',  # 8
    'down_blocks[2].attentions[0].transformer_blocks[0].attn2',  # 9
    'down_blocks[2].attentions[1].transformer_blocks[0].attn1',  # 10
    'down_blocks[2].attentions[1].transformer_blocks[0].attn2',  # 11

    'mid_block.attentions[0].transformer_blocks[0].attn1',
    'mid_block.attentions[0].transformer_blocks[0].attn2',

    'up_blocks[1].attentions[0].transformer_blocks[0].attn1',  # -18
    "up_blocks[1].attentions[0].transformer_blocks[0].attn2",  # -17
    'up_blocks[1].attentions[1].transformer_blocks[0].attn1',  # -16
    "up_blocks[1].attentions[1].transformer_blocks[0].attn2",  # -15
    'up_blocks[1].attentions[2].transformer_blocks[0].attn1',  # -14
    "up_blocks[1].attentions[2].transformer_blocks[0].attn2",  # -13
    'up_blocks[2].attentions[0].transformer_blocks[0].attn1',  # -12
    "up_blocks[2].attentions[0].transformer_blocks[0].attn2",  # -11
    'up_blocks[2].attentions[1].transformer_blocks[0].attn1',  # -10
    "up_blocks[2].attentions[1].transformer_blocks[0].attn2",  # -9
    'up_blocks[2].attentions[2].transformer_blocks[0].attn1',  # -8
    'up_blocks[2].attentions[2].transformer_blocks[0].attn2',  # -7
    "up_blocks[3].attentions[0].transformer_blocks[0].attn1",  # -6
    'up_blocks[3].attentions[0].transformer_blocks[0].attn2',  # -5
    "up_blocks[3].attentions[1].transformer_blocks[0].attn1",  # -4
    'up_blocks[3].attentions[1].transformer_blocks[0].attn2',  # -3
    "up_blocks[3].attentions[2].transformer_blocks[0].attn1",  # -2
    'up_blocks[3].attentions[2].transformer_blocks[0].attn2',  # -1
]


class AttnProcessorForCallBack:
    def __init__(self, model, layer):
        self.model = model
        self.layer = layer

    def __call__(
        self,
        attn,
        hidden_states: torch.Tensor,
        encoder_hidden_states=None,
        attention_mask=None,
        temb=None,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        attention_probs = attn.get_attention_scores(query, key, attention_mask)
        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        head_size = attn.heads
        batch_size, q_len, v_len = attention_probs.shape
        attention_probs = attention_probs.reshape(batch_size // head_size, head_size, q_len, v_len)
        self.model.attention_maps[self.layer] = attention_probs

        return hidden_states
