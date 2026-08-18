from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torchvision.ops import RoIAlign
from transformers import MBartConfig, MBartForCausalLM, PreTrainedTokenizer
from transformers.file_utils import ModelOutput
from transformers.models.mbart.modeling_mbart import _expand_mask, _make_causal_mask

from tflop.loss import TableCL
from tflop.model.decoder.utils import apply_fast_mbart_decoder
try:
    from tflop.model.decoder.mbart_decoder_weighted import integrate_weighted_ce_into_decoder_v2
except ImportError:
    integrate_weighted_ce_into_decoder_v2 = None


class Grid2DSinusoidalEncoding(nn.Module):
    """
    2D Sinusoidal Positional Encoding for grid-based structures.
    
    This implementation creates sinusoidal encodings for both row and column dimensions,
    similar to TAPAS approach but with fixed sinusoidal patterns instead of learned embeddings.
    """
    
    def __init__(self, d_model, max_row_num=40, max_col_num=40):
        super().__init__()
        self.d_model = d_model
        self.max_row_num = max_row_num
        self.max_col_num = max_col_num
        
        # Split d_model between row and column encodings
        self.row_dim = d_model // 2
        self.col_dim = d_model - self.row_dim  # Handle odd d_model
        
        # Create sinusoidal encoding tables
        self.register_buffer('row_encoding_table', self._create_sinusoidal_table(max_row_num, self.row_dim))
        self.register_buffer('col_encoding_table', self._create_sinusoidal_table(max_col_num, self.col_dim))
        
    def _create_sinusoidal_table(self, max_len, d_model):
        """Create sinusoidal encoding table"""
        encoding_table = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) * 
                           -(math.log(10000.0) / d_model))
        
        encoding_table[:, 0::2] = torch.sin(position * div_term)
        encoding_table[:, 1::2] = torch.cos(position * div_term)
        
        return encoding_table
        
    def forward(self, row_ids, col_ids):
        """Generate 2D sinusoidal encodings for given row and column positions."""
        # Clamp indices to valid range
        row_ids = torch.clamp(row_ids, 0, self.max_row_num - 1)
        col_ids = torch.clamp(col_ids, 0, self.max_col_num - 1)
        
        # Get row and column encodings
        row_encoding = self.row_encoding_table[row_ids]  # (..., row_dim)
        col_encoding = self.col_encoding_table[col_ids]  # (..., col_dim)
        
        # Concatenate row and column encodings
        encoding = torch.cat([row_encoding, col_encoding], dim=-1)  # (..., d_model)
        
        return encoding
    
    def get_encoding_for_position(self, row, col):
        """Get encoding for a single (row, col) position"""
        row = torch.clamp(torch.tensor(row), 0, self.max_row_num - 1)
        col = torch.clamp(torch.tensor(col), 0, self.max_col_num - 1)
        
        row_encoding = self.row_encoding_table[row]
        col_encoding = self.col_encoding_table[col]
        
        return torch.cat([row_encoding, col_encoding], dim=0)


class GridPositionTracker:
    """Tracks grid position during inference to apply positional encoding to cell tokens."""
    def __init__(self):
        self.current_row = 0
        self.current_col = 0
    
    def reset(self):
        """Reset position to (0, 0)"""
        self.current_row = 0
        self.current_col = 0
    
    def update_and_get_position(self, new_token, tokenizer):
        """Update position based on new token and return position if it's a cell token."""
        # Convert token ID to string if needed
        if isinstance(new_token, torch.Tensor):
            new_token = new_token.item()
        
        if isinstance(new_token, int):
            try:
                token_str = tokenizer.decode([new_token])
            except:
                token_str = str(new_token)
        else:
            token_str = str(new_token)
        
        # Update position based on structure tokens
        if token_str in ["<nl>", "NL", "NL-tag"]:
            self.current_row += 1
            self.current_col = 0
        
        # Return position if this is a cell token and increment column
        if token_str in ["C", "L", "X", "U", "C-tag", "L-tag", "X-tag", "U-tag"]:
            position = (self.current_row, self.current_col)
            self.current_col += 1  # Move to next column after processing cell token
            return position
        
        return None


class MBARTDecoder(nn.Module):
    def __init__(
        self: "MBARTDecoder",
        tokenizer: PreTrainedTokenizer,
        decoder_layer: int,
        max_length: int,
        name_or_path: str,
        max_position_embeddings: Union[int, None] = None,
        use_fast: bool = False,
        input_size: Tuple[int] = None,  # (width, height)
        bbox_token_cnt: int = None,
        max_num_row: int = None,
        max_num_col: int = None,
        use_bbox_HiMulConET: bool = False,
        use_imgRoiAlign: bool = False,
        contrastive_loss_config: dict = None,
        empty_cell_ptr_loss_coeff: float = 0.5,
        non_empty_cell_ptr_loss_coeff: float = 0.5,
        use_adjacent_penalty: bool = False,
        adjacent_penalty_config: dict = None,
        use_row_col_embedding: bool = False,
        row_col_embedding_config: dict = None,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.decoder_layer = decoder_layer
        self.max_position_embeddings = (
            max_position_embeddings
            if max_position_embeddings is not None
            else max_length
        )
        self.max_length = max_length
        self.name_or_path = name_or_path
        self.use_fast = use_fast
        self.input_size = input_size
        self.bbox_token_cnt = bbox_token_cnt

        self.max_num_row = max_num_row
        self.max_num_col = max_num_col
        self.use_bbox_HiMulConET = use_bbox_HiMulConET
        self.use_imgRoiAlign = use_imgRoiAlign
        self.contrastive_loss_config = contrastive_loss_config
        self.empty_cell_ptr_loss_coeff = empty_cell_ptr_loss_coeff
        self.non_empty_cell_ptr_loss_coeff = non_empty_cell_ptr_loss_coeff
        
        # Store adjacent penalty configuration
        self.use_adjacent_penalty = use_adjacent_penalty
        if adjacent_penalty_config and hasattr(adjacent_penalty_config, '_content'):
            self.adjacent_penalty_config = dict(adjacent_penalty_config)
        else:
            self.adjacent_penalty_config = adjacent_penalty_config or {
                'max_distance': 2,
                'weights': {1: 2.0, 2: 1.5}
            }
        
        # Store grid positional encoding configuration
        self.use_row_col_embedding = use_row_col_embedding
        if row_col_embedding_config and hasattr(row_col_embedding_config, '_content'):
            self.row_col_embedding_config = dict(row_col_embedding_config)
        else:
            self.row_col_embedding_config = row_col_embedding_config or {
                'encoding_type': 'learned',
                'row_embedding_dim': 128,
                'col_embedding_dim': 128
            }

        self.config = MBartConfig(
            is_decoder=True,
            is_encoder_decoder=False,
            add_cross_attention=True,
            decoder_layers=self.decoder_layer,
            max_position_embeddings=self.max_position_embeddings,
            vocab_size=len(self.tokenizer),
            scale_embedding=True,
            add_final_layer_norm=True,
        )

        self.model = MBartForCausalLM(config=self.config)
        
        # Resize token embeddings to match tokenizer vocabulary size
        # This must be done before loading pretrained weights
        self.model.resize_token_embeddings(len(self.tokenizer))
        
        if self.use_fast:
            apply_fast_mbart_decoder(self.model)
        self.model.forward = (
            self.forward
        )  # to get cross attentions and utilize `generate` function

        self.model.config.is_encoder_decoder = True  # to get cross-attention
        self.model.model.decoder.embed_tokens.padding_idx = self.tokenizer.pad_token_id
        self.model.prepare_inputs_for_generation = self.prepare_inputs_for_inference
        self.model.model.decoder._prepare_decoder_attention_mask = (
            self._custom_prepare_decoder_attention_mask
        )
        self.get_token_ids_to_token()
        
        # Initialize position tracker for inference (per-batch support)
        if self.use_row_col_embedding:
            self.position_trackers = {}  # Will create per-batch trackers as needed
            
            # Check if we should use 2D sinusoidal encoding
            encoding_type = self.row_col_embedding_config.get('encoding_type', 'learned')
            
            print(f"\n⚙️  MBARTDecoder Grid Positional Embedding:")
            print(f"  • Enabled: {self.use_row_col_embedding}")
            print(f"  • Encoding type: {encoding_type}")
            print(f"  • Grid dimensions: {self.max_num_row}x{self.max_num_col}")
            print(f"  • Config: {self.row_col_embedding_config}")
            
            if encoding_type == 'sinusoidal':
                # Use 2D Sinusoidal Encoding
                self.grid_2d_encoding = Grid2DSinusoidalEncoding(
                    d_model=self.model.config.d_model,
                    max_row_num=self.max_num_row,
                    max_col_num=self.max_num_col
                )
                print(f"  • Created 2D Sinusoidal Encoding: d_model={self.model.config.d_model}")
                print(f"  • Row dim: {self.grid_2d_encoding.row_dim}, Col dim: {self.grid_2d_encoding.col_dim}")
            else:
                # Use learned embeddings (default behavior)
                row_dim = self.row_col_embedding_config.get('row_embedding_dim', 128)
                col_dim = self.row_col_embedding_config.get('col_embedding_dim', 128)
                
                self.row_embedding = nn.Embedding(self.max_num_row, row_dim)
                self.col_embedding = nn.Embedding(self.max_num_col, col_dim)
                self.pos_projection = nn.Linear(row_dim + col_dim, self.model.config.d_model)
                print(f"  • Created learned embeddings: row_dim={row_dim}, col_dim={col_dim}")
                
        if self.use_adjacent_penalty:
            print(f"\n🎯 MBARTDecoder Weighted CrossEntropy:")
            print(f"  • Enabled: {self.use_adjacent_penalty}")
            print(f"  • Config: {self.adjacent_penalty_config}")
            
        print("-" * 50)

        # Set up pointer decoder network parameters
        assert self.input_size is not None
        self.k_linear = nn.Linear(
            self.model.config.d_model, self.model.config.d_model, bias=False
        )
        self.q_linear = nn.Linear(
            self.model.config.d_model, self.model.config.d_model, bias=False
        )

        assert self.model.config.d_model % 4 == 0, "d_model must be divisible by 4"

        # Set up coordinate embedding, NOTE: padding_idx is set to input_size + 3, as input_size +1, +2 are used for dummy coordinates
        self.x_coord_embedding = nn.Embedding(
            self.input_size[0] + 4,
            self.model.config.d_model // 4,
            padding_idx=self.input_size[0] + 3,
        )
        self.y_coord_embedding = nn.Embedding(
            self.input_size[1] + 4,
            self.model.config.d_model // 4,
            padding_idx=self.input_size[1] + 3,
        )

        if self.use_bbox_HiMulConET:
            # Set up modules for row-wise and column-wise linear transformation
            self.rowwise_linear = nn.Linear(
                self.model.config.d_model, self.model.config.d_model, bias=False
            )
            self.colwise_linear = nn.Linear(
                self.model.config.d_model, self.model.config.d_model, bias=False
            )
            self.TableCL_loss = TableCL(temperature=0.1)

        if self.use_imgRoiAlign:
            # Set up modules for Image ROIAlignment
            self.img_downsize_scale = 32
            assert (
                self.input_size[0] == 768 and self.input_size[1] == 768
            ), "input_size must be (768, 768) when use_imgRoiAlign is True"
            self.roi_align = RoIAlign(
                output_size=(2, 2),
                spatial_scale=1 / self.img_downsize_scale,
                sampling_ratio=-1,
                aligned=False,
            )
            self.roi_proj = nn.Sequential(
                nn.Linear(self.model.config.d_model * 4, self.model.config.d_model),
                nn.ReLU(),
                nn.Linear(self.model.config.d_model, self.model.config.d_model),
            )
            self.dummy_embed = nn.Embedding(1, self.model.config.d_model)
            self.bbox_coord_merge = nn.Sequential(
                nn.Linear(self.model.config.d_model, self.model.config.d_model),
                nn.ReLU(),
                nn.Linear(self.model.config.d_model, self.model.config.d_model),
            )
            self.roi_merge = nn.Sequential(
                nn.Linear(self.model.config.d_model, self.model.config.d_model),
                nn.ReLU(),
                nn.Linear(self.model.config.d_model, self.model.config.d_model),
            )

        if self.name_or_path is None:
            raise NotImplementedError

    def _custom_prepare_decoder_attention_mask(
        self, attention_mask, input_shape, inputs_embeds, past_key_values_length
    ):
        """Modification of attention mask for decoder

        NOTE:
            - This function overrides the default `_prepare_decoder_attention_mask` function
            - Aims to modify the attention mask to allow bi-directional attention for prefix tokens corresponding to bbox
        """

        if self.bbox_token_cnt:
            prefix_dimension = self.bbox_token_cnt + 1  # add 1 for bos token position
        else:
            prefix_dimension = None

        combined_attention_mask = None
        if input_shape[-1] > 1:
            combined_attention_mask = _make_causal_mask(
                input_shape,
                inputs_embeds.dtype,
                device=inputs_embeds.device,
                past_key_values_length=past_key_values_length,
            )

            if (
                prefix_dimension is not None
                and combined_attention_mask.shape[-2] >= prefix_dimension
            ):
                assert (
                    combined_attention_mask.shape[-2]
                    == combined_attention_mask.shape[-1]
                ), "Only square attention masks are allowed"
                combined_attention_mask[:, :, :prefix_dimension, :prefix_dimension] = (
                    0  # Allow bi-directional for prefix_dimension tokens
                )

        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            expanded_attn_mask = _expand_mask(
                attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1]
            ).to(inputs_embeds.device)
            combined_attention_mask = (
                expanded_attn_mask
                if combined_attention_mask is None
                else expanded_attn_mask + combined_attention_mask
            )

        return combined_attention_mask

    def prepare_inputs_for_inference(
        self: "MBARTDecoder",
        input_ids: torch.Tensor,
        encoder_outputs: torch.Tensor,
        past_key_values: torch.Tensor = None,
        past=None,
        use_cache: bool = None,
        attention_mask: torch.Tensor = None,
        input_coords: torch.Tensor = None,
        input_coords_length: torch.Tensor = None,
    ):
        """Custom function for preparing inputs for inference

        NOTE:
            - This function overrides the default `prepare_inputs_for_generation` function
        """

        if past is not None:
            past_key_values = past
        attention_mask = input_ids.ne(self.tokenizer.pad_token_id).long()
        if past_key_values is not None:
            input_ids = input_ids[:, -1:]

        output = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "past_key_values": past_key_values,
            "use_cache": use_cache,
            "encoder_hidden_states": encoder_outputs.last_hidden_state,
            "inference_mode": True,
            "input_coords": input_coords,
            "input_coords_length": input_coords_length,
        }

        return output

    def embed_coord_tensor(self, input_coord_tensor: torch.Tensor):
        """Embed coordinate tensor"""
        assert input_coord_tensor.shape[-1] == 4
        coord_embedding = torch.cat(
            [
                self.x_coord_embedding(input_coord_tensor[..., 0]),
                self.y_coord_embedding(input_coord_tensor[..., 1]),
                self.x_coord_embedding(input_coord_tensor[..., 2]),
                self.y_coord_embedding(input_coord_tensor[..., 3]),
            ],
            dim=-1,
        )

        return coord_embedding

    def create_extended_cell_mask(self, input_ids, pointer_mask_labels):
        """Create extended cell mask for all cell tokens (C, L, X, U tags)."""
        batch_size, seq_len = input_ids.shape
        extended_cell_mask = torch.zeros_like(input_ids, dtype=torch.bool)
        
        # Get token IDs for all cell-related tags
        cell_tag_ids = []
        for tag in ['C-tag', 'L-tag', 'X-tag', 'U-tag']:
            try:
                tag_id = self.tokenizer.convert_tokens_to_ids(tag)
                if tag_id != self.tokenizer.unk_token_id:  # Only add if token exists
                    cell_tag_ids.append(tag_id)
            except:
                pass  # Skip if token doesn't exist
        
        if not cell_tag_ids:
            # Fallback: if no cell tags found, use the original pointer mask  
            return pointer_mask_labels
            
        # Create mask for all cell tokens
        for batch_idx in range(batch_size):
            for seq_idx in range(seq_len):
                if input_ids[batch_idx, seq_idx].item() in cell_tag_ids:
                    extended_cell_mask[batch_idx, seq_idx] = True
                    
        return extended_cell_mask

    def extract_ctag_positions_from_otsl(self, input_ids, pointer_mask_label, tokenizer=None):
        """Extract exact row and column positions for C-tags from OTSL sequence."""
        row_ids = []
        col_ids = []
        
        if pointer_mask_label is None:
            return [], []
        
        # If we have input_ids and tokenizer, we can find exact NL-tag positions
        if input_ids is not None and tokenizer is not None:
            try:
                # Get NL-tag token ID
                nl_tag_id = tokenizer.convert_tokens_to_ids('NL-tag')
                
                # Find all NL-tag positions (row boundaries)
                nl_positions = []
                for i, token_id in enumerate(input_ids):
                    if token_id == nl_tag_id:
                        nl_positions.append(i)
                
                if nl_positions:
                    # Parse table structure using NL-tags
                    current_row = 0
                    current_col = 0
                    last_nl = -1
                    
                    # Convert mask to list
                    if torch.is_tensor(pointer_mask_label):
                        mask_list = pointer_mask_label.cpu().tolist()
                    else:
                        mask_list = list(pointer_mask_label)
                    
                    # Process each token
                    for i, is_ctag in enumerate(mask_list):
                        # Check if we passed an NL-tag
                        while current_row < len(nl_positions) and i > nl_positions[current_row]:
                            current_row += 1
                            current_col = 0
                            last_nl = nl_positions[current_row - 1] if current_row > 0 else -1
                        
                        if is_ctag:
                            # This is a C-tag
                            # Column is determined by counting cell tokens since last NL
                            actual_col = 0
                            for j in range(last_nl + 1, i):
                                if j < len(mask_list):
                                    # Count all cell tokens (C-tags and merge tags)
                                    # In OTSL, all non-NL tags between NLs are cells
                                    if j not in nl_positions:
                                        actual_col += 1
                            
                            row_ids.append(current_row)
                            col_ids.append(actual_col)
                    
                    return row_ids, col_ids
                    
            except Exception:
                pass  # Fall back to pattern-based detection
        
        # Fallback: Pattern-based detection when we don't have token IDs
        if torch.is_tensor(pointer_mask_label):
            mask_list = pointer_mask_label.cpu().tolist()
        else:
            mask_list = list(pointer_mask_label)
        
        ctag_positions = [i for i, is_ctag in enumerate(mask_list) if is_ctag]
        
        if len(ctag_positions) == 0:
            return [], []
        
        # Detect row boundaries by gaps (NL-tags create gaps)
        rows = []
        current_row = [ctag_positions[0]]
        
        for i in range(1, len(ctag_positions)):
            gap = ctag_positions[i] - ctag_positions[i-1]
            # Gap > 1 likely means we crossed an NL-tag
            if gap > 1:
                rows.append(current_row)
                current_row = [ctag_positions[i]]
            else:
                current_row.append(ctag_positions[i])
        
        if current_row:
            rows.append(current_row)
        
        # Now assign (row, col) based on detected structure
        max_cols = max(len(row) for row in rows) if rows else 1
        
        for row_idx, row_positions in enumerate(rows):
            for col_idx, pos in enumerate(row_positions):
                row_ids.append(row_idx)
                col_ids.append(col_idx)
        
        return row_ids, col_ids

    def add_grid_positional_encoding(self, text_embeddings, input_ids, extended_cell_mask):
        """Add grid positional encoding to cell tokens (C, L, X, U tags)."""
        if not self.use_row_col_embedding:
            return text_embeddings
            
        batch_size = text_embeddings.shape[0]
        device = text_embeddings.device
        
        for batch_idx in range(batch_size):
            # Extract grid positions for current batch
            row_ids, col_ids = self.extract_ctag_positions_from_otsl(
                input_ids[batch_idx] if input_ids is not None else None,
                extended_cell_mask[batch_idx],  # Use extended mask for all cell tokens
                self.tokenizer
            )
            
            # Find positions of cell tokens in the sequence
            cell_indices = torch.where(extended_cell_mask[batch_idx])[0]
            
            # Apply positional encoding to each cell token
            encoding_type = self.row_col_embedding_config.get('encoding_type', 'learned')
            
            if encoding_type == 'sinusoidal':
                # Use 2D Sinusoidal Encoding
                if len(row_ids) > 0 and len(col_ids) > 0:
                    # Convert to tensors
                    row_tensor = torch.tensor(row_ids, device=device)
                    col_tensor = torch.tensor(col_ids, device=device)
                    
                    # Get 2D sinusoidal encodings
                    pos_encodings = self.grid_2d_encoding(row_tensor, col_tensor)  # (num_cells, d_model)
                    
                    # Add to cell token embeddings
                    for i, pos_idx in enumerate(cell_indices):
                        if i < len(pos_encodings):
                            text_embeddings[batch_idx, pos_idx] += pos_encodings[i]
            else:
                # Use learned embeddings (original behavior)
                for i, pos_idx in enumerate(cell_indices):
                    if i < len(row_ids) and i < len(col_ids):
                        # Clamp positions to valid embedding range
                        row = min(row_ids[i], self.row_embedding.num_embeddings - 1)
                        col = min(col_ids[i], self.col_embedding.num_embeddings - 1)
                        
                        # Get embeddings for this position
                        row_emb = self.row_embedding(torch.tensor(row, device=device))
                        col_emb = self.col_embedding(torch.tensor(col, device=device))
                        
                        # Project to final dimension and add to text embedding
                        pos_emb = self.pos_projection(torch.cat([row_emb, col_emb]))
                        text_embeddings[batch_idx, pos_idx] += pos_emb
                    
        return text_embeddings

    def get_img_roiAlign(self, encoder_hidden_states, quad_input_coords):
        """
        Get Image ROIAlign based on input coordinates

        Args:
            encoder_hidden_states: (bsz, embed_h * embed_w, d_model)
            quad_input_coords: (bsz, bbox_token_length, 4)
        """
        # convert coords to roialign
        org_dtype = encoder_hidden_states.dtype
        img_idx_tensor = (
            torch.arange(
                encoder_hidden_states.shape[0], device=encoder_hidden_states.device
            )
            .unsqueeze(-1)
            .unsqueeze(-1)
        )  # (bsz, 1, 1)
        img_idx_tensor = img_idx_tensor.repeat(
            1, quad_input_coords.shape[1], 1
        )  # (bsz, bbox_token_length, 1)
        input_coord_with_idx = torch.cat(
            [img_idx_tensor, quad_input_coords], dim=-1
        )  # (bsz, bbox_token_length, 5)
        input_coord_with_idx = input_coord_with_idx.to(
            torch.float
        )  # (bsz, bbox_token_length, 5)
        bsz, bbox_token_cnt, _ = input_coord_with_idx.shape
        rois = input_coord_with_idx.view(-1, 5)  # (bsz * bbox_token_cnt, 5)

        # encoder_hidden_states (bsz, embed_h * embed_w, d_model) -> (bsz, d_model, embed_h, embed_w)
        embed_dim_h = int(self.input_size[1] / self.img_downsize_scale)
        embed_dim_w = int(self.input_size[0] / self.img_downsize_scale)
        feature_map = encoder_hidden_states.transpose(1, 2).view(
            bsz, encoder_hidden_states.shape[-1], embed_dim_h, embed_dim_w
        )  # (bsz, d_model, embed_h, embed_w)

        # typecast feature_map & rois to fp32
        pooled_features = self.roi_align(
            feature_map.to(torch.float), rois
        )  # (bsz * bbox_token_cnt, d_model, 2, 2)
        pooled_features = pooled_features.view(
            pooled_features.shape[0], -1
        )  # (bsz * bbox_token_cnt, d_model * 2 * 2)
        pooled_features = self.roi_proj(
            pooled_features.to(org_dtype)
        )  # (bsz * bbox_token_cnt, d_model)
        pooled_features = pooled_features.view(
            bsz, bbox_token_cnt, -1
        )  # (bsz, bbox_token_cnt, d_model)

        return pooled_features

    def forward(
        self: "MBARTDecoder",
        input_ids: torch.Tensor = None,
        input_coords: torch.Tensor = None,
        input_coords_length: torch.Tensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        past_key_values: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        pointer_labels: Optional[torch.Tensor] = None,
        pointer_mask_labels: Optional[torch.Tensor] = None,
        bbox_coeff_tensor: Optional[torch.Tensor] = None,
        use_cache: bool = None,
        output_attentions: Optional[torch.Tensor] = None,
        output_hidden_states: Optional[torch.Tensor] = None,
        return_dict: bool = None,
        inference_mode: bool = False,
    ):
        """
        input_ids shape:            (batch_size, text_token_length)
        input_coords shape:         (batch_size, bbox_token_length, 4)
        input_coords_length shape:  (batch_size,)
        pointer_labels shape:       (batch_size, text_token_length - 2, bbox_token_length)
        pointer_mask_labels shape:  (batch_size, text_token_length - 2)
        bbox_coeff_tensor:          (batch_size, 5, bbox_token_length, bbox_token_length)
        """
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.model.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.model.config.output_hidden_states
        )
        return_dict = (
            return_dict
            if return_dict is not None
            else self.model.config.use_return_dict
        )

        if inference_mode:
            assert labels is None
        (
            loss,
            tag2coord_pointer_acc,
            token_cls_loss,
            tag2coord_pointer_loss,
            bbox_TableCL_loss,
        ) = (None, None, None, None, None)
        (
            rowwise_loss,
            colwise_loss,
        ) = (None, None)

        batch_size = input_ids.shape[0]
        # Separate handling for inference mode
        total_bbox_token_length = input_coords.shape[1]
        
        # Initialize position tracker at the start of inference
        if inference_mode and self.use_row_col_embedding:
            if past_key_values is None:
                # Beginning of generation, reset all position trackers
                self.position_trackers = {}
        
        if inference_mode and past_key_values is not None:
            input_embeds = (
                self.model.model.decoder.embed_tokens(input_ids)
                * self.model.model.decoder.embed_scale
            )  # (batch_size, text_token_length, d_model)
            
            # Apply grid positional encoding for inference if enabled
            if self.use_row_col_embedding and input_ids.shape[1] == 1:
                # Process each batch separately
                for batch_idx in range(input_ids.shape[0]):
                    # Get or create position tracker for this batch
                    if batch_idx not in self.position_trackers:
                        self.position_trackers[batch_idx] = GridPositionTracker()
                    
                    # During generation, input_ids contains only the last generated token
                    last_token = input_ids[batch_idx, -1]
                    position = self.position_trackers[batch_idx].update_and_get_position(last_token, self.tokenizer)
                    
                    if position is not None:
                        # This is a cell token, apply positional encoding
                        row, col = position
                        device = input_embeds.device
                        
                        encoding_type = self.row_col_embedding_config.get('encoding_type', 'learned')
                        
                        if encoding_type == 'sinusoidal':
                            # Use 2D Sinusoidal Encoding
                            pos_emb = self.grid_2d_encoding.get_encoding_for_position(row, col).to(device)
                            input_embeds[batch_idx, -1] += pos_emb
                        else:
                            # Use learned embeddings (original behavior)
                            # Clamp positions to valid range
                            row = min(row, self.row_embedding.num_embeddings - 1)
                            col = min(col, self.col_embedding.num_embeddings - 1)
                            
                            # Get embeddings
                            row_emb = self.row_embedding(torch.tensor(row, device=device))
                            col_emb = self.col_embedding(torch.tensor(col, device=device))
                            
                            # Apply positional encoding to the last token embedding
                            pos_emb = self.pos_projection(torch.cat([row_emb, col_emb]))
                            input_embeds[batch_idx, -1] += pos_emb
        else:
            if self.use_imgRoiAlign:
                # Get Image ROIAlign
                img_ROIAlign = self.get_img_roiAlign(
                    encoder_hidden_states, input_coords
                )  # (batch_size, bbox_token_length, d_model)
                # A) Remove the last bbox_token_length entry from encoder_hidden_states
                img_ROIAlign = img_ROIAlign[
                    :, :-1
                ]  # (batch_size, bbox_token_length - 1, d_model)
                # B) Concat dummy bbox_token_length entry at the start
                dummy_ROIAlign = self.dummy_embed.weight.unsqueeze(0).repeat(
                    batch_size, 1, 1
                )  # (batch_size, 1, d_model)
                img_ROIAlign = torch.cat(
                    [dummy_ROIAlign, img_ROIAlign], dim=1
                )  # (batch_size, bbox_token_length, d_model)

            # Set up coordinate embedding of input coords along with addition of dummy coordinates
            batch_coord_embedding = self.embed_coord_tensor(
                input_coords[:, :-1]
            )  # (batch_size, bbox_token_length - 1, d_model)
            ## Concat dummy coordinate embedding at the start -> for cell data that has no corresponding dr_coord
            dummy_coord_embedding = torch.tensor(
                [
                    self.input_size[0] + 1,
                    self.input_size[1] + 1,
                    self.input_size[0] + 2,
                    self.input_size[1] + 2,
                ],
                dtype=input_coords.dtype,
                device=input_coords.device,
            )
            dummy_coord_embedding = self.embed_coord_tensor(
                dummy_coord_embedding.unsqueeze(0).unsqueeze(0)
            )  # (1, 1, d_model)
            dummy_coord_embedding = dummy_coord_embedding.repeat(
                batch_size, 1, 1
            )  # (batch_size, 1, d_model)
            batch_coord_embedding = torch.cat(
                [dummy_coord_embedding, batch_coord_embedding], dim=1
            )  # (batch_size, bbox_token_length, d_model)

            if self.use_imgRoiAlign:
                # Add Image ROIAlignment with coordinate embedding
                batch_coord_embedding = self.bbox_coord_merge(
                    batch_coord_embedding
                ) + self.roi_merge(
                    img_ROIAlign
                )  # (batch_size, bbox_token_length, d_model)

            # Text Embedding
            batch_text_embedding = (
                self.model.model.decoder.embed_tokens(input_ids)
                * self.model.model.decoder.embed_scale
            )  # (batch_size, text_token_length, d_model)
            
            # Apply grid positional encoding if enabled
            if self.use_row_col_embedding and pointer_mask_labels is not None:
                # Create extended cell mask for C, L, X, U tags
                extended_cell_mask = self.create_extended_cell_mask(input_ids, pointer_mask_labels)
                batch_text_embedding = self.add_grid_positional_encoding(
                    batch_text_embedding, input_ids, extended_cell_mask
                )

            # Combine all embeddings for input
            input_embeds = torch.cat(
                [
                    batch_text_embedding[:, 0:1],
                    batch_coord_embedding,
                    batch_text_embedding[:, 1:],
                ],
                dim=1,
            )  # (batch_size, max_seq, d_model)

            # Update labels
            if labels is not None:
                ignore_label = (
                    torch.zeros(
                        (batch_size, total_bbox_token_length),
                        dtype=labels.dtype,
                        device=labels.device,
                    )
                    - 100
                )
                labels = torch.cat([labels[:, 0:1], ignore_label, labels[:, 1:]], dim=1)

        if not inference_mode:
            input_embeds = input_embeds[:, :-1]  # (batch_size, max_seq-1, d_model)

        if labels is not None and not inference_mode:
            labels = labels[:, 1:]  # (batch_size, max_seq-1)

        # Derive attention mask for decoder that ignores padding bbox tokens
        if inference_mode:
            # from (bsz, 2) -> (bsz, 2 + total_bbox_token_length) 2 -> bos & s_start
            # attention_mask = torch.ones((attention_mask.shape[0], attention_mask.shape[1] + total_bbox_token_length), dtype=attention_mask.dtype, device=attention_mask.device)
            tmp_range_mask = torch.arange(
                attention_mask.shape[1] + total_bbox_token_length,
                device=attention_mask.device,
            ).unsqueeze(
            0
            )  # (1, 2 + total_bbox_token_length)
            valid_bbox_mask = tmp_range_mask <= (
                input_coords_length.unsqueeze(1) + 1
            )  # (bsz, 2 + total_bbox_token_length)
            non_bbox_valid_mask = tmp_range_mask >= (total_bbox_token_length + 1)
            attention_mask = torch.logical_or(valid_bbox_mask, non_bbox_valid_mask).to(
                attention_mask.dtype
            )  # (bsz, 2 + total_bbox_token_length)
        else:
            # NOTE: input_embeds is of shape (bsz, max_seq_length-1, d_model)
            tmp_range_mask = torch.arange(
                input_embeds.shape[1], device=input_embeds.device
            ).unsqueeze(
                0
            )  # (1, max_seq_length-1)
            valid_bbox_mask = tmp_range_mask <= (
                input_coords_length.unsqueeze(1) + 1
            )  # (bsz, max_seq_length-1)
            non_bbox_valid_mask = tmp_range_mask >= (total_bbox_token_length + 1)
            attention_mask = torch.logical_or(
                valid_bbox_mask, non_bbox_valid_mask
            ).long()  # (bsz, max_seq_length-1)

        outputs = self.model.model.decoder(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        logits = self.model.lm_head(outputs[0])  # (batch_size, max_seq-1, vocab_size)

        if labels is not None:
            # token classification loss
            token_cls_loss = self.get_token_classification_loss(logits, labels)

            tag2coord_pointer_loss = 0
            tag2coord_pointer_acc = 0
            bbox_TableCL_loss = 0
            (
                rowwise_loss,
                colwise_loss,
            ) = (0, 0)

            # Calculate pointer loss per data instance due to OOM
            sub_batchsize = 4
            num_sub_batches = batch_size // sub_batchsize
            if batch_size % sub_batchsize > 0:
                num_sub_batches += 1

            for sub_batch_i in range(num_sub_batches):
                # tag-to-coord pointer loss
                start_index = sub_batch_i * sub_batchsize
                end_index = (sub_batch_i + 1) * sub_batchsize
                curr_tag2coord_ptr_loss = self.get_tag2coord_ptr_loss(
                    output_seq=outputs[0][start_index:end_index],
                    total_bbox_token_length=total_bbox_token_length,
                    input_coords_length=input_coords_length[start_index:end_index],
                    pointer_label=pointer_labels[start_index:end_index],
                    pointer_mask_label=pointer_mask_labels[start_index:end_index],
                    use_adjacent_penalty=self.use_adjacent_penalty,
                    adjacent_penalty_config=self.adjacent_penalty_config,
                )

                if (
                    batch_size % sub_batchsize > 0
                    and sub_batch_i == num_sub_batches - 1
                ):
                    curr_batch_size = batch_size % sub_batchsize
                else:
                    curr_batch_size = sub_batchsize

                is_empty_loss, is_not_empty_loss, ptr_accuracy = curr_tag2coord_ptr_loss
                tag2coord_pointer_loss += (
                    (self.empty_cell_ptr_loss_coeff * is_empty_loss)
                    + (self.non_empty_cell_ptr_loss_coeff * is_not_empty_loss)
                ) * curr_batch_size
                tag2coord_pointer_acc += ptr_accuracy * curr_batch_size

                # HiMulConET loss
                if self.use_bbox_HiMulConET:
                    curr_bbox_TableCL_loss = self.get_bbox_TableCL_loss(
                        bbox_coeff_tensor=bbox_coeff_tensor[start_index:end_index],
                        output_seq=outputs[0][start_index:end_index],
                        total_bbox_token_length=total_bbox_token_length,
                        input_coords_length=input_coords_length[start_index:end_index],
                        contr_learning_config=self.contrastive_loss_config,
                    )

                    (
                        curr_rowwise_loss,
                        curr_colwise_loss,
                    ) = curr_bbox_TableCL_loss
                    curr_bbox_TableCL_loss = (
                        curr_rowwise_loss + curr_colwise_loss
                    ) * curr_batch_size
                    curr_bbox_TableCL_loss /= sum(self.contrastive_loss_config.values())
                    bbox_TableCL_loss += curr_bbox_TableCL_loss

                    rowwise_loss += curr_rowwise_loss * curr_batch_size
                    colwise_loss += curr_colwise_loss * curr_batch_size

            # Consolidate loss values
            tag2coord_pointer_acc /= batch_size
            tag2coord_pointer_loss /= batch_size
            loss = token_cls_loss + tag2coord_pointer_loss

            if self.use_bbox_HiMulConET:
                rowwise_loss /= batch_size  # For Logging purpose
                colwise_loss /= batch_size  # For Logging purpose

                bbox_TableCL_loss /= batch_size
                if loss is None:
                    loss = bbox_TableCL_loss
                else:
                    loss += bbox_TableCL_loss

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return ModelOutput(
            loss=loss,
            token_cls_loss=token_cls_loss,
            tag2coord_pointer_loss=tag2coord_pointer_loss,
            tag2coord_pointer_acc=tag2coord_pointer_acc,
            bbox_TableCL_loss=bbox_TableCL_loss,
            rowwise_loss=rowwise_loss,
            colwise_loss=colwise_loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            decoder_hidden_states=outputs.hidden_states,
            decoder_attentions=outputs.attentions,
            cross_attentions=outputs.cross_attentions,
        )

    def add_special_tokens(self: "MBARTDecoder", list_of_tokens: List[str]):
        """Add special tokens to tokenizer and resize token embeddings"""
        newly_added_num = self.tokenizer.add_special_tokens(
            {"additional_special_tokens": sorted(set(list_of_tokens))}
        )
        if newly_added_num > 0:
            self.model.resize_token_embeddings(len(self.tokenizer))

        return newly_added_num

    def get_token_ids_to_token(self: "MBARTDecoder"):
        """Get token_id to token and token to token_id mapping"""
        token_id_to_token = {}
        token_to_token_id = {}
        spec_chars = [
            "<tr>",
            "</tr>",
            "<td>",
            "</td>",
            "<thead>",
            "</thead>",
            "<tbody>",
            "</tbody>",
            'rowspan="',
            'colspan="',
        ]
        for c in spec_chars:
            token_id_to_token[self.tokenizer.convert_tokens_to_ids(c)] = c
            token_to_token_id[c] = self.tokenizer.convert_tokens_to_ids(c)

        self.token_id_to_token = token_id_to_token
        self.token_to_token_id = token_to_token_id

    def get_token_classification_loss(self, logits, labels, ignore_idx=-100):
        """Get token classification loss

        Args:
            logits: (batch_size, max_seq-1, vocab_size)
            labels: (batch_size, max_seq-1)
            ignore_idx: ignore index for loss calculation
        """
        loss_func = nn.CrossEntropyLoss(ignore_index=ignore_idx)
        token_cls_loss = loss_func(
            logits.reshape(-1, self.model.config.vocab_size), labels.reshape(-1)
        )

        return token_cls_loss

    def get_tag2coord_ptr_loss(
        self,
        output_seq,
        total_bbox_token_length,
        input_coords_length,
        pointer_label,
        pointer_mask_label,
        use_adjacent_penalty=False,
        adjacent_penalty_config=None,
    ):
        """Function to calculate tag2coord pointer loss

        Args:
            output_seq: (bsz, max_seq, d_model)
            total_bbox_token_length: total number of bbox tokens in the input sequence
            input_coords_length: number of bbox tokens in the input sequence
            pointer_label: (bsz, total_text_token_length - 2, total_bbox_token_length)
            pointer_mask_label: (bsz, total_text_token_length - 2)

        Note:
            Tag2Coord refers to the pointer network that points from text tokens to bbox tokens
        """
        # 1. calculate pointing probability
        # input_seq ->  <s><bbox1><bbox2>...<bboxN><s_start><thead><tr>....
        # output_seq -> <bbox1><bbox2>...<bboxN><s_start><thead><tr>....
        # Shape of output_seq is (bsz, max_seq, d_model)
        assert len(output_seq.shape) == 3, "output_seq must be (bsz, max_seq, d_model)"
        batch_size = output_seq.shape[0]
        key_feature = self.k_linear(
            output_seq[:, :total_bbox_token_length]
        )  # (bsz, total_bbox_token_length, d_model)
        query_feature = self.q_linear(
            output_seq[:, total_bbox_token_length + 1 :]
        )  # (bsz, total_text_token_length - 2, d_model)

        normalized_key_feature = F.normalize(
            key_feature, dim=-1
        )  # (bsz, total_bbox_token_length, d_model)
        normalized_query_feature = F.normalize(
            query_feature, dim=-1
        )  # (bsz, total_text_token_length - 2, d_model)
        data_combined_feat = torch.bmm(
            normalized_query_feature, normalized_key_feature.transpose(1, 2)
        )  # (bsz, total_text_token_length - 2, total_bbox_token_length)

        # 2. calculate loss
        if pointer_label.dtype != query_feature.dtype:
            pointer_label = pointer_label.to(query_feature.dtype)

        # First, extract out all is-data text tokens first
        # pointer_mask_label -> whether each token is data tag or not (e.g. OTSL -> C-tag)
        temperature = 0.1
        # data_combined_feat shape: (bsz, total_text_token_length - 2, total_bbox_token_length)
        # pointer_label shape: (bsz, total_text_token_length - 2, total_bbox_token_length)
        # pointer_mask_label shape: (bsz, total_text_token_length - 2)
        is_empty_loss = 0
        is_not_empty_loss = 0

        batchwise_pointing_acc = []
        for data_i in range(batch_size):
            is_data_only_pred = data_combined_feat[
                data_i, pointer_mask_label[data_i]
            ]  # (num_is_data_text_tokens, total_bbox_token_length)
            is_data_only_label = pointer_label[
                data_i, pointer_mask_label[data_i]
            ]  # (num_is_data_text_tokens, total_bbox_token_length)

            is_empty_loss += nn.BCEWithLogitsLoss()(
                is_data_only_pred[:, 0], is_data_only_label[:, 0]
            )

            is_not_empty_pred = is_data_only_pred[
                :, 1 : (input_coords_length[data_i] + 1)
            ]  # (num_is_data_text_tokens, input_coords_length)
            is_not_empty_label = is_data_only_label[
                :, 1 : (input_coords_length[data_i] + 1)
            ]  # (num_is_data_text_tokens, input_coords_length)
            valid_coords_tmp = (
                torch.sum(is_not_empty_label, 0) == 1
            )  # NOTE: While each data text token could correspond to multiple bbox tokens, each bbox token can only correspond to one data text token

            # Apply adjacent penalty if enabled
            if use_adjacent_penalty and integrate_weighted_ce_into_decoder_v2 is not None:
                # Use v2 with spatial weights and mass normalization
                
                # Extract row and column positions for each C-tag from OTSL
                row_ids, col_ids = self.extract_ctag_positions_from_otsl(
                    None,  # input_ids not available in this scope
                    pointer_mask_label[data_i],  # Current batch's mask
                    self.tokenizer
                )
                
                # Ensure we have positions for all candidates
                num_candidates = is_not_empty_pred.shape[0]
                if len(row_ids) < num_candidates:
                    # Fallback: extend with grid positions if needed
                    grid_cols = int(num_candidates ** 0.5) + 1
                    for i in range(len(row_ids), num_candidates):
                        row_ids.append(i // grid_cols)
                        col_ids.append(i % grid_cols)
                
                # Prepare config with spatial information
                config = adjacent_penalty_config.copy() if adjacent_penalty_config else {}
                
                # Add dynamic spatial information
                config.update({
                    'row_id': row_ids,
                    'col_id': col_ids,
                })
                
                # Use temperature from config, fallback to hardcoded value if not present
                if 'temperature' not in config:
                    config['temperature'] = temperature
                
                is_not_empty_loss += integrate_weighted_ce_into_decoder_v2(
                    is_not_empty_pred,
                    is_not_empty_label,
                    valid_coords_tmp,
                    use_adjacent_penalty=True,
                    adjacent_penalty_config=config
                )
            else:
                # Original CrossEntropy without penalties
                is_not_empty_pred = is_not_empty_pred / temperature  # Apply temperature for standard CE
                is_not_empty_loss += nn.CrossEntropyLoss()(
                    torch.transpose(is_not_empty_pred, 0, 1)[valid_coords_tmp],
                    torch.argmax(
                        torch.transpose(is_not_empty_label, 0, 1)[valid_coords_tmp],
                        dim=-1,
                    ),
                )

            with torch.no_grad():
                # is_not_empty_pred shape: (num_is_data_text_tokens, input_coords_length)
                pred_pointing = F.one_hot(
                    torch.argmax(is_not_empty_pred, dim=0),
                    num_classes=is_not_empty_pred.shape[0],
                ).transpose(
                    0, 1
                )  # (num_is_data_text_tokens, input_coords_length)
                pred_pointing = pred_pointing[:, valid_coords_tmp]

                gold_pointing = (
                    is_not_empty_label  # (num_is_data_text_tokens, input_coords_length)
                )
                gold_pointing = gold_pointing[:, valid_coords_tmp]

                equiv_tns = (
                    pred_pointing == gold_pointing
                )  # (num_is_data_text_tokens, input_coords_length)

                token_wise_equivalence = torch.sum(equiv_tns, dim=-1) == torch.sum(
                    valid_coords_tmp
                )  # (num_is_data_text_tokens)
                batchwise_pointing_acc.append(
                    torch.sum(token_wise_equivalence).float()
                    / token_wise_equivalence.shape[0]
                )

        is_not_empty_loss = is_not_empty_loss / batch_size
        is_empty_loss = is_empty_loss / batch_size
        pointing_acc = torch.mean(torch.stack(batchwise_pointing_acc, dim=0))

        return is_empty_loss, is_not_empty_loss, pointing_acc

    def get_bbox_TableCL_loss(
        self,
        bbox_coeff_tensor,
        output_seq,
        total_bbox_token_length,
        input_coords_length,
        contr_learning_config,
    ):
        """Function to calculate Contrastive Learning loss for bbox tokens

        Args:
            bbox_coeff_tensor: (batch_size, 5, bbox_token_length, bbox_token_length)
            output_seq: (batch_size, max_seq, d_model)
            total_bbox_token_length: total number of bbox tokens in the input sequence
            input_coords_length: number of bbox tokens in the input sequence
            contr_learning_config: configuration for contrastive learning
        """
        (
            rowwise_loss,
            colwise_loss,
        ) = (0, 0)
        bbox_feature_output = output_seq[
            :, :total_bbox_token_length
        ]  # (batch_size, total_bbox_token_length, d_model)

        if contr_learning_config["use_RowWise_contLearning"]:
            rowwise_feature = self.rowwise_linear(bbox_feature_output)
            rowwise_feature = F.normalize(rowwise_feature, dim=-1)
            coeff_index = (
                sum(
                    [
                        contr_learning_config["use_RowWise_contLearning"],
                    ]
                )
                - 1
            )
            rowwise_mask = bbox_coeff_tensor[:, coeff_index : (coeff_index + 1)]
            rowwise_loss = self.TableCL_loss(
                features=rowwise_feature,
                masks=rowwise_mask,
                input_coords_length=input_coords_length,
            )

        if contr_learning_config["use_ColWise_contLearning"]:
            colwise_feature = self.colwise_linear(bbox_feature_output)
            colwise_feature = F.normalize(colwise_feature, dim=-1)
            coeff_index = (
                sum(
                    [
                        contr_learning_config["use_RowWise_contLearning"],
                        contr_learning_config["use_ColWise_contLearning"],
                    ]
                )
                - 1
            )
            colwise_mask = bbox_coeff_tensor[:, coeff_index : (coeff_index + 1)]
            colwise_loss = self.TableCL_loss(
                features=colwise_feature,
                masks=colwise_mask,
                input_coords_length=input_coords_length,
            )

        return (
            rowwise_loss,
            colwise_loss,
        )

    @staticmethod
    def resize_bart_abs_pos_emb(weight: torch.Tensor, max_length: int) -> torch.Tensor:
        """
        Resize position embeddings
        Truncate if sequence length of Bart backbone is greater than given max_length,
        else interpolate to max_length
        """
        if weight.shape[0] > max_length:
            weight = weight[:max_length, ...]
        else:
            weight = (
                F.interpolate(
                    weight.permute(1, 0).unsqueeze(0),
                    size=max_length,
                    mode="linear",
                    align_corners=False,
                )
                .squeeze(0)
                .permute(1, 0)
            )
        return weight