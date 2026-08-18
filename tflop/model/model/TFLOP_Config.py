import os
from typing import Tuple, Union

import omegaconf
from transformers import PretrainedConfig


class TFLOPConfig(PretrainedConfig):
    model_type = "tflop"

    def _convert_nested_dictconfig(self, d):
        """Recursively convert any nested DictConfig to regular dict"""
        try:
            from omegaconf import DictConfig, OmegaConf
            if isinstance(d, DictConfig):
                return OmegaConf.to_container(d, resolve=True, enum_to_str=True)
            elif isinstance(d, dict):
                return {k: self._convert_nested_dictconfig(v) if isinstance(v, (dict, DictConfig)) else v 
                        for k, v in d.items()}
            else:
                return d
        except ImportError:
            return d if isinstance(d, dict) else {}
    
    def __init__(
        self: "TFLOPConfig",
        input_size: dict = {"height": 1280, "width": 960},
        align_along_axis: bool = False,
        window_size: int = 10,
        encoder_layer: Tuple[int] = (2, 2, 14, 2),
        decoder_layer: int = 4,
        max_position_embeddings: int = None,
        max_length: int = 768,
        name_or_path: Union[str, bytes, os.PathLike] = "",
        use_fast_decoder: bool = False,
        use_ptr_decoder: bool = False,
        bbox_token_cnt: int = None,
        use_cell_bbox: bool = False,
        max_num_row: int = 40,
        max_num_col: int = 40,
        use_bbox_HiMulConET: bool = False,
        use_imgRoiAlign: bool = False,
        use_RowWise_contLearning: bool = False,
        use_ColWise_contLearning: bool = False,
        empty_cell_ptr_loss_coeff: float = 0.5,
        non_empty_cell_ptr_loss_coeff: float = 0.5,
        use_adjacent_penalty: bool = False,
        adjacent_penalty_config: dict = None,
        use_row_col_embedding: bool = False,
        row_col_embedding_config: dict = None,
        **kwargs,
    ):
        super().__init__()

        if type(input_size) in [dict, omegaconf.dictconfig.DictConfig]:
            self.input_size = (
                input_size["width"],
                input_size["height"],
            )  # Set to default (width, height)
        else:
            self.input_size = input_size
        self.align_along_axis = align_along_axis
        self.window_size = window_size
        self.encoder_layer = encoder_layer
        self.decoder_layer = decoder_layer
        self.max_position_embeddings = (
            max_length if max_position_embeddings is None else max_position_embeddings
        )
        self.max_length = max_length
        self.name_or_path = name_or_path
        self.use_fast_decoder = use_fast_decoder
        self.use_ptr_decoder = use_ptr_decoder
        self.bbox_token_cnt = bbox_token_cnt
        self.use_cell_bbox = use_cell_bbox
        self.max_num_row = max_num_row
        self.max_num_col = max_num_col
        self.use_bbox_HiMulConET = use_bbox_HiMulConET
        self.use_imgRoiAlign = use_imgRoiAlign
        self.use_RowWise_contLearning = use_RowWise_contLearning
        self.use_ColWise_contLearning = use_ColWise_contLearning
        self.empty_cell_ptr_loss_coeff = empty_cell_ptr_loss_coeff
        self.non_empty_cell_ptr_loss_coeff = non_empty_cell_ptr_loss_coeff
        self.use_adjacent_penalty = use_adjacent_penalty
        # Convert DictConfig to regular dict for JSON serialization
        if adjacent_penalty_config is not None:
            # Force conversion from OmegaConf DictConfig to regular dict
            try:
                from omegaconf import OmegaConf, DictConfig
                if isinstance(adjacent_penalty_config, DictConfig):
                    # Ensure complete conversion including nested DictConfigs
                    self.adjacent_penalty_config = OmegaConf.to_container(adjacent_penalty_config, resolve=True, enum_to_str=True)
                elif isinstance(adjacent_penalty_config, dict):
                    # Convert any nested DictConfigs
                    self.adjacent_penalty_config = self._convert_nested_dictconfig(adjacent_penalty_config)
                else:
                    self.adjacent_penalty_config = {}
            except ImportError:
                self.adjacent_penalty_config = adjacent_penalty_config if isinstance(adjacent_penalty_config, dict) else {}
        else:
            self.adjacent_penalty_config = {}
        
        self.use_row_col_embedding = use_row_col_embedding
        # Convert DictConfig to regular dict for JSON serialization
        if row_col_embedding_config is not None:
            # Force conversion from OmegaConf DictConfig to regular dict
            try:
                from omegaconf import OmegaConf, DictConfig
                if isinstance(row_col_embedding_config, DictConfig):
                    # Ensure complete conversion including nested DictConfigs
                    self.row_col_embedding_config = OmegaConf.to_container(row_col_embedding_config, resolve=True, enum_to_str=True)
                elif isinstance(row_col_embedding_config, dict):
                    # Convert any nested DictConfigs
                    self.row_col_embedding_config = self._convert_nested_dictconfig(row_col_embedding_config)
                else:
                    self.row_col_embedding_config = {}
            except ImportError:
                self.row_col_embedding_config = row_col_embedding_config if isinstance(row_col_embedding_config, dict) else {}
        else:
            self.row_col_embedding_config = {
                'row_embedding_dim': 128,
                'col_embedding_dim': 128
            }

    @classmethod
    def get_member_variables(cls):
        return [
            "input_size",
            "align_along_axis",
            "window_size",
            "encoder_layer",
            "decoder_layer",
            "max_position_embeddings",
            "max_length",
            "name_or_path",
            "use_fast_decoder",
            "use_ptr_decoder",
            "bbox_token_cnt",
            "use_cell_bbox",
            "max_num_row",
            "max_num_col",
            "use_bbox_HiMulConET",
            "use_imgRoiAlign",
            "use_RowWise_contLearning",
            "use_ColWise_contLearning",
            "empty_cell_ptr_loss_coeff",
            "non_empty_cell_ptr_loss_coeff",
            "use_adjacent_penalty",
            "adjacent_penalty_config",
            "use_row_col_embedding",
            "row_col_embedding_config",
        ]
