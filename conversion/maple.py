from __future__ import annotations

from typing import Iterable, TYPE_CHECKING

import torch
from torch import Tensor

from .base import ModelBase, TextModel, gguf, logger


@ModelBase.register("MapleForCausalLM")
class MapleModel(TextModel):
    model_arch = gguf.MODEL_ARCH.MAPLE

    def set_gguf_parameters(self):
        super().set_gguf_parameters()
        h = self.hparams

        # Maple-specific: expert FFN width, 3:1 SWA-512:GA hybrid, partial RoPE 64/128
        if (moe_intermediate_size := h.get("moe_intermediate_size")) is not None:
            self.gguf_writer.add_expert_feed_forward_length(int(moe_intermediate_size))
            logger.info(f"gguf: expert feed forward length = {moe_intermediate_size}")

        self.gguf_writer.add_sliding_window(512)
        # 3:1 SWA:GA - GA layers have NO RoPE (explicit 0; loader default is full head dim),
        # SWA layers use partial RoPE 64/128 with theta 10000.
        self.gguf_writer.add_rope_dimension_count(0)
        self.gguf_writer.add_rope_dimension_count_swa(64)
        self.gguf_writer.add_rope_freq_base_swa(float(h.get("rope_theta", 10000.0)))

    _experts: list[dict[str, Tensor]] | None = None

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        # Maple stores per-expert projections (mlp.experts.{E}.{down,gate,up}_proj.weight);
        # GGUF aggregates them into one 3D tensor per projection (like Qwen2Moe).
        if name.find("experts") != -1:
            n_experts = int(self.hparams.get("num_experts", 256))
            assert bid is not None

            if self._experts is None:
                self._experts = [{} for _ in range(self.block_count)]

            self._experts[bid][name] = data_torch

            if len(self._experts[bid]) >= n_experts * 3:
                for w_name in ["down_proj", "gate_proj", "up_proj"]:
                    datas: list[Tensor] = []
                    for xid in range(n_experts):
                        ename = f"model.layers.{bid}.mlp.experts.{xid}.{w_name}.weight"
                        datas.append(self._experts[bid][ename])
                        del self._experts[bid][ename]
                    data_torch = torch.stack(datas, dim=0)
                    merged_name = f"model.layers.{bid}.mlp.experts.{w_name}.weight"
                    yield from super().modify_tensors(data_torch, merged_name, bid)
                return
            else:
                return

        yield from super().modify_tensors(data_torch, name, bid)

    def prepare_tensors(self):
        super().prepare_tensors()
        if self._experts is not None:
            experts = [k for d in self._experts for k in d.keys()]
            if len(experts) > 0:
                raise ValueError(f"Unprocessed experts: {experts}")
