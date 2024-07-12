import torch
import numpy as np
from typing import List, Union
from slar.base import SineLayer
from slar.transform import partial_xform_vis
from photonlib import AABox
from sirent import SirenT


class SirenTVis(SirenT):
    def __init__(
        self,
        cfg: dict,
        meta=None
        ):
        self.config_model = cfg["model"]

        # Initialize SirenT
        super().__init__(**self.config_model["network"])

        out_features = self.config_model["network"]["out_features"]
        self._n_outs = (
            sum(out_features) if isinstance(out_features, list) else out_features
        )

        if self.config_model.get("ckpt_file"):
            filepath = self.config_model.get("ckpt_file")
            print("[SirenTVis] loading model_dict from checkpoint", filepath)
            with open(filepath, "rb") as f:
                model_dict = torch.load(f, map_location="cpu")
                self.load_model_dict(model_dict)
            return

        # create meta
        if meta is not None:
            self._meta = meta
        elif "photonlib" in cfg:
            self._meta = AABox.load(cfg["photonlib"]["filepath"])

        # transform functions
        self.config_xform = cfg.get("transform_vis")
        self._xform_vis, self._inv_xform_vis = partial_xform_vis(self.config_xform)

        # extensions for visibility model
        self._init_output_scale(self.config_model)
        self._do_hardsigmoid = self.config_model.get("hardsigmoid", False)

    def to(self, device):
        self._meta.to(device)
        return super().to(device)

    def contain(self, pts):
        return self.meta.contain(pts)

    @property
    def meta(self):
        return self._meta

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def n_outs(self):
        return self._n_outs

    def update_meta(self, ranges: torch.Tensor):
        self._meta.update(ranges)

    def visibility(self, x):
        device = x.device
        x = x.to(self.device)
        pos = x.unsqueeze(0) if x.dim() == 1 else x
        vis = torch.zeros(
            pos.shape[0], self.n_outs, dtype=torch.float32, device=self.device
        )
        mask = self.meta.contain(pos)
        vis[mask] = self(self.meta.norm_coord(pos[mask]).to(self.device)).to(device)
        vis[mask] = self._inv_xform_vis(vis[mask])
        return vis.squeeze() if x.dim() == 1 else vis

    def forward(self, x):
        assert torch.all(
            (x >= -1) & (x <= 1)
        ), f"The input contains a value out of range [-1,1]"

        out = super().forward(x)

        if self._do_hardsigmoid:
            out = torch.nn.functional.hardsigmoid(out)

        out = out * self.output_scale

        return out

    def model_dict(self, opt=None, sch=None, epoch=-1):
        model_dict = {
            "state_dict": self.state_dict(),
            "xform_cfg": self.config_xform,
            "model_cfg": self.config_model,
            "aabox_ranges": self._meta.ranges,
        }
        if opt:
            model_dict["optimizer"] = opt.state_dict()
        if sch:
            model_dict["scheduler"] = sch.state_dict()
        if epoch >= 0:
            model_dict["epoch"] = epoch
        return model_dict

    def save_state(self, filename, opt=None, sch=None, epoch=-1):
        print("[SirenTVis] saving model_dict ", filename)
        torch.save(self.model_dict(opt, sch, epoch), filename)
        print("[SirenTVis] saving finished")

    def load_model_dict(self, model_dict):
        print("[SirenTVis] loading model_dict")

        self.config_model = model_dict.get("model_cfg")
        self.config_xform = model_dict.get("xform_cfg")
        if self.config_model is None:
            raise KeyError('The model dictionary is lacking the "model_cfg" data')

        self._init_output_scale(self.config_model)
        self._do_hardsigmoid = self.config_model.get("hardsigmoid", False)
        self._xform_vis, self._inv_xform_vis = partial_xform_vis(self.config_xform)

        self._meta = AABox(model_dict["aabox_ranges"])

        state_dict = model_dict["state_dict"]
        if "input_scale" in state_dict:
            state_dict.pop("input_scale")
        if "scale" in model_dict.keys():
            state_dict["output_scale"] = model_dict["scale"]

        self.load_state_dict(state_dict)

        print("[SirenTVis] loading finished\n")

    @classmethod
    def load(cls, cfg_or_fname: Union[str, dict]):
        if isinstance(cfg_or_fname, dict):
            if "model" not in cfg_or_fname:
                raise KeyError("The configuration dictionary must contain model")
            if "ckpt_file" in cfg_or_fname["model"]:
                filepath = cfg_or_fname["model"]["ckpt_file"]
            else:
                print("[SirenTVis] creating from a configuration dict...")
                return cls(cfg_or_fname)
        elif isinstance(cfg_or_fname, str):
            filepath = cfg_or_fname
        else:
            raise ValueError(
                f"The argument of load function must be str or dict (received {cfg_or_fname} {type(cfg_or_fname)})"
            )

        print("[SirenTVis] creating from checkpoint", filepath)
        with open(filepath, "rb") as f:
            model_dict = torch.load(f, map_location="cpu")
            return cls.create_from_model_dict(model_dict)

    @classmethod
    def create_from_model_dict(cls, model_dict):
        cfg = {
            "model": model_dict["model_cfg"],
            "transform_vis": model_dict["xform_cfg"],
        }

        if "ckpt_file" in cfg["model"]:
            cfg["model"].pop("ckpt_file")

        net = cls(cfg)
        net.load_model_dict(model_dict)
        return net

    def _init_output_scale(self, siren_cfg):
        scale_cfg = siren_cfg.get("output_scale", {})
        init = scale_cfg.get("init")

        if init is None:
            output_scale = np.ones(self._n_outs)
        elif isinstance(init, str):
            output_scale = np.load(init)
        else:
            output_scale = np.asarray(init)

        assert len(output_scale) == self._n_outs, "len(output_scale) != out_features"

        output_scale = torch.tensor(np.nan_to_num(output_scale), dtype=torch.float32)

        if scale_cfg.get("fix", True):
            self.register_buffer("output_scale", output_scale, persistent=True)
        else:
            self.register_parameter("output_scale", torch.nn.Parameter(output_scale))
