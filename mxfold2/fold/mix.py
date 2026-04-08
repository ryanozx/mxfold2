import copy
import time
import torch
from .. import interface
from .fold import AbstractFold
from .rnafold import RNAFold
from .zuker import ZukerFold

class MixedFold(AbstractFold):
    def __init__(self, init_param=None, model_type: ZukerFold.ZukerType = ZukerFold.ZukerType.M, max_helix_length=30, **kwargs):
        super(MixedFold, self).__init__(interface.predict_mxfold, interface.partfunc_mxfold)
        self.turner = RNAFold(init_param=init_param)
        self.zuker = ZukerFold(model_type=model_type, max_helix_length=max_helix_length, **kwargs)
        self.max_helix_length = max_helix_length


    def forward(self, seq, return_param=False, param=None, return_partfunc=False, return_aux=False,
            max_internal_length=30, constraint=None, reference=None,
            loss_pos_paired=0.0, loss_neg_paired=0.0, loss_pos_unpaired=0.0, loss_neg_unpaired=0.0, **kwargs):
        timing = {
            "batch_size": len(seq),
            "reused_param": param is not None,
            "make_param_s": 0.0,
            "zuker_make_param_s": 0.0,
            "cpu_copy_s": 0.0,
            "clear_count_s": 0.0,
            "cast_contiguous_s": 0.0,
            "predict_s": 0.0,
            "partfunc_s": 0.0,
            "diff_score_s": 0.0,
        }
        total_start = time.perf_counter()

        aux = None
        if param is None:
            t0 = time.perf_counter()
            if return_aux:
                param, aux = self.make_param(seq, return_aux=True, **kwargs)
            else:
                param = self.make_param(seq, **kwargs)
            timing["make_param_s"] = time.perf_counter() - t0
            timing["zuker_make_param_s"] = copy.deepcopy(getattr(self.zuker, "_last_make_param_timing", 0.0))

        ss = []
        preds = []
        pairs = []
        pfs = []
        bpps = []
        for i in range(len(seq)):
            t0 = time.perf_counter()
            param_on_cpu = { 
                'turner': {k: v.detach().to("cpu") for k, v in param[i]['turner'].items() },
                'positional': {k: v.detach().to("cpu") for k, v in param[i]['positional'].items() }
            }
            timing["cpu_copy_s"] += time.perf_counter() - t0

            t0 = time.perf_counter()
            param_on_cpu = {k: self.clear_count(v) for k, v in param_on_cpu.items()}
            timing["clear_count_s"] += time.perf_counter() - t0

            t0 = time.perf_counter()
            param_on_cpu = {
                'turner': {k: v.to(dtype=torch.float32).contiguous() for k, v in param_on_cpu['turner'].items()},
                'positional': {k: v.to(dtype=torch.float32).contiguous() for k, v in param_on_cpu['positional'].items()}
            }
            timing["cast_contiguous_s"] += time.perf_counter() - t0

            """
            # sanity check
            for k, v in param_on_cpu["positional"].items():
                if k.startswith("count_"):
                    print("[debug] positional count tensor:", k, v.shape, v.dtype, v.device, v.is_contiguous())
            for k, v in param_on_cpu["turner"].items():
                if k.startswith("count_"):
                    print("[debug] turner count tensor:", k, v.shape, v.dtype, v.device, v.is_contiguous())
            """

            with torch.no_grad():
                t0 = time.perf_counter()
                v, pred, pair = interface.predict_mxfold(seq[i], param_on_cpu,
                            max_internal_length=max_internal_length if max_internal_length is not None else len(seq[i]),
                            max_helix_length=self.max_helix_length,
                            constraint=constraint[i].tolist() if constraint is not None else None, 
                            reference=reference[i].tolist() if reference is not None else None, 
                            loss_pos_paired=loss_pos_paired, loss_neg_paired=loss_neg_paired,
                            loss_pos_unpaired=loss_pos_unpaired, loss_neg_unpaired=loss_neg_unpaired)
                timing["predict_s"] += time.perf_counter() - t0
                if return_partfunc:
                    t0 = time.perf_counter()
                    pf, bpp = interface.partfunc_mxfold(seq[i], param_on_cpu,
                                max_internal_length=max_internal_length if max_internal_length is not None else len(seq[i]),
                                max_helix_length=self.max_helix_length,
                                constraint=constraint[i].tolist() if constraint is not None else None, 
                                reference=reference[i].tolist() if reference is not None else None, 
                                loss_pos_paired=loss_pos_paired, loss_neg_paired=loss_neg_paired,
                                loss_pos_unpaired=loss_pos_unpaired, loss_neg_unpaired=loss_neg_unpaired)
                    timing["partfunc_s"] += time.perf_counter() - t0
                    pfs.append(pf)
                    bpps.append(bpp)
            if torch.is_grad_enabled():
                t0 = time.perf_counter()
                v = self.calculate_differentiable_score(v, param[i]['positional'], param_on_cpu['positional'])
                timing["diff_score_s"] += time.perf_counter() - t0
            ss.append(v)
            preds.append(pred)
            pairs.append(pair)

        device = next(iter(param[0]['positional'].values())).device
        ss = torch.stack(ss) if torch.is_grad_enabled() else torch.tensor(ss, device=device)
        timing["total_s"] = time.perf_counter() - total_start
        self._last_forward_timing = timing
        
        if return_param and return_aux:
            return ss, preds, pairs, param, aux
        if return_param:
            return ss, preds, pairs, param
        elif return_partfunc:
            return ss, preds, pairs, pfs, bpps
        else:
            return ss, preds, pairs


    def make_param(self, seq, return_aux: bool = False, **kwargs):
        ts = self.turner.make_param(seq)
        if return_aux:
            ps, aux = self.zuker.make_param(seq, return_aux=True, **kwargs)
            return [{'turner': t, 'positional': p} for t, p in zip(ts, ps)], aux
        ps = self.zuker.make_param(seq, **kwargs)
        return [{'turner': t, 'positional': p} for t, p in zip(ts, ps)]
