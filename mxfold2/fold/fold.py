import time
import torch
import torch.nn as nn
import torch.nn.functional as F

class AbstractFold(nn.Module):
    def __init__(self, predict, partfunc):
        super(AbstractFold, self).__init__()
        self.predict = predict
        self.partfunc = partfunc
        self._last_forward_timing = None


    def clear_count(self, param):
        param_count = {}
        for n, p in param.items():
            if n.startswith("score_"):
                param_count["count_"+n[6:]] = torch.zeros_like(p)
        param.update(param_count)
        return param


    """
    def calculate_differentiable_score(self, v, param, count):
        s = 0
        for n, p in param.items():
            if n.startswith("score_"):
                s += torch.sum(p * count["count_"+n[6:]].to(p.device))
        s += v - s.item()
        return s
    """

    """
    # CHANGE: memory stable calculate_differentiable_score to avoid OOM
    def calculate_differentiable_score(self, v, param, count):
        s = next(iter(param.values())).new_zeros(())

        for n, p in param.items():
            if n.startswith("score_"):
                surrogate = torch.sum(p)
                cpu_dot = torch.sum(p.detach().cpu() * count["count_" + n[6:]])
                # Move only the scalar back to GPU (cheap, safe)
                s = s + surrogate - surrogate.detach() + cpu_dot.to(device=p.device, dtype=p.dtype)

        # Preserve gradient path through `v`
        s = s + v - s.detach()
        return s
    """
    
    def calculate_differentiable_score(self, v, param, count):
        """
        Differentiable reconstruction of the Zuker score.

        IMPORTANT:
        - `count_*` tensors can be extremely large (O(L^2)).
        - Moving them to GPU causes massive allocations (tens of GB) and OOM.
        - We therefore keep `count_*` on CPU and reduce them to a scalar FIRST,
          then move only the scalar contribution to GPU.
        """
        score_list = []
        count_list = []

        for n, p in param.items():
            score_list.append(p.view(-1))
            count_list.append(count["count_" + n[6:]].view(-1))

        scores_flat = torch.cat(score_list)
        count_flat = torch.cat(count_list).to(device=scores_flat.device, dtype=scores_flat.dtype)

        sval = torch.dot(scores_flat, count_flat)

        return sval + (v - sval.detach())

    def forward(self, seq, return_param=False, param=None, return_partfunc=False, return_aux=False,
            max_internal_length=30, max_helix_length=30, constraint=None, reference=None,
            loss_pos_paired=0.0, loss_neg_paired=0.0, loss_pos_unpaired=0.0, loss_neg_unpaired=0.0, **kwargs):
        timing = {
            "batch_size": len(seq),
            "reused_param": param is not None,
            "make_param_s": 0.0,
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

        ss = []
        preds = []
        pairs = []
        pfs = []
        bpps = []
        for i in range(len(seq)):
            # IMPORTANT: the C++ DP backend expects score/count tensors on CPU to be float32 and contiguous.
            # If AMP produced fp16/bf16 (or non-contiguous views), the backend can misinterpret memory.
            t0 = time.perf_counter()
            param_on_cpu = {k: v.detach().to("cpu") for k, v in param[i].items()}
            timing["cpu_copy_s"] += time.perf_counter() - t0

            t0 = time.perf_counter()
            param_on_cpu = self.clear_count(param_on_cpu)
            timing["clear_count_s"] += time.perf_counter() - t0

            t0 = time.perf_counter()
            param_on_cpu = {k: v.to(dtype=torch.float32).contiguous() for k, v in param_on_cpu.items()}
            timing["cast_contiguous_s"] += time.perf_counter() - t0
            with torch.no_grad():
                t0 = time.perf_counter()
                v, pred, pair = self.predict(seq[i], param_on_cpu,
                            max_internal_length=max_internal_length if max_internal_length is not None else len(seq[i]),
                            max_helix_length=max_helix_length,
                            constraint=constraint[i].tolist() if constraint is not None else None, 
                            reference=reference[i].tolist() if reference is not None else None, 
                            loss_pos_paired=loss_pos_paired, loss_neg_paired=loss_neg_paired,
                            loss_pos_unpaired=loss_pos_unpaired, loss_neg_unpaired=loss_neg_unpaired)
                timing["predict_s"] += time.perf_counter() - t0
                if return_partfunc:
                    t0 = time.perf_counter()
                    pf, bpp = self.partfunc(seq[i], param_on_cpu,
                                max_internal_length=max_internal_length if max_internal_length is not None else len(seq[i]),
                                max_helix_length=max_helix_length,
                                constraint=constraint[i].tolist() if constraint is not None else None, 
                                reference=reference[i].tolist() if reference is not None else None, 
                                loss_pos_paired=loss_pos_paired, loss_neg_paired=loss_neg_paired,
                                loss_pos_unpaired=loss_pos_unpaired, loss_neg_unpaired=loss_neg_unpaired)
                    timing["partfunc_s"] += time.perf_counter() - t0
                    pfs.append(pf)
                    bpps.append(bpp)
            if torch.is_grad_enabled():
                t0 = time.perf_counter()
                v = self.calculate_differentiable_score(v, param[i], param_on_cpu)
                timing["diff_score_s"] += time.perf_counter() - t0
            ss.append(v)
            preds.append(pred)
            pairs.append(pair)

        device = next(iter(param[0].values())).device
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
