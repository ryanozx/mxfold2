import torch
import torch.nn as nn
import torch.nn.functional as F

class AbstractFold(nn.Module):
    def __init__(self, predict, partfunc):
        super(AbstractFold, self).__init__()
        self.predict = predict
        self.partfunc = partfunc


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
        s = next(iter(param.values())).new_zeros(())

        for n, p in param.items():
            if n.startswith("score_"):
                # 1. Targeted Surrogate: Grad(p) will be proportional to count
                # We move count to GPU, but only for this specific multiplication
                c_gpu = count["count_" + n[6:]].to(device=p.device, dtype=p.dtype)
                surrogate = torch.sum(p * c_gpu)
                
                # 2. CPU Dot Product: The exact value for the forward pass
                cpu_dot = torch.sum(p.detach().cpu() * count["count_" + n[6:]])
                
                # 3. Combine: Value comes from cpu_dot, Gradient comes from surrogate
                s = s + surrogate - surrogate.detach() + cpu_dot.to(p.device)

        # Preserve gradient path through `v`
        s = s + v - s.detach()
        return s

    def forward(self, seq, return_param=False, param=None, return_partfunc=False,
            max_internal_length=30, max_helix_length=30, constraint=None, reference=None,
            loss_pos_paired=0.0, loss_neg_paired=0.0, loss_pos_unpaired=0.0, loss_neg_unpaired=0.0, **kwargs):
        param = self.make_param(seq) if param is None else param # reuse param or not
        ss = []
        preds = []
        pairs = []
        pfs = []
        bpps = []
        for i in range(len(seq)):
            # IMPORTANT: the C++ DP backend expects score/count tensors on CPU to be float32 and contiguous.
            # If AMP produced fp16/bf16 (or non-contiguous views), the backend can misinterpret memory.
            param_on_cpu = {k: v.detach().to("cpu") for k, v in param[i].items()}
            param_on_cpu = self.clear_count(param_on_cpu)
            param_on_cpu = {k: v.to(dtype=torch.float32).contiguous() for k, v in param_on_cpu.items()}
            with torch.no_grad():
                v, pred, pair = self.predict(seq[i], param_on_cpu,
                            max_internal_length=max_internal_length if max_internal_length is not None else len(seq[i]),
                            max_helix_length=max_helix_length,
                            constraint=constraint[i].tolist() if constraint is not None else None, 
                            reference=reference[i].tolist() if reference is not None else None, 
                            loss_pos_paired=loss_pos_paired, loss_neg_paired=loss_neg_paired,
                            loss_pos_unpaired=loss_pos_unpaired, loss_neg_unpaired=loss_neg_unpaired)
                if return_partfunc:
                    pf, bpp = self.partfunc(seq[i], param_on_cpu,
                                max_internal_length=max_internal_length if max_internal_length is not None else len(seq[i]),
                                max_helix_length=max_helix_length,
                                constraint=constraint[i].tolist() if constraint is not None else None, 
                                reference=reference[i].tolist() if reference is not None else None, 
                                loss_pos_paired=loss_pos_paired, loss_neg_paired=loss_neg_paired,
                                loss_pos_unpaired=loss_pos_unpaired, loss_neg_unpaired=loss_neg_unpaired)
                    pfs.append(pf)
                    bpps.append(bpp)
            if torch.is_grad_enabled():
                v = self.calculate_differentiable_score(v, param[i], param_on_cpu)
            ss.append(v)
            preds.append(pred)
            pairs.append(pair)

        device = next(iter(param[0].values())).device
        ss = torch.stack(ss) if torch.is_grad_enabled() else torch.tensor(ss, device=device)
        if return_param:
            return ss, preds, pairs, param
        elif return_partfunc:
            return ss, preds, pairs, pfs, bpps
        else:
            return ss, preds, pairs
