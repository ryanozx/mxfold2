import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class StructuredLoss(nn.Module):
    def __init__(self, model, loss_pos_paired=0, loss_neg_paired=0, loss_pos_unpaired=0, loss_neg_unpaired=0, 
                l1_weight=0., l2_weight=0., verbose=False):
        super(StructuredLoss, self).__init__()
        self.model = model
        self.loss_pos_paired = loss_pos_paired
        self.loss_neg_paired = loss_neg_paired
        self.loss_pos_unpaired = loss_pos_unpaired
        self.loss_neg_unpaired = loss_neg_unpaired
        self.l1_weight = l1_weight
        self.l2_weight = l2_weight
        self.verbose = verbose


    def forward(self, seq, pairs, fname=None):
        pred, pred_s, _, param = self.model(seq, return_param=True, reference=pairs,
                                loss_pos_paired=self.loss_pos_paired, loss_neg_paired=self.loss_neg_paired, 
                                loss_pos_unpaired=self.loss_pos_unpaired, loss_neg_unpaired=self.loss_neg_unpaired)
        ref, ref_s, _ = self.model(seq, param=param, constraint=pairs, max_internal_length=None)
        l = torch.tensor([len(s) for s in seq], device=pred.device)
        margin_loss = (pred - ref) / l
        sl_loss = torch.zeros_like(margin_loss)
        loss = margin_loss + sl_loss
        if self.verbose:
            print("Loss = {} = ({} - {})".format(loss.item(), pred.item(), ref.item()))
            print(seq)
            print(pred_s)
            print(ref_s)
        if loss.item()> 1e10 or torch.isnan(loss).any():
            print()
            print(fname)
            print(loss.item(), pred.item(), ref.item())
            print(seq)

        l1_loss = torch.zeros_like(margin_loss)
        
        if self.l1_weight > 0.0:
            for p in self.model.parameters():
                l1_loss += self.l1_weight * torch.sum(torch.abs(p))

        # if self.l2_weight > 0.0:
        #     l2_reg = torch.zeros_like(margin_loss)
        #     for p in self.model.parameters():
        #         l2_reg += torch.sum(p ** 2)
        #     l2_loss += self.l2_weight * torch.sqrt(l2_reg)

        loss += l1_loss
        return loss, margin_loss, sl_loss, l1_loss


class StructuredLossWithTurner(nn.Module):
    def __init__(self, model, loss_pos_paired=0, loss_neg_paired=0, loss_pos_unpaired=0, loss_neg_unpaired=0, 
                l1_weight=0., l2_weight=0., sl_weight=1., verbose=False):
        super(StructuredLossWithTurner, self).__init__()
        self.model = model
        self.loss_pos_paired = loss_pos_paired
        self.loss_neg_paired = loss_neg_paired
        self.loss_pos_unpaired = loss_pos_unpaired
        self.loss_neg_unpaired = loss_neg_unpaired
        self.l1_weight = l1_weight
        self.l2_weight = l2_weight
        self.sl_weight = sl_weight
        self.verbose = verbose
        from .fold.rnafold import RNAFold
        from . import param_turner2004
        if getattr(self.model, "turner", None):
            self.turner = self.model.turner
        else:
            self.turner = RNAFold(param_turner2004).to(next(self.model.parameters()).device)


    def forward(self, seq, pairs, fname=None):
        pred, pred_s, _, param = self.model(seq, return_param=True, reference=pairs,
                                loss_pos_paired=self.loss_pos_paired, loss_neg_paired=self.loss_neg_paired, 
                                loss_pos_unpaired=self.loss_pos_unpaired, loss_neg_unpaired=self.loss_neg_unpaired)
        ref, ref_s, _ = self.model(seq, param=param, constraint=pairs, max_internal_length=None)
        with torch.no_grad():
            ref2, ref2_s, _ = self.turner(seq, constraint=pairs, max_internal_length=None)
        l = torch.tensor([len(s) for s in seq], device=pred.device)
        margin_loss = (pred - ref) / l  
        sl_loss = self.sl_weight * (ref-ref2) * (ref-ref2) / l

        loss = margin_loss + sl_loss

        if self.verbose:
            print("Loss = {} = ({} - {})".format(loss.item(), pred.item(), ref.item()))
            print(seq)
            print(pred_s)
            print(ref_s)
        if loss.item()> 1e10 or torch.isnan(loss).any():
            print()
            print(fname)
            print(loss.item(), pred.item(), ref.item())
            print(seq)

        l1_loss = torch.zeros_like(margin_loss)

        if self.l1_weight > 0.0:
            for p in self.model.parameters():
                l1_loss += self.l1_weight * torch.sum(torch.abs(p))

        # if self.l2_weight > 0.0:
        #     l2_reg = torch.zeros_like(margin_loss)
        #     for p in self.model.parameters():
        #         l2_reg += torch.sum(p ** 2)
        #     l2_loss += self.l2_weight * torch.sqrt(l2_reg)

        loss += l1_loss
        return loss, margin_loss, sl_loss, l1_loss