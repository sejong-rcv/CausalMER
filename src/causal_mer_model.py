import torch
from torch import nn
import torch.nn.functional as F
from .models import MULTModel
from .model_t import *
from .model_a import *
from .model_v import *
import torchvision

eps = 1e-12

class CAUSALMER(nn.Module):
    def __init__(self, hyp_params):
        super(CAUSALMER, self).__init__()
        self.modality = hyp_params.modality
        self.nde_t_weight = hyp_params.nde_t
        self.nde_a_weight = hyp_params.nde_a
        self.nde_v_weight = hyp_params.nde_v
        self.activation = nn.ReLU()
        self.fusion_mode = hyp_params.fusion_mode
        output_dim = hyp_params.output_dim

        self.tav_branch = MULTModel(hyp_params)

        # ---------- text model --------
        self.t_branch = AlbertTextModel(hyp_params)

        # ---------- audio model --------
        self.a_branch = AudioModel(hyp_params)

        # ---------- vision model --------
        self.v_branch = VisionModel(hyp_params)

        self.constant_t = nn.Parameter(torch.tensor(hyp_params.constant))
        self.constant_a = nn.Parameter(torch.tensor(hyp_params.constant))
        self.constant_v = nn.Parameter(torch.tensor(hyp_params.constant))
    
    def fusion(self, z_m, z_t, z_a, z_v, constant, m_fact=False, t_fact=False, a_fact=False, v_fact=False):
        z_m, z_t, z_a, z_v = self.transform(z_m, z_t, z_a, z_v, constant, m_fact, t_fact, a_fact, v_fact)
        if self.fusion_mode == 'sum':
            z = z_m + z_t + z_a + z_v
            z = torch.log(torch.sigmoid(z) + eps)
        elif self.fusion_mode == 'hm':
            z_m = torch.sigmoid(z_m)
            z_t = torch.sigmoid(z_t)
            z_a = torch.sigmoid(z_a)
            z_v = torch.sigmoid(z_v)
            z = z_m * z_t * z_a * z_v
            z = torch.log(z + eps) - torch.log1p(z)
        elif self.fusion_mode == 'lm':
            import pdb; pdb.set_trace()
        elif self.fusion_mode == 'tanh': # [ACL 2023] Causal Intervention and Counterfactual Reasoning for Multi-modal Fake News Detection
            z_t = torch.tanh(z_t)
            z_a = torch.tanh(z_a)
            z_v = torch.tanh(z_v)
            z = z_m + z_t + z_a + z_v
        elif self.fusion_mode == 'mask': # CausalABSC method
            z_tav = torch.sigmoid(z_t + z_a + z_v)
            z = z_m * z_tav
        else:
            print("error")
        return z

    def transform(self, z_m, z_t, z_a, z_v, constant, m_fact=False, t_fact=False, a_fact=False, v_fact=False): 
        if (not m_fact) and (type(z_m) != int):
            z_m = constant * torch.ones_like(z_m).cuda()
        
        if (not t_fact) and (type(z_t) != int):
            z_t = constant * torch.ones_like(z_t).cuda()

        if (not a_fact) and (type(z_a) != int):
            z_a = constant * torch.ones_like(z_a).cuda()

        if (not v_fact) and (type(z_v) != int):
            z_v = constant * torch.ones_like(z_v).cuda()

        return z_m, z_t, z_a, z_v

    def forward(self, x_l, x_a, x_v):
        out = {}

        # iemocap - align
        # x_l.shape # torch.Size([32, 20, 300])
        # x_a.shape # torch.Size([32, 20, 74])
        # x_v.shape # torch.Size([32, 20, 35])

        # iemocap - unalign
        # x_l.shape # torch.Size([32, 20, 300])
        # x_a.shape # torch.Size([32, 400, 74])
        # x_v.shape # torch.Size([32, 4500, 35])
        
        logits, mult_last_hs = self.tav_branch(x_l, x_a, x_v) # model prediction
        if self.modality == 'tav':
            t_pred = self.t_branch(x_l)
            a_pred = self.a_branch(x_a)
            v_pred = self.v_branch(x_v)
            
            constant_ref = (self.constant_t + self.constant_a + self.constant_v)

            # logits_all
            z_tavm = self.fusion(logits, t_pred, a_pred, v_pred, constant_ref.clone().detach(), m_fact=True, t_fact=True, a_fact=True, v_fact=True)
            z_t = self.fusion(logits, t_pred, a_pred, v_pred, self.constant_t.clone().detach(), m_fact=False, t_fact=True, a_fact=False, v_fact=False)
            z_a = self.fusion(logits, t_pred, a_pred, v_pred, self.constant_a.clone().detach(), m_fact=False, t_fact=False, a_fact=True, v_fact=False)
            z_v = self.fusion(logits, t_pred, a_pred, v_pred, self.constant_v.clone().detach(), m_fact=False, t_fact=False, a_fact=False, v_fact=True)
            z_ = self.fusion(logits, t_pred, a_pred, v_pred, constant_ref.clone().detach(), m_fact=False, t_fact=False, a_fact=False, v_fact=False)

            # tie
            logits_causal_mer = z_tavm - z_t - z_a - z_v + 2*z_ 

            # for loss
            te = self.fusion(logits.clone().detach(), t_pred.clone().detach(), a_pred.clone().detach(), v_pred.clone().detach(), constant_ref, m_fact=True, t_fact=True, a_fact=True, v_fact=True) - self.fusion(logits.clone().detach(), t_pred.clone().detach(), a_pred.clone().detach(), v_pred.clone().detach(), constant_ref, m_fact=False, t_fact=False, a_fact=False, v_fact=False)
            nde_t = self.fusion(logits.clone().detach(), t_pred.clone().detach(), a_pred.clone().detach(), v_pred.clone().detach(), self.constant_t, m_fact=False, t_fact=True, a_fact=False, v_fact=False) - self.fusion(logits.clone().detach(), t_pred.clone().detach(), a_pred.clone().detach(), v_pred.clone().detach(), constant_ref, m_fact=False, t_fact=False, a_fact=False, v_fact=False)
            nde_a = self.fusion(logits.clone().detach(), t_pred.clone().detach(), a_pred.clone().detach(), v_pred.clone().detach(), self.constant_a, m_fact=False, t_fact=False, a_fact=True, v_fact=False) - self.fusion(logits.clone().detach(), t_pred.clone().detach(), a_pred.clone().detach(), v_pred.clone().detach(), constant_ref, m_fact=False, t_fact=False, a_fact=False, v_fact=False)
            nde_v = self.fusion(logits.clone().detach(), t_pred.clone().detach(), a_pred.clone().detach(), v_pred.clone().detach(), self.constant_v, m_fact=False, t_fact=False, a_fact=False, v_fact=True) - self.fusion(logits.clone().detach(), t_pred.clone().detach(), a_pred.clone().detach(), v_pred.clone().detach(), constant_ref, m_fact=False, t_fact=False, a_fact=False, v_fact=False)
            
            out['z_te'] = te # te
            out['z_nde'] = nde_t + nde_a + nde_v # nde

            out['z_nde_t'] = nde_t
            out['z_nde_a'] = nde_a
            out['z_nde_v'] = nde_v

            out['logits_te'] = z_tavm # for optimization # logits_all
            out['logits_m'] = logits # prediction of the original MER
            out['logits_causal_mer'] = logits_causal_mer # predction of causalMER, i.e., TIE
            out['logits_t'] = t_pred # for optimization
            out['logits_a'] = a_pred # for optimization
            out['logits_v'] = v_pred # for optimization
            
            out['mult_last_hs'] = mult_last_hs # for check modality contribution
        else:
            print("error")

        return out
        