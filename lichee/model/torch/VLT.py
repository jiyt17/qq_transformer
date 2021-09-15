import torch
import torch.nn as nn
import torch.nn.functional as F

import math

# config
class config:
    def __init__(self):
        self.HIDDEN_SIZE = 768
        self.DROPOUT_R = 0.2
        self.FF_SIZE = 1024
        self.MULTI_HEAD = 12
        self.HIDDEN_SIZE_HEAD = 64
        self.X_LAYER = 5
        self.V_LAYER = 4
        self.L_LAYER = 1


# ------------------------------
# ---- Multi-Head Attention ----
# ------------------------------

class MHAtt(nn.Module):
    def __init__(self, C):
        super(MHAtt, self).__init__()
        self.C = C

        self.linear_v = nn.Linear(C.HIDDEN_SIZE, C.HIDDEN_SIZE)
        self.linear_k = nn.Linear(C.HIDDEN_SIZE, C.HIDDEN_SIZE)
        self.linear_q = nn.Linear(C.HIDDEN_SIZE, C.HIDDEN_SIZE)
        self.linear_merge = nn.Linear(C.HIDDEN_SIZE, C.HIDDEN_SIZE)

        self.dropout = nn.Dropout(C.DROPOUT_R)

    def forward(self, v, k, q, mask):
        n_batches = q.size(0)

        v = self.linear_v(v).view(
            n_batches,
            -1,
            self.C.MULTI_HEAD,
            self.C.HIDDEN_SIZE_HEAD
        ).transpose(1, 2)

        k = self.linear_k(k).view(
            n_batches,
            -1,
            self.C.MULTI_HEAD,
            self.C.HIDDEN_SIZE_HEAD
        ).transpose(1, 2)

        q = self.linear_q(q).view(
            n_batches,
            -1,
            self.C.MULTI_HEAD,
            self.C.HIDDEN_SIZE_HEAD
        ).transpose(1, 2)

        atted = self.att(v, k, q, mask)
        atted = atted.transpose(1, 2).contiguous().view(
            n_batches,
            -1,
            self.C.HIDDEN_SIZE
        )

        atted = self.linear_merge(atted)

        return atted

    def att(self, value, key, query, mask):
        d_k = query.size(-1)

        scores = torch.matmul(
            query, key.transpose(-2, -1)
        ) / math.sqrt(d_k)

        if mask is not None:
            scores = scores.masked_fill(mask, -1e9)

        att_map = F.softmax(scores, dim=-1)
        att_map = self.dropout(att_map)

        return torch.matmul(att_map, value)



# ---------------------------
# ---- Feed Forward Nets ----
# ---------------------------

class FFN(nn.Module):
    def __init__(self, C):
        super(FFN, self).__init__()

        self.fc1 = nn.Linear(C.HIDDEN_SIZE, C.FF_SIZE)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(C.DROPOUT_R)
        self.fc2 = nn.Linear(C.FF_SIZE, C.HIDDEN_SIZE)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x



class LayerNorm(nn.Module):
    def __init__(self, size, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.eps = eps

        self.a_2 = nn.Parameter(torch.ones(size))
        self.b_2 = nn.Parameter(torch.zeros(size))

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)

        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2



# ------------------------
# ---- Self Attention ----
# ------------------------

class SA(nn.Module):
    def __init__(self, C):
        super(SA, self).__init__()

        self.mhatt = MHAtt(C)
        self.ffn = FFN(C)

        self.dropout1 = nn.Dropout(C.DROPOUT_R)
        self.norm1 = LayerNorm(C.HIDDEN_SIZE)

        self.dropout2 = nn.Dropout(C.DROPOUT_R)
        self.norm2 = LayerNorm(C.HIDDEN_SIZE)

    def forward(self, x, x_mask):
        x = self.norm1(x + self.dropout1(
            self.mhatt(x, x, x, x_mask)
        ))

        x = self.norm2(x + self.dropout2(
            self.ffn(x)
        ))

        return x


# -------------------------------
# ------- Cross Attention -------
# -------------------------------

class CA(nn.Module):
    def __init__(self, C):
        super(CA, self).__init__()

        self.mhatt1 = MHAtt(C)
        self.mhatt2 = MHAtt(C)
        self.ffn = FFN(C)

        self.dropout1 = nn.Dropout(C.DROPOUT_R)
        self.norm1 = LayerNorm(C.HIDDEN_SIZE)

        self.dropout2 = nn.Dropout(C.DROPOUT_R)
        self.norm2 = LayerNorm(C.HIDDEN_SIZE)

        self.dropout3 = nn.Dropout(C.DROPOUT_R)
        self.norm3 = LayerNorm(C.HIDDEN_SIZE)

    def forward(self, x, y, x_mask, y_mask):  # y->x
        x = self.norm1(x + self.dropout1(
            self.mhatt1(x, x, x, x_mask)
        ))

        x = self.norm2(x + self.dropout2(
            self.mhatt2(y, y, x, y_mask)
        ))

        x = self.norm3(x + self.dropout3(
            self.ffn(x)
        ))

        return x


# ------------------------
# ----- CMEncoder --------
# ------------------------

class CMEmodule(nn.Module):
    def __init__(self, C):
        super(CMEmodule, self).__init__()
        self.SA1 = SA(C)
        self.SA2 = SA(C)
        self.CA1 = CA(C)
        self.CA2 = CA(C)

    def forward(self, x, y, x_mask, y_mask):
        x = self.SA1(x, x_mask)
        y = self.SA2(y, y_mask)
        x_res = self.CA1(x, y, x_mask, y_mask)
        y_res = self.CA2(y, x, y_mask, x_mask)
        return x_res, y_res


class CME(nn.Module):
    def __init__(self):
        super(CME, self).__init__()
        C = config()
        self.v_linear = nn.Linear(1536, C.HIDDEN_SIZE)
        self.SA1 = nn.ModuleList([SA(C) for _ in range(C.V_LAYER)])
        self.SA2 = nn.ModuleList([SA(C) for _ in range(C.L_LAYER)])
        self.enc_list = nn.ModuleList([CMEmodule(C) for _ in range(C.X_LAYER)])

    def forward(self, x, y, x_mask, y_mask): # vision language
        x = self.v_linear(x)
        for enc in self.SA1:
            x = enc(x, x_mask)
        for enc in self.SA2:
            y = enc(y, y_mask)
        for enc in self.enc_list:
            x, y = enc(x, y, x_mask, y_mask)

        x = x.mean(1)

        return x, y
