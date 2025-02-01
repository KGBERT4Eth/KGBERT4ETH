import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch
class MatrixVectorScaledDotProductAttention(nn.Module):
    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, q, k, v, mask=None):
        """
        q: tensor of shape (n*b, d_k) k: tensor of shape (n*b, l, d_k) v: tensor of shape (n*b, l, d_v)

        returns: tensor of shape (n*b, d_v), tensor of shape(n*b, l)
        """
        attn = (q.unsqueeze(1) * k).sum(2)  # (n*b, l)
        attn = attn / self.temperature
        if mask is not None:
            attn = attn.masked_fill(mask, -np.inf)
        attn = self.softmax(attn)
        attn = self.dropout(attn)
        output = (attn.unsqueeze(2) * v).sum(1)
        return output, attn

class MultiheadAttPoolLayer(nn.Module):
    def __init__(self, n_head, d_q_original, d_k_original, dropout=0.1):
        super().__init__()
        assert d_k_original % n_head == 0  # make sure the outpute dimension equals to d_k_origin
        self.n_head = n_head
        self.d_k = d_k_original // n_head
        self.d_v = d_k_original // n_head

        self.w_qs = nn.Linear(d_q_original, n_head * self.d_k)
        self.w_ks = nn.Linear(d_k_original, n_head * self.d_k)
        self.w_vs = nn.Linear(d_k_original, n_head * self.d_v)

        nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (d_q_original + self.d_k)))
        nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(2.0 / (d_k_original + self.d_k)))
        nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(2.0 / (d_k_original + self.d_v)))

        self.attention = MatrixVectorScaledDotProductAttention(temperature=np.power(self.d_k, 0.5))
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, mask=None):
        """
        q: tensor of shape (b, d_q_original) k: tensor of shape (b, l, d_k_original) mask: tensor of shape (b, l)
        (optional, default None) returns: tensor of shape (b, n*d_v)
        """
        n_head, d_k, d_v = self.n_head, self.d_k, self.d_v

        bs, _ = q.size()
        bs, len_k, _ = k.size()

        qs = self.w_qs(q).view(bs, n_head, d_k)  # (b, n, dk)
        ks = self.w_ks(k).view(bs, len_k, n_head, d_k)  # (b, l, n, dk)
        vs = self.w_vs(k).view(bs, len_k, n_head, d_v)  # (b, l, n, dv)

        qs = qs.permute(1, 0, 2).contiguous().view(n_head * bs, d_k)
        ks = ks.permute(2, 0, 1, 3).contiguous().view(n_head * bs, len_k, d_k)
        vs = vs.permute(2, 0, 1, 3).contiguous().view(n_head * bs, len_k, d_v)

        if mask is not None:
            mask = mask.repeat(n_head, 1)
        output, attn = self.attention(qs, ks, vs, mask=mask)

        output = output.view(n_head, bs, d_v)
        output = output.permute(1, 0, 2).contiguous().view(bs, n_head * d_v)  # (b, n*dv)
        output = self.dropout(output)
        return output, attn

class DIVEAttentionLayer(nn.Module):
    def __init__(self, sent_dim, node_dim, num_heads=2):
        super(DIVEAttentionLayer, self).__init__()
        self.sent_dim = sent_dim
        self.node_dim = node_dim
        self.num_heads = num_heads
        # self.config = config

        self.node2sent_proj = nn.Linear(node_dim, sent_dim)
        self.sent2node_proj = nn.Linear(sent_dim, node_dim)

        self.pooler = MultiheadAttPoolLayer(num_heads, node_dim, sent_dim)
        self.co_attention = nn.MultiheadAttention(embed_dim=sent_dim, num_heads=num_heads)

        self.fc = nn.Sequential(
            nn.Linear(sent_dim + sent_dim, sent_dim),  # Fusion of the dimensions
            nn.ReLU(),
            nn.Linear(sent_dim, sent_dim)
        )

    def forward(self, hidden_states, X):
        bs, seq_len, _ = hidden_states.size()
        _, max_num_nodes, _ = X.size()

        # Pool the sequence representations
        pooled_seq, _ = self.pooler(X[:, 0, :], hidden_states)

        # Co-attention: Project the node representation to the sequence dimension
        node_rep_proj = self.node2sent_proj(X)  # [bs, max_num_nodes, sent_dim]

        # Apply multi-head co-attention across patient sequence and graph representations
        co_attention_seq, _ = self.co_attention(
            query=hidden_states.transpose(0, 1),  # [seq_len, bs, sent_dim]
            key=node_rep_proj.transpose(0, 1),    # [max_num_nodes, bs, sent_dim]
            value=node_rep_proj.transpose(0, 1)   # [max_num_nodes, bs, sent_dim]
        )

        pooled_seq_expand = pooled_seq.unsqueeze(1).expand(-1, seq_len, -1)  # [bs, seq_len, sent_dim]
        fused_seq = self.fc(torch.cat((pooled_seq_expand, co_attention_seq.transpose(0, 1)), dim=2))

        co_attention_node, _ = self.co_attention(
            query=node_rep_proj.transpose(0, 1),  # [max_num_nodes, bs, sent_dim],
            key = hidden_states.transpose(0, 1),  # [seq_len, bs, sent_dim]
            value = hidden_states.transpose(0, 1) # [seq_len, bs, sent_dim]
        )

        fused_node = self.sent2node_proj(co_attention_node.transpose(0, 1))  # [bs, max_num_nodes, node_dim]

        return fused_seq, fused_node

class CLSNodeAttentionLayer(nn.Module):
    """
    Only allow [CLS] to perform multi-head attention with nodes:
    - node_dim and cls_dim can be different, need to project each other
    """

    def __init__(self, cls_dim, node_dim, num_heads=2, dropout=0.1):
        super().__init__()
        self.cls_dim = cls_dim
        self.node_dim = node_dim

        # Node -> cls_dim
        self.node_proj = nn.Linear(node_dim, cls_dim)
        # cls -> node_dim (if updating nodes to node_dim)
        self.cls_proj = nn.Linear(cls_dim, node_dim)

        self.attn = nn.MultiheadAttention(
            embed_dim=cls_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        # Fuse [CLS] (original) with attention output
        self.fuse_cls = nn.Sequential(
            nn.Linear(cls_dim + cls_dim, cls_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Update node representation:
        self.fuse_node = nn.Sequential(
            nn.Linear(cls_dim, node_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

    def forward(self, cls_hidden, node_repr):
        """
        cls_hidden: [bs, 1, cls_dim]
        node_repr:  [bs, num_nodes, node_dim]
        Returns: fused_cls, fused_node
        """
        bs, num_nodes, _ = node_repr.size()
        # 1) Project node to cls_dim
        node_proj = self.node_proj(node_repr)  # [bs, num_nodes, cls_dim]

        # 2) Multihead attention:
        #   - query = cls_hidden [bs,1,cls_dim]
        #   - key,value = node_proj [bs,num_nodes,cls_dim]
        fused_cls_attn, attn_weights = self.attn(
            query=cls_hidden,  # (bs,1,cls_dim)
            key=node_proj,     # (bs,num_nodes,cls_dim)
            value=node_proj
        )
        # fused_cls_attn => [bs,1,cls_dim]

        # 3) Concatenate with original cls and fuse
        fused_cls = self.fuse_cls(torch.cat([cls_hidden, fused_cls_attn], dim=-1))
        # [bs,1,cls_dim]

        # 4) Allow nodes to attend to cls
        node_attn, _ = self.attn(
            query=node_proj,  # [bs,num_nodes,cls_dim]
            key=cls_hidden,   # [bs,1,cls_dim]
            value=cls_hidden
        )
        # node_attn => [bs,num_nodes,cls_dim]

        # 5) Fuse to node_dim
        fused_nodes = self.fuse_node(node_attn)

        return fused_cls, fused_nodes