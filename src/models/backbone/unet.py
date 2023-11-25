import torch
from torch_geometric.nn import Linear, MLP, PointTransformerConv, fps, knn, knn_graph
from torch_geometric.utils import scatter

import torch
from torch_geometric.nn import PositionalEncoding
from src.models.backbone import PointNetEncoder


class FeatureTransfer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, k):
        super().__init__()
        self.silu = torch.nn.SiLU()
        self.pos_nn = MLP([3, out_channels], norm=None, plain_last=False, act=torch.nn.SiLU())
        self.attn_nn = MLP([out_channels, out_channels], norm=None, plain_last=False, act=torch.nn.SiLU())
        self.transformer = PointTransformerConv(in_channels, out_channels, pos_nn=self.pos_nn, attn_nn=self.attn_nn, add_self_loops=False)
        self.k = k

    def forward(self, source_pos, target_pos, source_feat, target_feat, x_batch=None, y_batch=None):
        id_kneighbor = knn(source_pos, target_pos, k=self.k, batch_x=x_batch, batch_y=y_batch)
        id_kneighbor[[0, 1]] = id_kneighbor[[1, 0]]
        return self.transformer(x=(source_feat, target_feat), pos=(source_pos, target_pos), edge_index=id_kneighbor)


class TransformerBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.mlp_in = MLP([in_channels, in_channels], plain_last=False, act='silu')
        self.mlp_out = MLP([out_channels, out_channels], plain_last=False, act=None)
        self.mlp_ctx = MLP([-1, out_channels], plain_last=False, act='silu')

        self.pos_nn = MLP([3, out_channels, out_channels], norm=None, plain_last=False, act=torch.nn.SiLU())
        self.attn_nn = MLP([out_channels, out_channels, out_channels], norm=None, plain_last=False, act=torch.nn.SiLU())
        self.transformer = PointTransformerConv(in_channels, out_channels, pos_nn=self.pos_nn, attn_nn=self.attn_nn)

    def forward(self, x, pos, edge_index):
        x = self.mlp_in(x)
        x = self.transformer(x, pos, edge_index)
        x = self.mlp_out(x)
        return x


class TransitionDown(torch.nn.Module):
    """Samples the input point cloud by a ratio percentage to reduce
    cardinality and uses an mlp to augment features dimensionnality.
    """
    def __init__(self, in_channels, out_channels, ratio=0.25, k=16):
        super().__init__()
        self.k = k
        self.ratio = ratio
        self.mlp = MLP([in_channels, out_channels], plain_last=False, act='silu')

    def forward(self, x, pos, batch):
        id_clusters = fps(pos, ratio=self.ratio, batch=batch)
        sub_batch = batch[id_clusters] if batch is not None else None
        id_k_neighbor = knn(pos, pos[id_clusters], k=self.k, batch_x=batch, batch_y=sub_batch)
        x = self.mlp(x)
        x_out = scatter(x[id_k_neighbor[1]], id_k_neighbor[0], dim=0, dim_size=id_clusters.size(0), reduce='mean')
        sub_pos, out = pos[id_clusters], x_out
        return out, sub_pos, sub_batch


class TransitionUp2(torch.nn.Module):
    def __init__(self, in_channels, out_channels, k=8):
        super().__init__()
        self.k = k
        self.mlp = MLP([out_channels, out_channels], plain_last=False, act='silu')
        self.ft = FeatureTransfer((in_channels, out_channels), out_channels, k=self.k)


    def forward(self, x, x_sub, pos, pos_sub, batch=None, batch_sub=None):
        x_interpolated = self.ft(source_pos=pos_sub, target_pos=pos, source_feat=x_sub, target_feat=x, x_batch=batch_sub, y_batch=batch)
        x = self.mlp(x) + x_interpolated
        return x


class ContextMerge(torch.nn.Module):
    def __init__(self, in_channels, ctx_channels, out_channels):
        super().__init__()
        self.ctx_mlp = MLP([ctx_channels, out_channels], plain_last=False, act='silu')
        self.mlp = MLP([in_channels, out_channels], plain_last=False, act='silu')

    def forward(self, x, batch, ctx):
        ctx = self.ctx_mlp(ctx)
        ctx = ctx[batch]
        x = self.mlp(x) + ctx
        return x


class UNet(torch.nn.Module):
    def __init__(self, in_channels, out_channels, dim_model, dim_ctx, k=16, normalize=True):
        super().__init__()
        self.k = k
        in_channels = max(in_channels, 1)

        self.time_embedding = PositionalEncoding(dim_ctx)
        self.enc_embedding = PointNetEncoder(dim_ctx)

        self.mlp_input = MLP([in_channels, dim_model[0]], plain_last=False)
        self.skip_mlp_input = Linear(in_channels, dim_model[0])
        self.mlp_par = MLP([in_channels, dim_model[0]], plain_last=False)

        self.time_merge_input = ContextMerge(dim_model[0], dim_ctx, dim_model[0])
        self.emc_merge_input = ContextMerge(dim_model[0], dim_ctx, dim_model[0])

        self.transfer_input = FeatureTransfer((dim_model[0], dim_model[0]), dim_model[0], k=self.k)
        self.transformer_input = TransformerBlock(dim_model[0], dim_model[0])
        
        self.transformers_down = torch.nn.ModuleList()
        self.transition_down = torch.nn.ModuleList()
        
        self.transformers_up = torch.nn.ModuleList()
        self.transition_up = torch.nn.ModuleList()

        self.time_merge_up = torch.nn.ModuleList()
        self.enc_merge_up = torch.nn.ModuleList()

        self.time_merge_down = torch.nn.ModuleList()
        self.enc_merge_down = torch.nn.ModuleList()

        for i in range(0, len(dim_model) - 1):
            self.transition_down.append(TransitionDown(dim_model[i], dim_model[i + 1], k=self.k))
            self.transformers_down.append(TransformerBlock(dim_model[i + 1], dim_model[i + 1]))
            self.time_merge_down.append(ContextMerge(dim_model[i + 1], dim_ctx, dim_model[i + 1]))
            self.enc_merge_down.append(ContextMerge(dim_model[i + 1], dim_ctx, dim_model[i + 1]))

            self.transition_up.append(TransitionUp2(dim_model[i + 1], dim_model[i]))
            self.transformers_up.append(TransformerBlock(dim_model[i], dim_model[i]))
            self.time_merge_up.append(ContextMerge(dim_model[i], dim_ctx, dim_model[i]))
            self.enc_merge_up.append(ContextMerge(dim_model[i], dim_ctx, dim_model[i]))

        self.mlp_summit = MLP([dim_model[-1], dim_model[-1]], norm=None, plain_last=False)
        self.transformer_summit = TransformerBlock(dim_model[-1], dim_model[-1])
        self.mlp_output = MLP([dim_model[0], dim_model[0], out_channels])


    def forward(self, x: torch.Tensor, t: torch.Tensor, par: torch.Tensor, batch: torch.Tensor, par_batch: torch.Tensor):
        out_x, out_pos, out_batch = [], [], []
        pos = x

        tctx = self.time_embedding(t)
        ctx = self.enc_embedding(par, par_batch)

        x = self.skip_mlp_input(pos) + self.mlp_input(pos)
        x_par = self.mlp_par(par)
        x = self.transfer_input(
            target_pos=pos, target_feat=x, y_batch=batch,
            source_pos=par, source_feat=x_par, x_batch=par_batch, 
        )

        x = self.time_merge_input(x, batch, tctx)
        x = self.emc_merge_input(x, batch, ctx)
        
        edge_index = knn_graph(pos, k=self.k, batch=batch)
        x = self.transformer_input(x, pos, edge_index)

        out_x.append(x); out_pos.append(pos); out_batch.append(batch)
        for i in range(len(self.transformers_down)):
            x, pos, batch = self.transition_down[i](x, pos, batch=batch)
            x = self.transformers_down[i](x, pos, knn_graph(pos, k=self.k, batch=batch))
            x = self.time_merge_down[i](x, batch, tctx)
            x = self.enc_merge_down[i](x, batch, ctx)

            out_x.append(x); out_pos.append(pos); out_batch.append(batch)


        x = self.mlp_summit(x)
        edge_index = knn_graph(pos, k=self.k, batch=batch)
        x = self.transformer_summit(x, pos, edge_index)

        n = len(self.transformers_down)
        for i in range(n):
            down_x, down_pos, down_batch = out_x[-i - 1], out_pos[-i - 1], out_batch[-i - 1]
            up_x, up_pos, up_batch = out_x[-i - 2], out_pos[-i - 2], out_batch[-i - 2]

            x = self.transition_up[-i - 1](x=up_x, pos=up_pos, batch=up_batch, x_sub=down_x, pos_sub=down_pos, batch_sub=down_batch)
            x = self.transformers_up[-i - 1](x, up_pos, knn_graph(up_pos, k=self.k, batch=up_batch))
            x = self.time_merge_up[-i - 1](x, up_batch, tctx)
            x = self.enc_merge_up[-i - 1](x, up_batch, ctx)

        out = self.mlp_output(x)
        return out
    

