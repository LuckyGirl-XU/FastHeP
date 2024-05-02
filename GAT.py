import torch
import torch.nn as nn
import enum

"""
@misc{Gordic2020PyTorchGAT,
  author = {Gordic, Aleksa},
  title = {pytorch-GAT},
  year = {2020},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {url{https://github.com/gordicaleksa/pytorch-GAT}},
}

"""

class LayerType(enum.Enum):
    IMP2 = 1,
    IMP3 = 2
class GAT(torch.nn.Module):

    def __init__(self, num_of_layers, num_heads_per_layer, num_features_per_layer, add_skip_connection=True, bias=True,
                 dropout=0.6, layer_type=LayerType.IMP3, log_attention_weights=False):
        super().__init__()
        assert num_of_layers == len(num_heads_per_layer) == len(num_features_per_layer) - 1, f'Enter valid arch params.'

        GATLayer = get_layer_type(layer_type)  
        num_heads_per_layer = [1] + num_heads_per_layer  

        gat_layers = []  
        for i in range(num_of_layers):
            layer = GATLayer(
                num_in_features=num_features_per_layer[i] * num_heads_per_layer[i],  
                num_out_features=num_features_per_layer[i+1],
                num_of_heads=num_heads_per_layer[i+1],
                concat=True if i < num_of_layers - 1 else False,  #
                activation=nn.ELU() if i < num_of_layers - 1 else None,  
                dropout_prob=dropout,
                add_skip_connection=add_skip_connection,
                bias=bias,
                log_attention_weights=log_attention_weights
            )
            gat_layers.append(layer)

        self.gat_net = nn.Sequential(
            *gat_layers,
        )

    def forward(self, data):
        return self.gat_net(data)

            

class GATLayer(torch.nn.Module):

    head_dim = 1

    def __init__(self, num_in_features, num_out_features, num_of_heads, layer_type, concat=True, activation=nn.ELU(),
                 dropout_prob=0.6, add_skip_connection=True, bias=True, log_attention_weights=False):

        super().__init__()

       
        self.num_of_heads = num_of_heads
        self.num_out_features = num_out_features
        self.concat = concat  
        self.add_skip_connection = add_skip_connection

        

        self.linear_proj = nn.Linear(num_in_features, num_of_heads * num_out_features, bias=False)

       
        self.scoring_fn_target = nn.Parameter(torch.Tensor(1, num_of_heads, num_out_features))
        self.scoring_fn_source = nn.Parameter(torch.Tensor(1, num_of_heads, num_out_features))


        
        if bias and concat:
            self.bias = nn.Parameter(torch.Tensor(num_of_heads * num_out_features))
        elif bias and not concat:
            self.bias = nn.Parameter(torch.Tensor(num_out_features))
        else:
            self.register_parameter('bias', None)

        
        self.skip_proj = nn.Linear(num_in_features, num_of_heads * num_out_features, bias=False)
        

        self.leakyReLU = nn.LeakyReLU(0.2)  
        self.softmax = nn.Softmax(dim=-1)  
        self.activation = activation
        
        self.dropout = nn.Dropout(p=dropout_prob)

        self.log_attention_weights = log_attention_weights  
        self.attention_weights = None  

        self.init_params(layer_type)

    def init_params(self, layer_type):
        
        nn.init.xavier_uniform_(self.linear_proj.weight)
        nn.init.xavier_uniform_(self.scoring_fn_target)
        nn.init.xavier_uniform_(self.scoring_fn_source)

        if self.bias is not None:
            torch.nn.init.zeros_(self.bias)

    def skip_concat_bias(self, attention_coefficients, in_nodes_features, out_nodes_features):
        if self.log_attention_weights:  
            self.attention_weights = attention_coefficients

        
        if not out_nodes_features.is_contiguous():
            out_nodes_features = out_nodes_features.contiguous()

        if self.add_skip_connection:  
            if out_nodes_features.shape[-1] == in_nodes_features.shape[-1]:  
                
                out_nodes_features += in_nodes_features.unsqueeze(1)
            else:
                
                out_nodes_features += self.skip_proj(in_nodes_features).view(-1, self.num_of_heads, self.num_out_features)

        if self.concat:
            
            out_nodes_features = out_nodes_features.view(-1, self.num_of_heads * self.num_out_features)
        else:
            
            out_nodes_features = out_nodes_features.mean(dim=self.head_dim)

        if self.bias is not None:
            out_nodes_features += self.bias

        return out_nodes_features if self.activation is None else self.activation(out_nodes_features)


class GATLayerImp3(GATLayer):
    
    src_nodes_dim = 0 
    trg_nodes_dim = 1  

    nodes_dim = 0     
    head_dim = 1       

    def __init__(self, num_in_features, num_out_features, num_of_heads, concat=True, activation=nn.ELU(),
                 dropout_prob=0.6, add_skip_connection=True, bias=True, log_attention_weights=False):

        
        super().__init__(num_in_features, num_out_features, num_of_heads, LayerType.IMP3, concat, activation, dropout_prob,
                      add_skip_connection, bias, log_attention_weights)

    def forward(self, data):
        
        in_nodes_features, edge_index, num_of_nodes = data  
        
        assert edge_index.shape[0] == 2, f'Expected edge index with shape=(2,E) got {edge_index.shape}'

        in_nodes_features = self.dropout(in_nodes_features)

        nodes_features_proj = self.linear_proj(in_nodes_features).view(-1, self.num_of_heads, self.num_out_features)
       
        nodes_features_proj = self.dropout(nodes_features_proj)  # in the official GAT imp they did dropout here as well
      
        scores_source = (nodes_features_proj * self.scoring_fn_source).sum(dim=-1)
        scores_target = (nodes_features_proj * self.scoring_fn_target).sum(dim=-1)

        
        scores_source_lifted, scores_target_lifted, nodes_features_proj_lifted = scores_source, scores_target, nodes_features_proj
        
        scores_per_edge = self.leakyReLU(scores_source_lifted + scores_target_lifted)
        
        nodes_features_proj_lifted_weighted = nodes_features_proj_lifted
        
        out_nodes_features = self.aggregate_neighbors(nodes_features_proj_lifted_weighted, edge_index, in_nodes_features, num_of_nodes)

        
        out_nodes_features = self.skip_concat_bias(None, in_nodes_features, out_nodes_features)
        return (out_nodes_features, edge_index, None)

   
    def neighborhood_aware_softmax(self, scores_per_edge, trg_index, num_of_nodes):
        
        scores_per_edge = scores_per_edge - scores_per_edge.max()
        exp_scores_per_edge = scores_per_edge.exp()  # softmax

        
        neigborhood_aware_denominator = self.sum_edge_scores_neighborhood_aware(exp_scores_per_edge, trg_index, num_of_nodes)

        
        attentions_per_edge = exp_scores_per_edge / (neigborhood_aware_denominator + 1e-16)

        
        return attentions_per_edge.unsqueeze(-1)

    def sum_edge_scores_neighborhood_aware(self, exp_scores_per_edge, trg_index, num_of_nodes):
        
        trg_index_broadcasted = self.explicit_broadcast(trg_index, exp_scores_per_edge)

        
        size = list(exp_scores_per_edge.shape)  
        size[self.nodes_dim] = num_of_nodes
        neighborhood_sums = torch.zeros(size, dtype=exp_scores_per_edge.dtype, device=exp_scores_per_edge.device)

        
        neighborhood_sums.scatter_add_(self.nodes_dim, trg_index_broadcasted, exp_scores_per_edge)

        
        return neighborhood_sums.index_select(self.nodes_dim, trg_index)

    def aggregate_neighbors(self, nodes_features_proj_lifted_weighted, edge_index, in_nodes_features, num_of_nodes):
        size = list(nodes_features_proj_lifted_weighted.shape)  
        size[self.nodes_dim] = num_of_nodes  
        out_nodes_features = torch.zeros(size, dtype=in_nodes_features.dtype, device=in_nodes_features.device)

        
        trg_index_broadcasted = self.explicit_broadcast(edge_index[self.trg_nodes_dim], nodes_features_proj_lifted_weighted)
        
        out_nodes_features.scatter_add_(self.nodes_dim, trg_index_broadcasted, nodes_features_proj_lifted_weighted)
        

        return out_nodes_features

    def lift(self, scores_source, scores_target, nodes_features_matrix_proj, edge_index):
        
        src_nodes_index = edge_index[self.src_nodes_dim]
        trg_nodes_index = edge_index[self.trg_nodes_dim]

        
        scores_source = scores_source.index_select(self.nodes_dim, src_nodes_index)
        scores_target = scores_target.index_select(self.nodes_dim, trg_nodes_index)
        nodes_features_matrix_proj_lifted = nodes_features_matrix_proj.index_select(self.nodes_dim, src_nodes_index)

        return scores_source, scores_target, nodes_features_matrix_proj_lifted

    def explicit_broadcast(self, this, other):
        
        for _ in range(this.dim(), other.dim()):
            this = this.unsqueeze(-1)

        
        return this.expand_as(other)


class GATLayerImp2(GATLayer):
    
    def __init__(self, num_in_features, num_out_features, num_of_heads, concat=True, activation=nn.ELU(),
                 dropout_prob=0.6, add_skip_connection=True, bias=True, log_attention_weights=False):

        super().__init__(num_in_features, num_out_features, num_of_heads, LayerType.IMP2, concat, activation, dropout_prob,
                         add_skip_connection, bias, log_attention_weights)

    def forward(self, data):
        

        in_nodes_features, connectivity_mask = data  # unpack data
        num_of_nodes = in_nodes_features.shape[0]
        assert connectivity_mask.shape == (num_of_nodes, num_of_nodes), \
            f'Expected connectivity matrix with shape=({num_of_nodes},{num_of_nodes}), got shape={connectivity_mask.shape}.'

        
        in_nodes_features = self.dropout(in_nodes_features)

        
        nodes_features_proj = self.linear_proj(in_nodes_features).view(-1, self.num_of_heads, self.num_out_features)

        nodes_features_proj = self.dropout(nodes_features_proj)  # in the official GAT imp they did dropout here as well

        
        scores_source = torch.sum((nodes_features_proj * self.scoring_fn_source), dim=-1, keepdim=True)
        scores_target = torch.sum((nodes_features_proj * self.scoring_fn_target), dim=-1, keepdim=True)

       
        scores_source = scores_source.transpose(0, 1)
        scores_target = scores_target.permute(1, 2, 0)

        
        all_scores = self.leakyReLU(scores_source + scores_target)
        
        all_attention_coefficients = self.softmax(all_scores + connectivity_mask)


        
        out_nodes_features = torch.bmm(all_attention_coefficients, nodes_features_proj.transpose(0, 1))

        
        out_nodes_features = out_nodes_features.permute(1, 0, 2)

       

        out_nodes_features = self.skip_concat_bias(all_attention_coefficients, in_nodes_features, out_nodes_features)
        return (out_nodes_features, connectivity_mask)



def get_layer_type(layer_type):
    assert isinstance(layer_type, LayerType), f'Expected {LayerType} got {type(layer_type)}.'
 
    if layer_type == LayerType.IMP2:
        return GATLayerImp2
    elif layer_type == LayerType.IMP3:
        return GATLayerImp3
    else:
        raise Exception(f'Layer type {layer_type} not yet supported.')
