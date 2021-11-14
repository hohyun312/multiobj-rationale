import utils
from utils import vocab
import torch
import torch.nn.functional as F
from torch import nn


class GGNN(nn.Module):
    
    def __init__(self, node_hidden_size, n_message_passes,
                n_message_types):
        '''
        node_hidden_size (int)   : Size of node representation
        n_message_passes (int)   : Number of message passing steps.
        n_message_types (int)    : Number of edge types.
        '''

        super().__init__()
        
        self.node_hidden_size = node_hidden_size
        self.n_message_passes  = n_message_passes
        self.n_message_types   = n_message_types
        
        self.message_hidden_size  = 2 * node_hidden_size
        
        self.msg_nns = torch.nn.ModuleList()
        for _ in range(n_message_types):
            self.msg_nns.append(
                nn.Linear(node_hidden_size, self.message_hidden_size)
            )
        
        self.gru = nn.GRUCell(
            input_size=self.message_hidden_size,
            hidden_size=self.node_hidden_size,
            bias=True
        )
        
    
    def forward(self, h_node, adjacency): 
        '''
        h_node (tensor)            : Hidden representation of nodes (n, d)
        adjacency (sparse tensor)  : Adjacency tensor of size (e, n, n)
        '''
        for _ in range(self.n_message_passes):
            msgs = []
            for i in range(self.n_message_types):
                msg = self.msg_nns[i](h_node)
                msgs.append(torch.sparse.mm(adjacency[i], msg))

            msgs = sum(msgs).relu()
            h_node = self.gru(msgs, h_node)
        return h_node
    
    
class GraphEmbedding(nn.Module):
    def __init__(self, node_hidden_size, graph_hidden_size):
        super().__init__()
        self.node_hidden_size  = node_hidden_size
        self.graph_hidden_size = graph_hidden_size

        # Embed graphs
        self.node_gating = nn.Sequential(
            nn.Linear(node_hidden_size, 1),
            nn.Sigmoid()
        )
        self.node_to_graph = nn.Linear(node_hidden_size,
                                       graph_hidden_size)

    def forward(self, h_node, ptr):
        '''
        h_node (tensor)  : Hidden representation of nodes (n, d)
        '''
        attn   = self.node_gating(h_node).t()
        h_node = self.node_to_graph(h_node)
        
        h_graph = [torch.matmul(attn[:, i:j], h_node[i:j]) 
                   for i,j in zip(ptr[:-1], ptr[1:])]
        h_graph = torch.cat(h_graph, dim=0)
        return h_graph
    
    
class VAEEncoder(nn.Module):
    
    def __init__(self, embedding, n_message_passes, n_message_types, 
                 graph_hidden_size, latent_size):
        super().__init__()
        
        self.embedding = embedding
        self.ggnn      = GGNN(embedding.embedding_dim, n_message_passes, n_message_types)
        self.readout   = GraphEmbedding(embedding.embedding_dim, graph_hidden_size)
        
        self.loc_nn    = torch.nn.Linear(graph_hidden_size, latent_size)
        self.logvar_nn = torch.nn.Linear(graph_hidden_size, latent_size)
        
        
    def forward(self, g):
        h_node  = self.embedding(g.node_types)
        h_node  = self.ggnn(h_node, g.to_adjacency())
        h_graph = self.readout(h_node, g.ptr)
        
        # compute mean and log variance
        z_loc    = self.loc_nn(h_graph)
        z_logvar = self.logvar_nn(h_graph)
        
        return z_loc, z_logvar
    
    def sample(self, z_loc, z_logvar):
        return z_loc + torch.exp(z_logvar / 2) * torch.randn(*z_logvar.shape)
    

class VAEDecoder(nn.Module):
    
    def __init__(self, embedding, n_message_passes, n_message_types, 
                 graph_hidden_size, latent_size, n_node_types, n_edge_types):
        super().__init__()
        
        
        # neural net for subgraph embedding
        self.embedding = embedding
        self.ggnn      = GGNN(embedding.embedding_dim, n_message_passes, n_message_types)
        self.readout   = GraphEmbedding(embedding.embedding_dim, graph_hidden_size)
        
        # predict atom type
        hidden_size = embedding.embedding_dim + graph_hidden_size + latent_size
        self.atom_nn = nn.Sequential(
            nn.Linear(hidden_size, 2*hidden_size),
            nn.ReLU(),
            nn.Linear(2*hidden_size, n_node_types+1)
        )
        
        # predict bond type
        self.proj_nn = nn.Linear(embedding.embedding_dim, hidden_size)
        self.bond_nn = nn.Sequential(
            nn.TransformerEncoderLayer(d_model=hidden_size, nhead=2, batch_first=True),
            nn.Linear(hidden_size, n_edge_types+1)
        )
        
    def forward(self, g):
        h_node  = self.embedding(g.node_types)
        h_node  = self.ggnn(h_node, g.to_adjacency())
        h_graph = self.readout(h_node, g.ptr)
        return h_node, h_graph
        
    def atom_forward(self, h_node_focus, h_graph, z_sampled):
        h = torch.cat([h_node_focus, h_graph, z_sampled], dim=1)
        atom_pred = self.atom_nn(h)
        return atom_pred
    
    def bond_forward(self, h_node_seq, h_node_new, h_graph, z_sampled):
        h = torch.cat([h_node_new, h_graph, z_sampled], dim=1)
        proj = self.proj_nn(h_node_seq)
        return self.bond_nn(h.unsqueeze(1) + proj)
    
    
class VAE(nn.Module):
    
    def __init__(self, encoder, decoder, device="cpu"):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device  = device
        
    def forward(self, g):

        # encoder part
        z_loc, z_logvar = self.encoder(g)
        z_sampled = self.encoder.sample(z_loc, z_logvar)

        kl_loss = kullback_leibler_loss(z_loc, z_logvar).mean()


        # decoder part 
        sub_g = g.random_subgraph()
        lgen = utils.LabelGenerator(g, sub_g)

            # -- atom
        f_idx, y_atoms = lgen.front_index_and_node_labels()

        h_node, h_graph = self.decoder(sub_g)
        h_front = h_node[f_idx]
        atom_pred = self.decoder.atom_forward(h_front, h_graph, z_sampled)

        atom_loss = F.cross_entropy(atom_pred, y_atoms)


            # -- bond
        mask = ( y_atoms != vocab.n_atom_types )
        if any(mask):
            q_idx, y_bonds = lgen.queue_index_and_edge_labels(mask)

            # added node representation initialised to (node embedding + h_front)
            h_node_new = self.decoder.embedding(y_atoms[mask]) + h_front[mask]

            # node in queue
            h_node = torch.vstack([h_node, torch.zeros(h_node.size(1))])
            h_node_seq = h_node[q_idx]

            bond_pred = self.decoder.bond_forward(h_node_seq, 
                                                  h_node_new, 
                                                  h_graph[mask], 
                                                  z_sampled[mask])

        bond_loss = F.cross_entropy(bond_pred.view(-1, bond_pred.size(-1)), y_bonds.view(-1))  
        
        return kl_loss, atom_loss, bond_loss


def kullback_leibler_loss(loc, logvar):
    return -0.5 * torch.sum(1 + logvar - torch.square(loc) - torch.exp(logvar), dim=1)