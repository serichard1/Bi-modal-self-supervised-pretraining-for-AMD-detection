""" 
A PyTorch implementation of the Vision Transformer encoder.

- adapted from the work of <mildlyoverfitted> from his GitHub page:
https://github.com/jankrepl/mildlyoverfitted/blob/master/github_adventures/vision_transformer/custom.py

- which is itself a simplified re-implementation of the code from:
https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/vision_transformer.py

"""
import torch
import torch.nn as nn

class PatchEmbed(nn.Module):
    """ 
    Split input volume into non-overlapping patches and directly embed them.
    The proposed approach uses a single convolution layer,
    with a kernel and stride equals to the desired patch size.
    (instead of manually dividing the input and using a dense layer for the embedding).

    Since one of the modality is 3-dimensionnal (OCTs), 2 implementations are proposed:
        - using the individual OCT slices as channels = using Conv2D
        - reconstructing the 3D volume = using a Conv3D (slices = depth)

    Parameters
    ----------

    size: int
        Size of the image (must be square)
    
    patch_size: int
        Size of the patches (patches are squares)
    
    embed_dim: int
        Embedding dimension of the output

    conv3D: bool
        Type of convolutions to be used in the case of 3D input volume.

    Attributes
    ----------

    n_patches: int
        Number of patches extracted from the input volume
    
    projector: nn.Conv
        Convolution layer (2D or 3D) used for creating and embedding the patches

    input_shape, 
    patch_size, 
    embed_dim,
    """
    def __init__(self, size, patch_size=16, embed_dim=768, conv3D=False):
        super().__init__()

        self.size = size
        self.patch_size = patch_size
        self.n_patches = (self.size // self.patch_size) ** 2

        self.embed_dim = embed_dim

        # using lambda to compute any number of OCT slices
        self.projector = lambda in_channels: nn.Conv2d(
                                                in_channels, 
                                                embed_dim, 
                                                kernel_size=patch_size, 
                                                stride=patch_size,
                                                device="cuda:0")
        
        # utiliser conv3d va faire exploser le coût computationnel
        # + il faut s'assurer d'obtenir le même nombre de patch!
        # (input_shape[0] // patch_size) ** 2 * (input_shape[2] // patch_size)
        # if conv3D:
        #     self.projector = nn.Conv3d(
        #             1, 
        #             embed_dim, 
        #             kernel_size=patch_size, 
        #             stride=patch_size)
        # TODO
            
    def forward(self, x):
        """ 
        Forward pass

        Parameters
        ----------

        x: torch.Tensor
            Tensor of shape (Batch_size, Channels/Depth, Height, Width)

        Returns
        ----------

        x: torch.Tensor
            Embeddings of all the patches for every sample in the batch, shape (Batch_size, n_patches, embed_dim)

        """
        
        if len(x) > 1:
            x = torch.concat((self.projector(x[0].shape[1])(x[0]), self.projector(x[1].shape[1])(x[1])), dim=0)
        else: 
            x = self.projector(x.shape[1])(x) # (B, embed_dim, sqrt(n_patches), sqrt(n_patches))
        print("here!: ", x.shape)
        x = x.flatten(2) # (B, embed_dim, n_patches)
        x = x.transpose(1,2) # (B, n_patches, embed_dim)
        print(x.shape)
        return x
    
class MLP(nn.Module):
    """ 
    Implementation of the multilayer perceptron that is used as a projection head into the final embedding space
    Just like attention layer, input and output dimension remains the same

    Parameters
    ----------

    dim: int
        Number of input/output features

    hidden dim: int
        Number of nodes in the hidden layer

    p: float
        Dropout probability

    Attributes
    ----------

    fc1, fc2: nn.Linear
        The nth linear layer

    activ: nn.GELU
        GELU activation function
    
    drop: nn.Dropout
        Dropout layer

    """
    def __init__(self, dim, hidden_dim, p=0.2):
        super().__init__()
 
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.activ = nn.GELU()
        self.drop = nn.Dropout(p)

    def forward(self, x):

        """ 
        Forward pass

        Parameters
        ----------

        x: torch.Tensor
            Tensor of shape (Batch_size, n_patches, dim) 

        Returns
        ----------

        x: torch.Tensor
            Tensor of shape (Batch_size, n_patches, dim)

        """ 
        x = self.fc1(x)
        x = self.activ(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)

        return x

class Attention(nn.Module):
    """ 
    Implementation of the self-attention 

    Parameters
    ----------

    dim: int
        Dimension used for the tokens

    n_heads: int
        Number of attention heads

    qvk_p: float
        Dropout probability applied to the query, value, key tensors

    proj_p: float
        Dropout probability applied to the projector

    Attributes
    ----------

    scale: float
        Normalizing factor applied to the dot product
   
    qvk: nn.Linear
        Linear projection of the query, value, key tensors.

    projector: nn.Linear
        Linear projection of the concatenated output of all the attention heads
    
    qvk_drop, proj_drop: nn.Dropout
        Dropout layers for the query, value, key and the projector

    head_dim: int
        Dimension that is distributed equally to each of the attention head
    
    dim,
    n_heads
    """

    def __init__(self, dim, n_heads=12, qvk_p=0.2, proj_p=0.2):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.scale = self.head_dim ** -0.5 # not to feed too large values into the softmax which could lead to small gradient
                                           # ^0.5 revient à e^(0.5*ln(self.head_dim)) et avec le - on passe à l'inverse 
                                           # resulting graph is very close to CE loss ^


        # x 3 car on veut un embedding pour chaque objet q,v,k, 
        # SELF-attention, car les 3 obtenus à partir de la même source
        self.qvk = nn.Linear(dim, dim * 3) 

        
        self.projector = nn.Linear(dim, dim)

        self.qvk_drop = nn.Dropout(qvk_p)
        self.proj_drop = nn.Dropout(proj_p)


    def forward(self, x):

        """ 
        Forward pass

        Parameters
        ----------

        x: torch.Tensor
            Tensor of shape (Batch_size, n_patches, dim) 
            Contrairement aux implementations originelles, on se propose de ne pas ajouter le cls embedding,
            qui n'est a priori qu'une relique du transformer NLP:
            https://github.com/google-research/vision_transformer/issues/61#issuecomment-802233921

            mais simplement d'utiliser un average pooling de l'ensemble des tokens lors de la dernière étape.
            Ceci dit, pour le self-supervised, même pas besoin de projeter, on garde les features

        Returns
        ----------

        x: torch.Tensor
            Tensor of shape (Batch_size, n_patches + 1, dim)

        """

        n_batch, n_tokens, dim = x.shape # 1 token par batch

        if dim != self.dim:
            raise ValueError("Patches embeddings dimension should be the same as attention head dimension")
        
        
        # note: inutile de flatten, par défaut la couche linéaire ne s'interesse qu'à la dernière dimension.
        qvk = self.qvk(x) # output (batch, n_patches, dim * 3)

        # n_heads et head_dim ont été créé à partir de dim donc on se contente de les séparer à nouveau, avec le 3*dim aussi
        qvk = qvk.reshape(n_batch, n_tokens, 3, self.n_heads, self.head_dim) 
        qvk = qvk.permute(2,0,3,1,4) # (3, n_batch, n_heads, n_tokens, head_dim)

        # cette permutation permet de faciler l'extraction suivante:
        q, v, k = qvk[0], qvk[1], qvk[2]

        # transposition des keys pour que les dimensions s'alignent symétriquemment pour le dot product
        k_t = k.transpose(-2, -1) # (n_batch, n_heads, head_dim, n_tokens)
        dot_p = (q @ k_t) * self.scale # obtient (n_batch, n_heads, n_tokens, n_tokens)
        # compute dot avec @ et multiplie par scale pour limiter les valeurs trop élevées = regularization
        # i. e si head dim est très grand, alors scale est très faible (car ** -0.5) et limite gradient

        attn = dot_p.softmax(dim=-1)
        # allows to create a probability distribution that sums up to 1 and can be used as weights for our soon to be calculed weighted average
        attn = self.qvk_drop(attn)

        
        weighted_avg = attn @ v # (n_batch, n_heads, n_tokens, head_dim)
        weighted_avg = weighted_avg.transpose(1,2) # (n_batch, n_tokens, n_heads, head_dim)
        weighted_avg = weighted_avg.flatten(2) # (n_batch, n_tokens, dim)

        x = self.projector(weighted_avg) # (n_batch, n_tokens, dim)
        x = self.proj_drop(x)

        return x
    
    
class Block(nn.Module):
    """ 
    Basic block of Transformer

    Parameters
    ----------

    dim: int
        Dimension of the embedding

    n_heads: int
        Number of attention heads
    
    mlp_ratio: float
        Determines the hidden dimension of the MLP module w.r.t the input dimension

    proj_p, qvk_p, lin_p: float
        Dropout probability

    Attributes
    ----------

    norm1, norm2: nn.LayerNorm
        Normalization layers for the Linear projection

    attn: Attention
        Attention module

    mlp: MLP
        MLP module

    """
    def __init__(self, dim, n_heads, mlp_ratio=4.0, proj_p=0.2, qvk_p=0.2, lin_p=0.2):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = Attention(
                        dim,
                        n_heads=n_heads,
                        qvk_p=qvk_p,
                        proj_p=proj_p
                )
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        hidden_dim = int(dim * mlp_ratio)

        self.MLP = MLP(
                    dim,
                    hidden_dim=hidden_dim,
                    p=lin_p
                )
        

    def forward(self, x):

        """ 
        Forward pass

        Parameters
        ----------

        x: torch.Tensor
            Tensor of shape (Batch_size, n_patches + 1, dim) 

        Returns
        ----------

        x: torch.Tensor
            Tensor of shape (Batch_size, n_patches + 1, dim)

        """ 

        x += self.attn(self.norm1(x)) # here we prepare the MultiHead by adding the current x to this new attention instance (and norm ofc)
        x += self.MLP(self.norm2(x)) # two separate layer car each separate parameters

        return x
    
class ViT(nn.Module):
    """ 
    Basic block of Transformer

    Parameters
    ----------

    input_shape: tuple (int, int, int)
        Shape of the input volume corresponding to (Height, Width, Channels/Depth)
    
    patch_size: int
        Height or Width of the patches (patches are squares)

    n_classes: int
        NUmber of classes (in case of fine-tuning)

    embed_dim: int
        Dmensionnality of the patch embeddings

    n_blocks: int
        Number of Attention/MLP blocks
    
    n_heads: int
        Number of attention heads per MultiHeadAttention module
    
    mlp_ratio: float
        Determines the hidden dimension of the MLP module w.r.t the input dimension

    proj_p, qvk_p, lin_p: float
        Dropout probability

    Attributes
    ----------

    patch_embed: PatchEmbed
        Instance of the PatchEmbed layer/module?

    cls_token: nn.Parameter
        Learnable parameter representing a global view of the image, based on self-attention from all the patches

    pos_embed: nn.Parameter
        Positionnal embedding added to each of the embedded token.

    pos_drop: nn.Dropout
        Dropout layer

    ViT: nn.ModuleList
        List of all the Blocks of the network

    norm: nn.LayerNorm
        Layer Normalization
    """
    def __init__(self, 
                size=250, 
                patch_size=50,
                embed_dim=768, 
                n_blocks=12,
                n_heads=12, 
                mlp_ratio=4.,
                proj_p=0.2,
                qvk_p=0.2,
                lin_p=0.2):
        super().__init__()
        
        self.patch_embed = PatchEmbed(
                            size=size,
                            patch_size=patch_size,
                            embed_dim=embed_dim,
                        )
       
        # trainable positional embedding associated with the position,
        # while in the usual NLP transformer the embedding isn't trainable (uses sin/cos instead)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.patch_embed.n_patches, embed_dim)) 
        # 1,1 dim are added for convenience so it fits shape of the rest of the tokens.
        self.pos_drop = nn.Dropout(p=lin_p)

        self.ViT = nn.ModuleList(
            [
                Block(dim=embed_dim,
                      n_heads=n_heads,
                      mlp_ratio=mlp_ratio,
                      proj_p=proj_p,
                      qvk_p=qvk_p,
                      lin_p=lin_p)
                for _ in range(n_blocks)    
            ]
        )
        self.norm = nn.LayerNorm(embed_dim, eps=1e6)
        # self.pooling = nn.AvgPool2d(16)

    def forward(self, x):

        """ 
        Forward pass

        Parameters
        ----------

        x: torch.Tensor
            Tensor of shape (Batch_size, Depth, Height, Width) 
            Contrairement aux implementations originelles, on se propose de ne pas ajouter le cls embedding,
            qui n'est a priori qu'une relique du transformer NLP:
            https://github.com/google-research/vision_transformer/issues/61#issuecomment-802233921

            mais simplement d'utiliser un average pooling de l'ensemble des tokens lors de la dernière étape.
            Ceci dit, pour le self-supervised, même pas besoin de projeter, on garde les features
        Returns
        ----------

        logits: torch.Tensor
            Tensor of shape (Batch_size, out_dim) 
        """ 

        x = self.patch_embed(x) # returns (Batch_size, n_patches, embed_dim)
        # print(x.is_cuda)
        x += self.pos_embed # on ajoute os embeddings, vu que shape déjà nickel, Pytorch s'occupe de maatcher les dimensions
        # btw it is added as sum donc added to each patch equally, d'ou le pattern :think:
        x = self.pos_drop(x)

        for block in self.ViT:
            x = block(x)

        x = self.norm(x)
        x = x.mean(dim=1)
        print(x.shape)

        return x




