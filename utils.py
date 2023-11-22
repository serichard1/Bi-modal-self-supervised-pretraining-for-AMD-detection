from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset
import albumentations as A
import cv2
import numpy as np
import os
import torch
import torch.nn as nn
import glob 

class DataAugmentation:
    """ 
    Creates augmentations with random probability,
    for each image and type (OCT and CFP).

    Parameters
    ----------

    resize: int
        Size of the output image
    
    n_augm: int
        Number of augmentations image per sample

    Attributes
    ----------

    base_aug: Albumentation <Compose>
        Augmentations to be applied to every sample
    
    projector: Albumentation <Compose>
        Augmentations to be applied to CFP samples only
    """
    def __init__(
        self,
        resize=420,
        n_augm=4
    ):
        self.n_aug = n_augm
        self.base_aug = A.Compose(
            [
                A.Resize(height=resize,width=resize),
                A.HorizontalFlip(),
                A.VerticalFlip(),
                ToTensorV2()
                # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ]
        )

        self.specific = A.Compose(
            [
                A.GridDistortion(p=0.2),
                A.CLAHE(p=0.2),
                A.GaussianBlur(p=0.2),
                A.ColorJitter(brightness=0.4, 
                              contrast=0.4,
                              saturation=0.2,
                              hue=0.1,
                              p=0.2),
                self.base_aug
            ]
        )
        
    def __call__(self, cfp_img, oct_img):
        """ Apply the given transformations to a duo of joint modality samples

        Parameters
        ----------
        cfp_img : np.array
            One CFP image as numpy array

        oct_img : np.array
            One OCT image as numpy array

        Returns
        -------

        augmented_imgs : list
            List of `torch.Tensor` representing different views of the two input images

        """
        augmented_imgs = []
        augmented_imgs.extend([self.specific(image=cfp_img)['image'] for _ in range(self.n_aug)])
        # h, w, _ = oct_img.shape
        # if w > h*2:
        #     split = w // 2
        #     oct_img = np.concatenate((oct_img[:, :split, :], oct_img[:, split:, :]), axis=2)

        # print(oct_img.shape)
        augmented_imgs.extend([self.base_aug(image=oct_img)['image'] for _ in range(self.n_aug)])
        # print(len(augmented_imgs))
        # for aug in augmented_imgs:
        #     print(aug.shape)

        return augmented_imgs


class BimodalDataset(Dataset):
    """ 
    Dataset class to import bimodal/joint image CFP and OCT.

    Parameters
    ----------

    root: Path
        Path to the data directory
    
    transform: DataAugmentation or torch.transforms or albumentations
        Type of transformations/augmentations to apply to the imported samples

    Attributes
    ----------

    cfp_path, oct_path Path
        Path to the directory containing the corresponding modality
    
    classes: List of str
        List of the names of the classes, note that it is not mandatory as it is self-supervised,
        it is only used for evaluation

    instances: List of str
        List of the paths to all the images
    """

    def __init__(self, root, transform):
        super().__init__()

        self.root = root
        self.transform = transform

        self.cfp_path = os.path.join(self.root, "cfp")
        self.oct_path = os.path.join(self.root, "oct")

        self.classes = []
        self.instances = []
        for root, dirs, imgs in os.walk(self.cfp_path, topdown=True):
            for classe in sorted(dirs):
                self.classes.append(classe)
            for img in sorted(imgs):
                self.instances.append(((os.path.join(root.split('/')[-1], img))))


    @staticmethod
    def np_loader(path, oct=False):
        # customized loader, instead of PIL to be able to load into numpy arrays and therefore use Albumentations augmentations
        if oct:
            img = [cv2.imread(slice, cv2.IMREAD_GRAYSCALE) for slice in glob.glob(f"{path}/*")] # shape [19,np.array(500,1000)]
            img = np.transpose(np.array(img),(1,2,0)) # shape (500,1000,19) = slices == channels
            return img
        
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # shape (420,420,3)
        return img

    def __len__(self):
        return len(self.instances)

    def __getitem__(self, idx):

        cfp_img = self.np_loader(os.path.join(self.cfp_path,self.instances[idx]))
        oct_img = self.np_loader(os.path.join(self.oct_path,self.instances[idx][:-4]), oct=True)

        imgs = self.transform(cfp_img, oct_img)

        label = self.classes.index(self.instances[idx].split('/')[0])
        # print(label)

        return imgs, label # list de 3 et 3 aug [6, resize, resize, channels]
    

class ProjectorHead(nn.Module):
    """ 
    Multi-Layer Perceptron head to project the tokens embeddings into the final feature space.

    Parameters
    ----------

    in_dim, hidden_dim, out_dim: int
        Respective dimensions of the input, hidden and output feature space.

    Attributes
    ----------

    projector: nn.Sequential
        Serie of linear layers and activation function
    """
    def __init__(
        self,
        in_dim=10,
        hidden_dim = 256,
        out_dim=1024
    ):
        super().__init__()
        self.projector = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim)
        )

        self.apply(self._init_weights)
        self.norm = nn.utils.weight_norm(
            nn.Linear(out_dim, out_dim, bias=False)
        )
        self.norm.weight_g.data.fill_(1) # we set the weight_G equals to 1 and disable learnable so only weight_v is learnable
        self.norm.weight_g.requires_grad = False

    # Comme décrit dans le papier du même nom, weight norm permet de découpler
    # la magnitude et la direction en weight_g and weight_v respectivement.
    # Note that the layer still has a weight attribute but not learnable anymore. It is now computed/calculated FROM the weight_g and weight_v.
    # Apprentissage de telle sorte que la weight finale soit égale à 1, donc tous les elements sont dans l'intevalle 0;1 = advanced standardisation

    def _init_weights(self, m):
        """Initialize learnable parameters."""
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """Run forward pass.
        Parameters
        ----------
        x : torch.Tensor
            Of shape `(n_samples, in_dim)`.
        Returns
        -------
        torch.Tensor
            Of shape `(n_samples, out_dim)`.
        """
        x = self.projector(x)  # (n_samples, bottleneck_dim)
        x = nn.functional.normalize(x, dim=-1, p=2)  # (n_samples, bottleneck_dim)
        x = self.norm(x)  # (n_samples, out_dim)

        return x


class BatchSampleWrapper(nn.Module):
    """ 
    Module used to encapsulate the ViT encoder and the projection head,
    while dealing with the joint augmentations from each modality.

    Parameters
    ----------

    encoder: ViT encoder
    projector: MLP head

    """
    def __init__(self, backbone, projector, n_aug, device):
        super().__init__()
        self.encoder = backbone
        self.projector= projector
        self.n_aug = n_aug
        self.device= device

    def forward(self, x):
        cfps = torch.stack(x[:self.n_aug]).flatten(0, 1)
        octs = torch.stack(x[self.n_aug:]).flatten(0, 1)
        print(cfps.shape)
        print(octs.shape)

        inputs = cfps.to(self.device).type(torch.float), octs.to(self.device).type(torch.float)
        output_features = self.projector(self.encoder(inputs))
        return output_features
    

class BarlowSampleLoss(nn.Module):
    """ Barlow Twin adapted for joint modality and sample wise normalisation
    Parameters
    ----------
    out_dim: int
        The dimensionality of the output of the projection head.
    temp: float
        Temperature for the scaling/regularization of the network
    n_aug: int
        Number of augmentations per sample
    n_batch: int
        Batch size
    """
    def __init__(
        self, 
        out_dim, 
        temp=0.01,
        batch=2):
        super().__init__()

        self.dim = out_dim
        self.temp = temp
        self.n_batch = batch

    @staticmethod
    def off_diagonal(x):
        # returns a flattened view of the off-diagonal elements of a square matrix
        n = x.shape[0]
        # flatten matrix into remove last item, reshape = view (n - 1, n + 1) 
        # la 1ere column devient seulement la diagonale, que l'on peut simplement enlever avec slice [:, 1:]
        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()
    
    @staticmethod
    def stand(tensor, dim):
        # standardisation ; referred to as "normalisation" in the paper
        m = torch.mean(tensor, dim)
        std = torch.std(tensor, dim)
        return (tensor-m)/std

    def forward(self, outputs):
        """ Compute loss
        Parameters
        ----------
        output: torch.Tensor
            Tensor corresponding of the output of the projection head for a batch,
            with shape (Batch, n_aug, dim)

        Returns
        -------
        loss : torch.Tensor
            Scalar representing the average loss.
        """

        output_cfps, output_octs = outputs.chunk(2, dim=0)
        print(output_cfps.shape)
        print(output_octs.shape)
        # standardisation along batch or feature dim
        x_norm, y_norm = self.stand(output_cfps, 0), self.stand(output_octs, 0)

        x_norm_t = x_norm.transpose(-2, -1)

        cross_corr = x_norm_t @ y_norm / (self.n_batch-1)

        on_diag = torch.diagonal(cross_corr)
        on_diag_loss = on_diag.add_(-1).pow_(2).sum()
        off_diag = self.off_diagonal(cross_corr)
        off_diag_loss = off_diag.pow_(2).sum()

        BT_loss = on_diag_loss + off_diag_loss * self.temp

        return BT_loss

