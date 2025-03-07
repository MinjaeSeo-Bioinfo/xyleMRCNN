import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    """Double convolution block used in U-Net"""
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    """Downsampling path"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    """Upsampling path"""
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # Choose upsampling method
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # Handle padding if input tensor size is not a multiple of 2
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # Connect skip connection
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    """Output convolution layer"""
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class AttentionGate(nn.Module):
    """Attention Gate Module"""
    def __init__(self, F_g, F_l, F_int):
        super(AttentionGate, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        
        return x * psi

class XylemUNet(nn.Module):
    """U-Net for segmentation and boundary detection with focus on boundary awareness"""
    def __init__(self, n_channels=3, n_classes=1, bilinear=True, with_attention=True):
        super(XylemUNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.with_attention = with_attention

        # Encoder
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)

        # Attention Gates
        if with_attention:
            self.attention1 = AttentionGate(512, 256, 128)
            self.attention2 = AttentionGate(256, 128, 64)
            self.attention3 = AttentionGate(128, 64, 32)

        # Decoder
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        
        # Output heads: segmentation mask and boundary mask
        self.outc = OutConv(64, n_classes)
        self.outc_boundary = OutConv(64, 1)  # Additional output for boundary detection

        # Initialize model parameters
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Encoder path
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        # Apply attention (if enabled)
        if self.with_attention:
            x3 = self.attention1(x4, x3)
            x2 = self.attention2(x3, x2)
            x1 = self.attention3(x2, x1)

        # Decoder path
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        
        # Output segmentation and boundary masks
        logits = self.outc(x)
        boundary_logits = self.outc_boundary(x)
        
        return logits, boundary_logits

    def use_checkpointing(self):
        """Enable checkpointing for memory optimization"""
        self.inc = torch.utils.checkpoint.checkpoint(self.inc)
        self.down1 = torch.utils.checkpoint.checkpoint(self.down1)
        self.down2 = torch.utils.checkpoint.checkpoint(self.down2)
        self.down3 = torch.utils.checkpoint.checkpoint(self.down3)
        self.down4 = torch.utils.checkpoint.checkpoint(self.down4)
        self.up1 = torch.utils.checkpoint.checkpoint(self.up1)
        self.up2 = torch.utils.checkpoint.checkpoint(self.up2)
        self.up3 = torch.utils.checkpoint.checkpoint(self.up3)
        self.up4 = torch.utils.checkpoint.checkpoint(self.up4)
        self.outc = torch.utils.checkpoint.checkpoint(self.outc)
        self.outc_boundary = torch.utils.checkpoint.checkpoint(self.outc_boundary)

class XylemUNetLoss(nn.Module):
    """Composite loss function for Xylem segmentation"""
    def __init__(self, boundary_weight=2.0, dice_weight=1.0, bce_weight=1.0):
        super(XylemUNetLoss, self).__init__()
        self.boundary_weight = boundary_weight
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.bce_loss = nn.BCEWithLogitsLoss()
        
    def dice_loss(self, pred, target):
        """Calculate Dice loss"""
        smooth = 1.0
        pred = torch.sigmoid(pred)
        
        # Flatten along batch dimension
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)
        
        intersection = (pred_flat * target_flat).sum()
        union = pred_flat.sum() + target_flat.sum()
        
        dice_score = (2. * intersection + smooth) / (union + smooth)
        return 1 - dice_score
        
    def forward(self, pred_mask, pred_boundary, target_mask, target_boundary=None):
        """
        Calculate integrated loss
        
        Args:
            pred_mask: Predicted segmentation mask
            pred_boundary: Predicted boundary mask
            target_mask: Ground truth segmentation mask
            target_boundary: Ground truth boundary mask (if None, derived from target)
        """
        # Generate boundary mask if not provided
        if target_boundary is None:
            # Define kernel for boundary extraction
            kernel_size = 3
            # Use max_pool2d to dilate the mask
            dilated = F.max_pool2d(
                target_mask, 
                kernel_size=kernel_size, 
                stride=1, 
                padding=kernel_size//2
            )
            # Apply max_pool2d to inverted mask for erosion
            eroded = 1.0 - F.max_pool2d(
                1.0 - target_mask,
                kernel_size=kernel_size,
                stride=1,
                padding=kernel_size//2
            )
            # Boundary = dilation - erosion
            target_boundary = (dilated - eroded).clamp(0, 1)
        
        # Segmentation loss
        bce_seg_loss = self.bce_loss(pred_mask, target_mask)
        dice_seg_loss = self.dice_loss(pred_mask, target_mask)
        seg_loss = self.bce_weight * bce_seg_loss + self.dice_weight * dice_seg_loss
        
        # Boundary loss (with weight applied)
        bce_boundary_loss = self.bce_loss(pred_boundary, target_boundary)
        dice_boundary_loss = self.dice_loss(pred_boundary, target_boundary)
        boundary_loss = self.boundary_weight * (self.bce_weight * bce_boundary_loss + self.dice_weight * dice_boundary_loss)
        
        # Total loss
        total_loss = seg_loss + boundary_loss
        
        return total_loss, {
            'seg_loss': seg_loss.item(),
            'boundary_loss': boundary_loss.item()
        }

# Example usage
def create_xylem_unet(n_channels=3, n_classes=1, with_attention=True, device='cuda'):
    """Create U-Net model for Xylem segmentation"""
    model = XylemUNet(n_channels, n_classes, bilinear=True, with_attention=with_attention)
    return model.to(device)

def train_xylem_unet(model, train_loader, val_loader, num_epochs=100, lr=0.001, device='cuda'):
    """Train Xylem U-Net model"""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)
    criterion = XylemUNetLoss(boundary_weight=2.0)
    
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        # Training mode
        model.train()
        train_loss = 0.0
        
        for batch_idx, (images, masks) in enumerate(train_loader):
            images = images.to(device)
