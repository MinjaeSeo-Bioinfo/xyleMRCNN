import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    """U-Net에서 사용되는 이중 컨볼루션 블록"""
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
    """다운샘플링 경로"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    """업샘플링 경로"""
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # 업샘플링 방식 선택
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # 입력 텐서 크기가 2의 배수가 아닌 경우 패딩 처리
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # skip connection 연결
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    """출력 컨볼루션 레이어"""
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class AttentionGate(nn.Module):
    """어텐션 게이트 모듈"""
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
    """경계 인식에 중점을 둔, 세그멘테이션 및 경계 탐지를 위한 U-Net"""
    def __init__(self, n_channels=3, n_classes=1, bilinear=True, with_attention=True):
        super(XylemUNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.with_attention = with_attention

        # 인코더
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)

        # 어텐션 게이트
        if with_attention:
            self.attention1 = AttentionGate(512, 256, 128)
            self.attention2 = AttentionGate(256, 128, 64)
            self.attention3 = AttentionGate(128, 64, 32)

        # 디코더
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        
        # 출력 헤드: 세그멘테이션 마스크와 경계 마스크
        self.outc = OutConv(64, n_classes)
        self.outc_boundary = OutConv(64, 1)  # 경계 검출을 위한 추가 출력

        # 모델 파라미터 초기화
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
        # 인코더 경로
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        # 어텐션 적용 (있는 경우)
        if self.with_attention:
            x3 = self.attention1(x4, x3)
            x2 = self.attention2(x3, x2)
            x1 = self.attention3(x2, x1)

        # 디코더 경로
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        
        # 세그멘테이션 및 경계 마스크 출력
        logits = self.outc(x)
        boundary_logits = self.outc_boundary(x)
        
        return logits, boundary_logits

    def use_checkpointing(self):
        """메모리 최적화를 위한 체크포인팅 활성화"""
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
    """Xylem 세그멘테이션을 위한 복합 손실 함수"""
    def __init__(self, boundary_weight=2.0, dice_weight=1.0, bce_weight=1.0):
        super(XylemUNetLoss, self).__init__()
        self.boundary_weight = boundary_weight
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.bce_loss = nn.BCEWithLogitsLoss()
        
    def dice_loss(self, pred, target):
        """Dice 손실 계산"""
        smooth = 1.0
        pred = torch.sigmoid(pred)
        
        # 배치 차원을 따라 평탄화
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)
        
        intersection = (pred_flat * target_flat).sum()
        union = pred_flat.sum() + target_flat.sum()
        
        dice_score = (2. * intersection + smooth) / (union + smooth)
        return 1 - dice_score
        
    def forward(self, pred_mask, pred_boundary, target_mask, target_boundary=None):
        """
        통합 손실 계산
        
        Args:
            pred_mask: 예측된 세그멘테이션 마스크
            pred_boundary: 예측된 경계 마스크
            target_mask: 정답 세그멘테이션 마스크
            target_boundary: 정답 경계 마스크 (없으면 타겟에서 유도)
        """
        # 경계 마스크가 없는 경우 생성
        if target_boundary is None:
            # 경계 추출을 위한 커널 정의
            kernel_size = 3
            # 마스크를 팽창시키기 위해 max_pool2d 사용
            dilated = F.max_pool2d(
                target_mask, 
                kernel_size=kernel_size, 
                stride=1, 
                padding=kernel_size//2
            )
            # 마스크를 침식시키기 위해 반전된 마스크에 max_pool2d 적용
            eroded = 1.0 - F.max_pool2d(
                1.0 - target_mask,
                kernel_size=kernel_size,
                stride=1,
                padding=kernel_size//2
            )
            # 경계 = 팽창 - 침식
            target_boundary = (dilated - eroded).clamp(0, 1)
        
        # 세그멘테이션 손실
        bce_seg_loss = self.bce_loss(pred_mask, target_mask)
        dice_seg_loss = self.dice_loss(pred_mask, target_mask)
        seg_loss = self.bce_weight * bce_seg_loss + self.dice_weight * dice_seg_loss
        
        # 경계 손실 (가중치 적용)
        bce_boundary_loss = self.bce_loss(pred_boundary, target_boundary)
        dice_boundary_loss = self.dice_loss(pred_boundary, target_boundary)
        boundary_loss = self.boundary_weight * (self.bce_weight * bce_boundary_loss + self.dice_weight * dice_boundary_loss)
        
        # 총 손실
        total_loss = seg_loss + boundary_loss
        
        return total_loss, {
            'seg_loss': seg_loss.item(),
            'boundary_loss': boundary_loss.item()
        }

# 모델 사용 예시
def create_xylem_unet(n_channels=3, n_classes=1, with_attention=True, device='cuda'):
    """Xylem 세그멘테이션을 위한 U-Net 모델 생성"""
    model = XylemUNet(n_channels, n_classes, bilinear=True, with_attention=with_attention)
    return model.to(device)

def train_xylem_unet(model, train_loader, val_loader, num_epochs=100, lr=0.001, device='cuda'):
    """Xylem U-Net 모델 훈련"""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)
    criterion = XylemUNetLoss(boundary_weight=2.0)
    
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        # 훈련 모드
        model.train()
        train_loss = 0.0
        
        for batch_idx, (images, masks) in enumerate(train_loader):
            images = images.to(device)
