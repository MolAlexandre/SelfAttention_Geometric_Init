class ModelConfig:
    """Configuration du modèle BERT encoder-only"""

    def __init__(self,
                 vocab_size=30522,
                 d_model=256,
                 num_heads=4,
                 num_layers=4,
                 d_hidden=1024,
                 max_len=128,
                 dropout=0.1,
                 symmetric_init=True):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.d_hidden = d_hidden
        self.max_len = max_len
        self.dropout = dropout
        self.symmetric_init = symmetric_init

        assert d_model % num_heads == 0, "d_model doit être divisible par num_heads"


class VITConfig:
    """Configuration du modèle Vision Transformer"""

    def __init__(self,
                 num_classes=10,
                 img_size=32,
                 patch_size=4,
                 in_channels=3,
                 d_model=512,
                 num_heads=8,
                 num_layers=6,
                 d_hidden=2048,
                 dropout=0.0,
                 attention_dropout=0.0,
                 qkv_bias=True,
                 symmetric_init=False):
        self.num_classes = num_classes
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.d_hidden = d_hidden
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.qkv_bias = qkv_bias
        self.symmetric_init = symmetric_init

        self.num_patches = (img_size // patch_size) ** 2
        self.seq_len = self.num_patches + 1

        assert img_size % patch_size == 0, "img_size doit être divisible par patch_size"
        assert d_model % num_heads == 0, "d_model doit être divisible par num_heads"


class TrainingConfig:
    """Configuration de l'entraînement MLM"""

    def __init__(self,
                 batch_size=128,
                 gradient_accumulation_steps=2,
                 learning_rate=5e-5,
                 num_epochs=10,
                 warmup_ratio=0.1,
                 weight_decay=0.01,
                 gradient_clip=1.0,
                 mlm_probability=0.15,
                 num_workers=1,
                 device='cuda',
                 mixed_precision=True):
        self.batch_size = batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.effective_batch_size = batch_size * gradient_accumulation_steps
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.warmup_ratio = warmup_ratio
        self.weight_decay = weight_decay
        self.gradient_clip = gradient_clip
        self.mlm_probability = mlm_probability
        self.num_workers = num_workers
        self.device = device
        self.mixed_precision = mixed_precision


class VitTrainingConfig:
    """Configuration de l'entraînement ViT sur CIFAR-10"""

    def __init__(self,
                 batch_size=128,
                 gradient_accumulation_steps=1,
                 learning_rate=1e-4,
                 num_epochs=4,
                 warmup_ratio=0.1,
                 weight_decay=0.01,
                 gradient_clip=1.0,
                 num_workers=1,
                 device='cuda',
                 mixed_precision=True,
                 label_smoothing=0.01,
                 mixup_alpha=0.2,
                 cutmix_alpha=1.0,
                 ema_decay=0.99998,
                 ema_update_frequency=32):
        self.batch_size = batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.effective_batch_size = batch_size * gradient_accumulation_steps
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.warmup_ratio = warmup_ratio
        self.weight_decay = weight_decay
        self.gradient_clip = gradient_clip
        self.num_workers = num_workers
        self.device = device
        self.mixed_precision = mixed_precision

        self.label_smoothing = label_smoothing
        self.mixup_alpha = mixup_alpha
        self.cutmix_alpha = cutmix_alpha

        self.ema_decay = ema_decay
        self.ema_update_frequency = ema_update_frequency

        self.betas = (0.9, 0.999)
        self.eps = 1e-8


# ---------------------------------------------------------------------------
# Named presets 
# ---------------------------------------------------------------------------

class ViT6LayerCIFAR10(VITConfig):
    """ViT 6-layer pour CIFAR-10"""

    def __init__(self, symmetric_init=False):
        super().__init__(
            num_classes=10, img_size=32, patch_size=4, in_channels=3,
            d_model=512, num_heads=8, num_layers=6, d_hidden=2048,
            dropout=0.0, attention_dropout=0.0, qkv_bias=True,
            symmetric_init=symmetric_init,
        )


class BERTMiniConfig(ModelConfig):
    """BERT-Mini (4 layers) comme dans le papier"""

    def __init__(self, symmetric_init=False):
        super().__init__(
            vocab_size=30522, d_model=256, num_heads=4, num_layers=4,
            d_hidden=1024, max_len=128, dropout=0.1,
            symmetric_init=symmetric_init,
        )


class BERTBaseConfig(ModelConfig):
    """BERT-Base (12 layers) comme dans le papier"""

    def __init__(self, symmetric_init=False):
        super().__init__(
            vocab_size=30522, d_model=768, num_heads=12, num_layers=12,
            d_hidden=3072, max_len=512, dropout=0.1,
            symmetric_init=symmetric_init,
        )


class BERTLargeConfig(ModelConfig):
    """BERT-Large (24 layers) comme dans le papier"""

    def __init__(self, symmetric_init=False):
        super().__init__(
            vocab_size=30522, d_model=1024, num_heads=16, num_layers=24,
            d_hidden=4096, max_len=512, dropout=0.1,
            symmetric_init=symmetric_init,
        )
