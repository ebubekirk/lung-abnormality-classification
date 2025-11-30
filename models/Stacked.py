import tensorflow as tf
from tensorflow.keras import layers, Model

from models.DenseNet121 import DenseBlock, TransitionLayer
from models.EfficientNetB0 import MBConv
from models.ResNet34 import BasicBlock

# Stack 1: DenseNet + EfficientNet
class StackDenseEfficientNet(Model):
    def __init__(self, growth_rate=32, block_layers=(3, 6, 12, 8), num_classes=2):
        super().__init__()
        
        # DenseNet part
        self.densenet_stem = tf.keras.Sequential([
            layers.Conv2D(64, kernel_size=7, strides=2, padding="same", use_bias=False),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.MaxPooling2D(pool_size=3, strides=2, padding="same")
        ])
        
        num_channels = 64
        self.dense_blocks = []
        self.dense_transitions = []
        
        # Build Dense Blocks + Transition Layers
        for i, num_layers in enumerate(block_layers):
            block = DenseBlock(num_layers, growth_rate)
            self.dense_blocks.append(block)
            num_channels += num_layers * growth_rate
            if i != len(block_layers) - 1:
                trans = TransitionLayer(num_channels // 2)
                self.dense_transitions.append(trans)
                num_channels = num_channels // 2
        
        # EfficientNet part
        self.efficient_stem = tf.keras.Sequential([
            layers.Conv2D(32, 3, strides=1, padding="same", use_bias=False),
            layers.BatchNormalization(),
            layers.Activation(tf.nn.silu)
        ])
        
        # EfficientNet blocks
        self.efficient_blocks = [
            *self._make_efficient_blocks(32, 16, 1, 3, 1, 1),
            *self._make_efficient_blocks(16, 24, 6, 3, 2, 2),
        ]
        
        # Final layers
        self.final_norm = layers.BatchNormalization()
        self.final_relu = layers.ReLU()
        self.global_pool = layers.GlobalAveragePooling2D()
        self.classifier = layers.Dense(num_classes)
        
    def _make_efficient_blocks(self, in_ch, out_ch, expansion, kernel, stride, n):
        blocks = []
        for i in range(n):
            blocks.append(MBConv(
                in_ch=in_ch if i == 0 else out_ch,
                out_ch=out_ch,
                expansion=exp,
                kernel=kernel,
                stride=stride if i == 0 else 1
            ))
        return blocks
    
    def call(self, x, training=False):
        # DenseNet pathway
        x_dense = self.densenet_stem(x, training=training)
        
        for block, trans in zip(self.dense_blocks, self.dense_transitions):
            x_dense = block(x_dense, training=training)
            x_dense = trans(x_dense, training=training)
        
        x_dense = self.dense_blocks[-1](x_dense, training=training)
        x_dense = self.final_norm(x_dense, training=training)
        x_dense = self.final_relu(x_dense)
        
        # EfficientNet pathway
        x_eff = self.efficient_stem(x, training=training)
        for block in self.efficient_blocks:
            x_eff = block(x_eff)
        
        # Combine both pathways
        x_combined = tf.concat([x_dense, x_eff], axis=-1)
        
        # Final classification
        x_combined = self.global_pool(x_combined)
        x_combined = self.classifier(x_combined)
        
        return x_combined

# Stack 2: DenseNet + ResNet
class StackDenseResNet(Model):
    def __init__(self, growth_rate=32, block_layers=(3, 6, 12, 8), num_classes=2):
        super().__init__()
        
        # DenseNet part
        self.densenet_stem = tf.keras.Sequential([
            layers.Conv2D(64, kernel_size=7, strides=2, padding="same", use_bias=False),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.MaxPooling2D(pool_size=3, strides=2, padding="same")
        ])
        
        num_channels = 64
        self.dense_blocks = []
        self.dense_transitions = []
        
        for i, num_layers in enumerate(block_layers):
            block = DenseBlock(num_layers, growth_rate)
            self.dense_blocks.append(block)
            num_channels += num_layers * growth_rate
            if i != len(block_layers) - 1:
                trans = TransitionLayer(num_channels // 2)
                self.dense_transitions.append(trans)
                num_channels = num_channels // 2
        
        # ResNet part
        self.resnet_stem = tf.keras.Sequential([
            layers.Conv2D(64, kernel_size=7, strides=2, padding="same", use_bias=False),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.MaxPooling2D(pool_size=3, strides=2, padding="same")
        ])
        
        self.resnet_blocks = [
            *self._make_resnet_blocks(64, 64, 2, 1),
            *self._make_resnet_blocks(64, 128, 2, 2),
        ]
        
        # Final layers
        self.final_norm = layers.BatchNormalization()
        self.global_pool = layers.GlobalAveragePooling2D()
        self.classifier = layers.Dense(num_classes)
        
    def _make_resnet_blocks(self, in_channels, out_channels, blocks, stride):
        layers_list = []
        layers_list.append(BasicBlock(in_channels, out_channels, stride=stride))
        for _ in range(1, blocks):
            layers_list.append(BasicBlock(out_channels, out_channels, stride=1))
        return layers_list
    
    def call(self, x, training=False):
        # DenseNet pathway
        x_dense = self.densenet_stem(x, training=training)
        
        for block, trans in zip(self.dense_blocks, self.dense_transitions):
            x_dense = block(x_dense, training=training)
            x_dense = trans(x_dense, training=training)
        
        x_dense = self.dense_blocks[-1](x_dense, training=training)
        x_dense = self.final_norm(x_dense, training=training)
        
        # ResNet pathway
        x_res = self.resnet_stem(x, training=training)
        for block in self.resnet_blocks:
            x_res = block(x_res, training=training)
        
        # Combine both pathways
        x_combined = tf.concat([x_dense, x_res], axis=-1)
        
        # Final classification
        x_combined = self.global_pool(x_combined)
        x_combined = self.classifier(x_combined)
        
        return x_combined

# Stack 3: EfficientNet + ResNet
class StackEfficientResNet(Model):
    def __init__(self, num_classes=2):
        super().__init__()
        
        # EfficientNet part
        self.efficient_stem = tf.keras.Sequential([
            layers.Conv2D(32, 3, strides=2, padding="same", use_bias=False),
            layers.BatchNormalization(),
            layers.Activation(tf.nn.silu)
        ])
        
        self.efficient_blocks = [
            *self._make_efficient_blocks(32, 16, 1, 3, 1, 1),
            *self._make_efficient_blocks(16, 24, 6, 3, 2, 2),
        ]
        
        # ResNet part
        self.resnet_stem = tf.keras.Sequential([
            layers.Conv2D(64, kernel_size=7, strides=2, padding="same", use_bias=False),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.MaxPooling2D(pool_size=3, strides=2, padding="same")
        ])
        
        self.resnet_blocks = [
            *self._make_resnet_blocks(64, 64, 2, 1),
            *self._make_resnet_blocks(64, 128, 2, 2),
        ]
        
        # Final layers
        self.global_pool = layers.GlobalAveragePooling2D()
        self.classifier = layers.Dense(num_classes)
        
    def _make_efficient_blocks(self, in_ch, out_ch, expansion, kernel, stride, n):
        blocks = []
        for i in range(n):
            blocks.append(MBConv(
                in_ch=in_ch if i == 0 else out_ch,
                out_ch=out_ch,
                expansion=expansion,
                kernel=kernel,
                stride=stride if i == 0 else 1
            ))
        return blocks
    
    def _make_resnet_blocks(self, in_channels, out_channels, blocks, stride):
        layers_list = []
        layers_list.append(BasicBlock(in_channels, out_channels, stride=stride))
        for _ in range(1, blocks):
            layers_list.append(BasicBlock(out_channels, out_channels, stride=1))
        return layers_list
    
    def call(self, x, training=False):
        # EfficientNet pathway
        x_eff = self.efficient_stem(x, training=training)
        for block in self.efficient_blocks:
            x_eff = block(x_eff)
        
        # ResNet pathway
        x_res = self.resnet_stem(x, training=training)
        for block in self.resnet_blocks:
            x_res = block(x_res, training=training)
        
        # Combine both pathways
        x_combined = tf.concat([x_eff, x_res], axis=-1)
        
        # Final classification
        x_combined = self.global_pool(x_combined)
        x_combined = self.classifier(x_combined)
        
        return x_combined

# Stack 4: DenseNet + EfficientNet + ResNet
class StackDenseEfficientResNet(Model):
    def __init__(self, growth_rate=32, block_layers=(2, 4, 8, 6), num_classes=2):
        super().__init__()
        
        # DenseNet part (simplified)
        self.densenet_stem = tf.keras.Sequential([
            layers.Conv2D(32, kernel_size=7, strides=2, padding="same", use_bias=False),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.MaxPooling2D(pool_size=3, strides=2, padding="same")
        ])
        
        num_channels = 32
        self.dense_blocks = []
        
        # Simplified DenseNet blocks
        for num_layers in block_layers[:2]:  # Only use first 2 blocks
            block = DenseBlock(num_layers, growth_rate // 2)  # Reduced growth rate
            self.dense_blocks.append(block)
            num_channels += num_layers * (growth_rate // 2)
        
        # EfficientNet part (simplified)
        self.efficient_stem = tf.keras.Sequential([
            layers.Conv2D(32, 3, strides=2, padding="same", use_bias=False),
            layers.BatchNormalization(),
            layers.Activation(tf.nn.silu)
        ])
        
        self.efficient_blocks = self._make_efficient_blocks(32, 16, 1, 3, 1, 2)
        
        # ResNet part (simplified)
        self.resnet_stem = tf.keras.Sequential([
            layers.Conv2D(32, kernel_size=7, strides=2, padding="same", use_bias=False),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.MaxPooling2D(pool_size=3, strides=2, padding="same")
        ])
        
        self.resnet_blocks = self._make_resnet_blocks(32, 64, 2, 1)
        
        # Feature fusion
        self.feature_fusion = tf.keras.Sequential([
            layers.Conv2D(128, 1, padding="same", use_bias=False),
            layers.BatchNormalization(),
            layers.ReLU()
        ])
        
        # Final layers
        self.global_pool = layers.GlobalAveragePooling2D()
        self.classifier = layers.Dense(num_classes)
        
    def _make_efficient_blocks(self, in_ch, out_ch, expansion, kernel, stride, n):
        blocks = []
        for i in range(n):
            blocks.append(MBConv(
                in_ch=in_ch if i == 0 else out_ch,
                out_ch=out_ch,
                expansion=expansion,
                kernel=kernel,
                stride=stride if i == 0 else 1
            ))
        return blocks
    
    def _make_resnet_blocks(self, in_channels, out_channels, blocks, stride):
        layers_list = []
        layers_list.append(BasicBlock(in_channels, out_channels, stride=stride))
        for _ in range(1, blocks):
            layers_list.append(BasicBlock(out_channels, out_channels, stride=1))
        return layers_list
    
    def call(self, x, training=False):
        # DenseNet pathway
        x_dense = self.densenet_stem(x, training=training)
        for block in self.dense_blocks:
            x_dense = block(x_dense, training=training)
        
        # EfficientNet pathway
        x_eff = self.efficient_stem(x, training=training)
        for block in self.efficient_blocks:
            x_eff = block(x_eff)
        
        # ResNet pathway
        x_res = self.resnet_stem(x, training=training)
        for block in self.resnet_blocks:
            x_res = block(x_res, training=training)
        
        # Combine all three pathways
        # Resize features to common spatial dimensions if needed
        target_shape = tf.shape(x_dense)[1:3]
        x_eff_resized = tf.image.resize(x_eff, target_shape)
        x_res_resized = tf.image.resize(x_res, target_shape)
        
        x_combined = tf.concat([x_dense, x_eff_resized, x_res_resized], axis=-1)
        x_combined = self.feature_fusion(x_combined, training=training)
        
        # Final classification
        x_combined = self.global_pool(x_combined)
        x_combined = self.classifier(x_combined)
        
        return x_combined
