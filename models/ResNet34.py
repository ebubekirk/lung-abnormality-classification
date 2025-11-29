import tensorflow as tf
from tensorflow.keras import layers, Model


# ------------------------------------------------
# Basic Residual Block (same as PyTorch BasicBlock)
# ------------------------------------------------
class BasicBlock(layers.Layer):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        self.conv1 = layers.Conv2D(out_channels, kernel_size=3, strides=stride,
                                   padding="same", use_bias=False)
        self.bn1 = layers.BatchNormalization()

        self.conv2 = layers.Conv2D(out_channels, kernel_size=3, strides=1,
                                   padding="same", use_bias=False)
        self.bn2 = layers.BatchNormalization()

        self.relu = layers.ReLU()

        # Downsample if needed (1×1 conv)
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = tf.keras.Sequential([
                layers.Conv2D(out_channels, kernel_size=1, strides=stride, use_bias=False),
                layers.BatchNormalization()
            ])

    def call(self, x, training=False):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out, training=training)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out, training=training)

        if self.downsample is not None:
            identity = self.downsample(identity, training=training)

        out = layers.add([out, identity])
        out = self.relu(out)

        return out


# ------------------------------------------------
# ResNet-34 Model
# ------------------------------------------------
class ResNet34(Model):
    def __init__(self, num_classes):
        super().__init__()

        # Initial conv (7×7) + maxpool
        self.conv1 = tf.keras.Sequential([
            layers.Conv2D(64, kernel_size=7, strides=2, padding="same", use_bias=False),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.MaxPooling2D(pool_size=3, strides=2, padding="same")
        ])

        # ResNet34 stages: 3, 4, 6, 3 blocks
        self.layer1 = self._make_layer(64, 64, blocks=3, stride=1)
        self.layer2 = self._make_layer(64, 128, blocks=4, stride=2)
        self.layer3 = self._make_layer(128, 256, blocks=6, stride=2)
        self.layer4 = self._make_layer(256, 512, blocks=3, stride=2)

        # Classifier
        self.pool = layers.GlobalAveragePooling2D()
        self.fc = layers.Dense(num_classes)

    def _make_layer(self, in_channels, out_channels, blocks, stride):
        layers_list = []

        # First block with stride (downsample)
        layers_list.append(BasicBlock(in_channels, out_channels, stride=stride))

        # Remaining blocks
        for _ in range(1, blocks):
            layers_list.append(BasicBlock(out_channels, out_channels, stride=1))

        return layers_list

    def call(self, x, training=False):
        x = self.conv1(x, training=training)

        for block in self.layer1:
            x = block(x, training=training)
        for block in self.layer2:
            x = block(x, training=training)
        for block in self.layer3:
            x = block(x, training=training)
        for block in self.layer4:
            x = block(x, training=training)

        x = self.pool(x)
        x = self.fc(x)
        return x
