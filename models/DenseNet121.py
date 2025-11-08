import tensorflow as tf
from tensorflow.keras import layers, Model


class DenseLayer(layers.Layer):
    def __init__(self, growth_rate, bn_size=4, drop_rate=0.0):
        super().__init__()
        inter_channels = bn_size * growth_rate

        self.norm1 = layers.BatchNormalization()
        self.relu1 = layers.ReLU()
        self.conv1 = layers.Conv2D(inter_channels, kernel_size=1, use_bias=False)

        self.norm2 = layers.BatchNormalization()
        self.relu2 = layers.ReLU()
        self.conv2 = layers.Conv2D(growth_rate, kernel_size=3, padding="same", use_bias=False)

        self.drop = layers.Dropout(drop_rate) if drop_rate > 0 else None

    def call(self, x, training=False):
        out = self.norm1(x, training=training)
        out = self.relu1(out)
        out = self.conv1(out)

        out = self.norm2(out, training=training)
        out = self.relu2(out)
        out = self.conv2(out)

        if self.drop is not None:
            out = self.drop(out, training=training)

        # Concatenate input with new features (DenseNet key idea!)
        out = tf.concat([x, out], axis=-1)
        return out


class DenseBlock(layers.Layer):
    def __init__(self, num_layers, growth_rate, bn_size=4, drop_rate=0.0):
        super().__init__()
        self.layers_list = [
            DenseLayer(growth_rate, bn_size, drop_rate)
            for _ in range(num_layers)
        ]

    def call(self, x, training=False):
        for layer in self.layers_list:
            x = layer(x, training=training)
        return x


class TransitionLayer(layers.Layer):
    def __init__(self, out_channels):
        super().__init__()
        self.norm = layers.BatchNormalization()
        self.relu = layers.ReLU()
        self.conv = layers.Conv2D(out_channels, kernel_size=1, use_bias=False)
        self.pool = layers.AveragePooling2D(pool_size=2, strides=2)

    def call(self, x, training=False):
        x = self.norm(x, training=training)
        x = self.relu(x)
        x = self.conv(x)
        x = self.pool(x)
        return x


class DenseNet(Model):
    def __init__(self, growth_rate=32, block_layers=(6, 12, 24, 16), num_classes=2):
        super().__init__()

        # Initial layer
        self.stem = tf.keras.Sequential([
            layers.Conv2D(64, kernel_size=7, strides=2, padding="same", use_bias=False),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.MaxPooling2D(pool_size=3, strides=2, padding="same")
        ])

        num_channels = 64
        self.blocks = []
        self.transitions = []

        # Build Dense Blocks + Transition Layers
        for i, num_layers in enumerate(block_layers):
            block = DenseBlock(num_layers, growth_rate)
            self.blocks.append(block)
            num_channels += num_layers * growth_rate
            if i != len(block_layers) - 1:
                trans = TransitionLayer(num_channels // 2)
                self.transitions.append(trans)
                num_channels = num_channels // 2

        # Final layers
        self.norm = layers.BatchNormalization()
        self.relu = layers.ReLU()
        self.global_pool = layers.GlobalAveragePooling2D()
        self.classifier = layers.Dense(num_classes)

    def call(self, x, training=False):
        x = self.stem(x, training=training)

        for block, trans in zip(self.blocks, self.transitions):
            x = block(x, training=training)
            x = trans(x, training=training)

        # Last Dense Block (no transition after it)
        x = self.blocks[-1](x, training=training)

        x = self.norm(x, training=training)
        x = self.relu(x)
        x = self.global_pool(x)
        x = self.classifier(x)
        return x
