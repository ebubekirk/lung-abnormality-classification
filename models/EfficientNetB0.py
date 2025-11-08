import tensorflow as tf
from tensorflow.keras import layers, models


class SqueezeExcite(layers.Layer):
    def __init__(self, in_channels, reduction=4):
        super().__init__()
        reduced = in_channels // reduction
        self.avg_pool = layers.GlobalAveragePooling2D()
        self.fc1 = layers.Dense(reduced, activation="relu")
        self.fc2 = layers.Dense(in_channels, activation="sigmoid")

    def call(self, x):
        scale = self.avg_pool(x)
        scale = layers.Reshape((1, 1, scale.shape[-1]))(scale)
        scale = self.fc1(scale)
        scale = self.fc2(scale)
        return x * scale


class MBConv(layers.Layer):
    def __init__(self, in_ch, out_ch, expansion, kernel, stride):
        super().__init__()
        self.use_residual = (stride == 1 and in_ch == out_ch)
        hidden = in_ch * expansion

        self.expand = models.Sequential()
        if expansion != 1:
            self.expand.add(layers.Conv2D(hidden, 1, padding="same", use_bias=False))
            self.expand.add(layers.BatchNormalization())
            self.expand.add(layers.Activation(tf.nn.silu))

        self.depthwise = models.Sequential([
            layers.DepthwiseConv2D(kernel, strides=stride, padding="same", use_bias=False),
            layers.BatchNormalization(),
            layers.Activation(tf.nn.silu)
        ])

        self.se = SqueezeExcite(hidden)

        self.project = models.Sequential([
            layers.Conv2D(out_ch, 1, padding="same", use_bias=False),
            layers.BatchNormalization()
        ])

    def call(self, x):
        out = x
        if hasattr(self, "expand") and len(self.expand.layers) > 0:
            out = self.expand(out)

        out = self.depthwise(out)
        out = self.se(out)
        out = self.project(out)

        if self.use_residual:
            return x + out
        return out


class EfficientNetB0(tf.keras.Model):
    def __init__(self, num_classes=1000):
        super().__init__()

        self.stem = models.Sequential([
            layers.Conv2D(32, 3, strides=2, padding="same", use_bias=False),
            layers.BatchNormalization(),
            layers.Activation(tf.nn.silu)
        ])

        def make_blocks(in_ch, out_ch, exp, k, s, n):
            blocks = []
            for i in range(n):
                blocks.append(MBConv(
                    in_ch=in_ch if i == 0 else out_ch,
                    out_ch=out_ch,
                    expansion=exp,
                    kernel=k,
                    stride=s if i == 0 else 1
                ))
            return blocks

        self.blocks = [
            *make_blocks(32, 16, 1, 3, 1, 1),
            *make_blocks(16, 24, 6, 3, 2, 2),
            *make_blocks(24, 40, 6, 5, 2, 2),
            *make_blocks(40, 80, 6, 3, 2, 3),
            *make_blocks(80, 112, 6, 5, 1, 3),
            *make_blocks(112, 192, 6, 5, 2, 4),
            *make_blocks(192, 320, 6, 3, 1, 1),
        ]

        self.final_conv = models.Sequential([
            layers.Conv2D(1280, 1, padding="same", use_bias=False),
            layers.BatchNormalization(),
            layers.Activation(tf.nn.silu),
        ])

        self.classifier = models.Sequential([
            layers.GlobalAveragePooling2D(),
            layers.Dropout(0.2),
            layers.Dense(num_classes)
        ])

    def call(self, x):
        x = self.stem(x)
        for block in self.blocks:
            x = block(x)
        x = self.final_conv(x)
        x = self.classifier(x)
        return x

