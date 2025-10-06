from tensorflow.keras import layers, models

def se_block(x, reduction=16):
    # Squeeze-and-Excitation for channel attention
    ch = x.shape[-1]
    se = layers.GlobalAveragePooling2D()(x)
    se = layers.Dense(ch // reduction, activation='relu')(se)
    se = layers.Dense(ch, activation='sigmoid')(se)
    se = layers.Reshape((1, 1, ch))(se)
    return layers.Multiply()([x, se])

def spatial_attention(x, kernel_size=7):
    # Spatial attention: concat avg+max across channel, conv -> sigmoid
    avg_pool = layers.Lambda(lambda t: layers.backend.mean(t, axis=-1, keepdims=True))(x)
    max_pool = layers.Lambda(lambda t: layers.backend.max(t, axis=-1, keepdims=True))(x)
    concat = layers.Concatenate(axis=-1)([avg_pool, max_pool])
    sa = layers.Conv2D(1, kernel_size, padding='same', activation='sigmoid')(concat)
    return layers.Multiply()([x, sa])

def residual_block(x, filters, stride=1):
    # Basic residual unit with BN+ReLU, stride on first conv for downsample
    shortcut = x
    y = layers.Conv2D(filters, 3, strides=stride, padding='same', use_bias=False)(x)
    y = layers.BatchNormalization()(y)
    y = layers.ReLU()(y)
    y = layers.Conv2D(filters, 3, padding='same', use_bias=False)(y)
    y = layers.BatchNormalization()(y)
    # match shortcut
    if shortcut.shape[-1] != filters or stride != 1:
        shortcut = layers.Conv2D(filters, 1, strides=stride, padding='same', use_bias=False)(shortcut)
        shortcut = layers.BatchNormalization()(shortcut)
    y = layers.Add()([shortcut, y])
    y = layers.ReLU()(y)
    return y

def mfab(x, filters):
    # Multi-scale Fusion Attention Block (channel-focused multi-scale)
    # parallel convs 1x1, 3x3, 5x5 then fuse with channel attention
    b1 = layers.Conv2D(filters, 1, padding='same', use_bias=False)(x)
    b1 = layers.BatchNormalization()(b1)
    b1 = layers.ReLU()(b1)

    b2 = layers.Conv2D(filters, 3, padding='same', use_bias=False)(x)
    b2 = layers.BatchNormalization()(b2)
    b2 = layers.ReLU()(b2)

    b3 = layers.Conv2D(filters, 5, padding='same', use_bias=False)(x)
    b3 = layers.BatchNormalization()(b3)
    b3 = layers.ReLU()(b3)

    m = layers.Add()([b1, b2, b3])
    m = se_block(m)
    return m

def pab(x, filters):
    # Position-wise Attention Block (spatial attention on fused multi-scale features)
    m = mfab(x, filters)
    m = spatial_attention(m)
    return m

def decoder_block(x, skip, filters):
    x = layers.UpSampling2D()(x)
    # apply spatial attention on skip before concatenation (gated skip)
    if skip is not None:
        s = spatial_attention(skip)
        x = layers.Concatenate()([x, s])
    x = layers.Conv2D(filters, 3, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(filters, 3, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    # channel recalibration
    x = se_block(x)
    return x

def build_manet(input_shape=(256, 256, 1), num_classes=1, pretrained_encoder=None, trainable=False):
    inputs = layers.Input(shape=input_shape)

    # Encoder: optional pretrained backbone; otherwise residual encoder with attention
    skips = []
    if pretrained_encoder:
        base = pretrained_encoder(include_top=False, weights='imagenet', input_tensor=inputs)
        for layer in base.layers:
            layer.trainable = trainable
        # Collect typical skip stages by spatial size
        # Try to select 4 descending-resolution feature maps
        feature_maps = {}
        for l in base.layers:
            try:
                shape = l.output.shape
                if len(shape) == 4:
                    feature_maps[(int(shape[1]), int(shape[2]))] = l.output
            except:
                pass
        # pick largest to smallest four maps
        sizes = sorted(feature_maps.keys(), key=lambda s: (s[0] if s[0] is not None else 0, s[1] if s[1] is not None else 0), reverse=True)
        selected = [feature_maps[s] for s in sizes[:4]]
        # ensure order: high->low resolution
        skips = selected[:-1]
        x = selected[-1]
    else:
        # Custom residual encoder
        x = inputs
        e1 = residual_block(x, 64, stride=1)
        e1 = pab(e1, 64)  # attention at shallow stage
        skips.append(e1)
        e2 = residual_block(e1, 128, stride=2)
        e2 = pab(e2, 128)
        skips.append(e2)
        e3 = residual_block(e2, 256, stride=2)
        e3 = pab(e3, 256)
        skips.append(e3)
        x = residual_block(e3, 512, stride=2)  # bottleneck input

    # Bridge with multi-scale attention
    b = mfab(x, x.shape[-1])
    b = pab(b, b.shape[-1])

    # Align skip list to 3 stages for decoder
    # If coming from pretrained backbone, reduce/trim to 3 relevant skips
    if len(skips) > 3:
        skips = skips[:3]
    # Ensure order: deepest first for decoding
    skips = skips[::-1]  # now deepest skip first

    # Decoder stages
    d1 = decoder_block(b, skips[0] if len(skips) > 0 else None, 256)
    d2 = decoder_block(d1, skips[1] if len(skips) > 1 else None, 128)
    d3 = decoder_block(d2, skips[2] if len(skips) > 2 else None, 64)

    # Final upsample to input resolution if needed
    x = d3
    # Head
    activation = 'sigmoid' if num_classes == 1 else 'softmax'
    outputs = layers.Conv2D(num_classes, 1, activation=activation)(x)

    return models.Model(inputs, outputs)
