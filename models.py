#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author: Martin Savko (martin.savko@synchrotron-soleil.fr)
# part of the MURKO project


from murko import get_tiramisu_layer, get_dense_block, get_num_segmentation_classes


def tiramisu(
    nfilters=48,
    growth_rate=16,
    layers_scheme=[4, 5, 7, 10, 12],
    bottleneck=15,
    activation="relu",
    convolution_type="SeparableConv2D",
    padding="same",
    last_convolution=False,
    dropout_rate=0.2,
    weight_standardization=True,
    model_img_size=(None, None),
    input_channels=3,
    use_bias=False,
    kernel_initializer="he_normal",
    kernel_regularizer="l2",
    weight_decay=1e-4,
    heads=[
        {"name": "crystal", "type": "binary_segmentation"},
        {"name": "loop_inside", "type": "binary_segmentation"},
        {"name": "loop", "type": "binary_segmentation"},
        {"name": "stem", "type": "binary_segmentation"},
        {"name": "pin", "type": "binary_segmentation"},
        {"name": "foreground", "type": "binary_segmentation"},
    ],
    verbose=False,
    model_name="tiramisu",
    normalization_type="GroupNormalization",
    gn_groups=16,
    bn_momentum=0.9,
    bn_epsilon=1.1e-5,
    input_dropout=0.0,
    neck_layers=4,
    filter_size=3,
    final_filter_size=3,
):
    print("tiramisu heads", heads)
    boilerplate = {
        "activation": activation,
        "convolution_type": convolution_type,
        "padding": padding,
        "dropout_rate": dropout_rate,
        "use_bias": use_bias,
        "kernel_initializer": kernel_initializer,
        "kernel_regularizer": kernel_regularizer,
        "weight_decay": weight_decay,
        "normalization_type": normalization_type,
        "weight_standardization": weight_standardization,
    }

    inputs = keras.layers.Input(shape=(model_img_size) + (input_channels,))

    nfilters_start = nfilters

    if input_dropout > 0.0:
        x = keras.layers.Dropout(dropout_rate=input_dropout)(inputs)
    else:
        x = inputs

    x = get_tiramisu_layer(
        x,
        nfilters,
        filter_size=filter_size,
        padding=padding,
        activation=activation,
        convolution_type="Conv2D",
        dropout_rate=dropout_rate,
        use_bias=use_bias,
        kernel_initializer=kernel_initializer,
        weight_decay=weight_decay,
        bn_momentum=bn_momentum,
        bn_epsilon=bn_epsilon,
        normalization_type=normalization_type,
        weight_standardization=weight_standardization,
    )

    _skips = []

    # DOWN
    for l, number_of_layers in enumerate(layers_scheme):
        x, block_to_upsample = get_dense_block(
            x, growth_rate, number_of_layers, **boilerplate
        )
        _skips.append(x)
        nfilters += number_of_layers * growth_rate
        x = get_transition_down(x, nfilters, **boilerplate)
        if verbose:
            print("layer:", l, number_of_layers, "shape:", x.shape)

    # BOTTLENECK
    x, block_to_upsample = get_dense_block(x, growth_rate, bottleneck, **boilerplate)
    if verbose:
        print("bottleneck:", l, number_of_layers, "shape:", x.shape)
    _skips = _skips[::-1]
    extended_layers_scheme = layers_scheme + [bottleneck]
    extended_layers_scheme.reverse()

    # UP
    for l, number_of_layers in enumerate(layers_scheme[::-1]):
        n_filters_keep = growth_rate * extended_layers_scheme[l]
        if verbose:
            print("n_filters_keep", n_filters_keep)
        x = get_transition_up(_skips[l], block_to_upsample, n_filters_keep)
        x_up, block_to_upsample = get_dense_block(
            x, growth_rate, number_of_layers, **boilerplate
        )
        if verbose:
            print(
                "layer:",
                l,
                number_of_layers,
                "shape:",
                x.shape,
                "x_up.shape",
                x_up.shape,
            )

    # OUTPUTS
    outputs = []

    for head in heads:
        neck, _ = get_dense_block(x_up, growth_rate, neck_layers, **boilerplate)
        output = keras.layers.Conv2D(
            head["channels"],
            final_filter_size,
            activation=head["activation"],
            padding="same",
            dtype="float32",
            name=head["name"],
        )(neck)

        outputs.append(output)

    model = keras.Model(inputs=inputs, outputs=outputs, name=model_name)
    return model
