import tensorflow as tf
from tensorflow import keras


def get_model(models, length, app_type, name):
    model_input = tf.keras.Input(shape=(256, 256, 3), name=name)

    dummy = tf.keras.layers.Lambda(lambda x: x)(model_input)

    outputs = []
    for i in models:
        constructor = getattr(app_type, i)

        x = constructor(
            include_top=False,
            weights="imagenet",
            input_shape=(256, 256, 3),
            pooling="avg",
        )(dummy)

        x = tf.keras.layers.Dense(1, activation="sigmoid")(x)
        outputs.append(x)

    model = tf.keras.Model(model_input, outputs, name=name)
    model.summary()
    return model


def compile_new_model(
    models, app_type, name, strategy, label_smoothing=None, learning_rate=None
):
    length = len(models)
    with strategy.scope():
        model = get_model(models=models, length=length, app_type=app_type, name=name)
        if label_smoothing:
            losses = [
                tf.keras.losses.BinaryCrossentropy(label_smoothing=label_smoothing)
                for i in range(length)
            ]
        else:
            losses = [tf.keras.losses.BinaryCrossentropy() for i in range(length)]
        if learning_rate:
            optimizers = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        else:
            optimizers = tf.keras.optimizers.Adam()

        model.compile(
            optimizer=optimizers,
            loss=losses,
            metrics=[
                tf.keras.metrics.Precision(name="Prec"),
                tf.keras.metrics.Recall(name="Rec"),
                tf.keras.metrics.Accuracy(name="Accuracy"),
            ],
        )

    return model


if __name__ == "__main__":
    compile_new_model()
