import numpy as np
import tensorflow as tf

import main

tf.keras.backend.clear_session()


def create_training_data(samples, image_size, max_radius, noise_level):
    noisy_images = np.zeros((samples, image_size, image_size, 1))
    orig_images = np.zeros((samples, image_size, image_size, 1))

    for i in range(samples):
        _, image, orig_image = main.noisy_circle(image_size, max_radius, noise_level)
        # preprocess the image by normalization
        image /= np.amax(image)
        noisy_images[i, ...] = np.expand_dims(image, axis=-1)
        orig_images[i, ...] = np.expand_dims(orig_image, axis=-1)

    return noisy_images, orig_images


def Denoisemodel(img_size):

    inp = tf.keras.Input(shape=(img_size, img_size, 1))

    x = tf.keras.layers.Conv2D(32, 3, padding="same", name="Conv1")(inp)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu", name="rELU1")(x)
    x = tf.keras.layers.MaxPooling2D((2, 2), name="MaxPool1")(x)
    encode = tf.keras.layers.Conv2D(64, 3, padding="same", name="Conv2")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu", name="rELU2")(x)
    x = tf.keras.layers.MaxPooling2D((2, 2), name="MaxPool2")(x)
    encode = tf.keras.layers.Conv2D(64, 3, padding="same", name="Conv3")(x)

    # at this point context representation in lower dim

    x = tf.keras.layers.Conv2D(64, 3, padding="same", name="Conv4")(encode)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu", name="rELU4")(x)
    x = tf.keras.layers.UpSampling2D((2, 2), name="UpSample1")(x)
    x = tf.keras.layers.Conv2D(32, 3, padding="same", name="Conv5")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu", name="rELU5")(x)
    x = tf.keras.layers.UpSampling2D((2, 2), name="UpSample2")(x)
    x = tf.keras.layers.Conv2D(1, 3, padding="same", name="Conv7")(x)

    decode = tf.keras.layers.Activation("sigmoid", name="DecodeSigmoid")(x)

    model = tf.keras.Model(inputs=inp, outputs=decode, name="autoencoder")

    return model


if __name__ == "__main__":

    train_data, train_label = create_training_data(
        5000, main.IMAGE_SIZE, main.MAX_RAD, main.N_LVL
    )
    val_data, val_label = create_training_data(
        500, main.IMAGE_SIZE, main.MAX_RAD, main.N_LVL
    )
    model = Denoisemodel(main.IMAGE_SIZE)
    model.summary()
    # loss_typ = tf.keras.losses.MeanSquaredError()
    model.compile(optimizers="rmsprop", loss="binary_crossentropy")
    model.fit(
        train_data,
        train_label,
        epochs=20,
        batch_size=64,
        shuffle=True,
        validation_data=(val_data, val_label),
        callbacks=[
            tf.keras.callbacks.TensorBoard(
                log_dir="./", histogram_freq=0, write_graph=False
            )
        ],
    )

    # Save the model
    tf.keras.models.save_model(
        model, filepath="./noise_detection_autoenc.h5", save_format="h5"
    )

