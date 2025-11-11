import tensorflow as tf
from transformers import TFResNetModel
from config import IMAGE_SIZE, NUM_CLASSES, EPOCHS, MODEL_NAME

def build_and_train_model(train_ds, val_ds):
    base_model = TFResNetModel.from_pretrained(MODEL_NAME)
    
    # Freezing the backbone/original ResNet architecture layers for Transfer Learning
    base_model.trainable = False

    input_layer = tf.keras.Input(shape = (IMAGE_SIZE, IMAGE_SIZE, 3), name = "input_image")

    # Transposing to CHW format expected by Hugging Face ResNet
    x = tf.keras.layers.Lambda(lambda x: tf.transpose(x, [0, 3, 1, 2]))(input_layer)

    # Getting pooled output from ResNet: shape (None, 2048, 1, 1)
    # This is the main ResNet architecture layer. We are only modifying the output layer as per our image classes/indices, and the flatten layer
    # training=False does NOT freeze weights, it sets the forward pass to inference mode (disables dropout and causes BatchNorm layers to use population statistics rather than updating them)
    x = base_model(pixel_values = x, training = False).pooler_output

    # Flattenning to (None, 2048)
    x = tf.keras.layers.Flatten()(x)

    # Classification head
    x = tf.keras.layers.Dense(256, activation = "relu")(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    output_layer = tf.keras.layers.Dense(NUM_CLASSES, activation = "softmax")(x)

    model = tf.keras.Model(inputs = input_layer, outputs = output_layer)

    model.compile(
        optimizer = tf.keras.optimizers.Adam(1e-4),
        loss = 'categorical_crossentropy',
        metrics = ['accuracy']
    )

    model.summary()
    
    model.fit(
        train_ds,
        validation_data = val_ds,
        epochs=EPOCHS
    )
    
    model.save_weights("saved_model/resnet_weights.h5")
    
    return model