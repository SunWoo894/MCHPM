# model/proposed.py
import numpy as np
import tensorflow as tf

from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import (
    Input,
    Dense,
    Dropout,
    Add,
    Concatenate,
    Multiply,
    Flatten,
    Layer,
    Lambda,
    LayerNormalization,
    MultiHeadAttention,
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

class GateComplement(Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, inputs):
        return 1.0 - inputs

    def get_config(self):
        config = super().get_config()
        return config


# ============================================================
# 1) Blocks
# ============================================================

def point_wise_feed_forward_network(d_model: int, dff: int) -> Sequential:
    """Transformer-style FFN."""
    return Sequential(
        [
            Dense(dff, activation="linear"),
            Dense(d_model),
        ],
        name="PWFFN",
    )


class CoAttentionBlock(Layer):
    """
    Co-Attention Block:
    - (optional) expand dims to [B, 1, D]
    - MultiHeadAttention(query=text, key/value=image)
    - Residual + LayerNorm
    - FFN + Residual + LayerNorm
    - Flatten to [B, D]
    """

    def __init__(
        self,
        num_heads: int,
        d_model: int,
        key_dim: int,
        dff: int,
        dropout_rate: float = 0.1,
        epsilon: float = 1e-6,
        expand_dims: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.expand_dims = expand_dims

        self.mha = MultiHeadAttention(
            num_heads=num_heads,
            key_dim=key_dim,
            value_dim=key_dim,
            name=f"{self.name}_MHA" if self.name else "CoAtt_MHA",
        )
        self.dropout = Dropout(dropout_rate)
        self.add = Add()
        self.ln = LayerNormalization(epsilon=epsilon)

        self.ffn = point_wise_feed_forward_network(d_model=d_model, dff=dff)
        self.flatten = Flatten()

    def call(self, inputs, training=None):
        text_input, image_input = inputs

        if self.expand_dims:
            text_input = tf.expand_dims(text_input, axis=1)
            image_input = tf.expand_dims(image_input, axis=1)

        # Co-attention: query=text, key/value=image
        attn = self.mha(query=text_input, value=image_input, key=image_input, training=training)
        attn = self.dropout(attn, training=training)
        attn = self.add([attn, text_input])
        attn = self.ln(attn)

        # FFN
        ffn_out = self.ffn(attn, training=training)
        ffn_out = self.dropout(ffn_out, training=training)
        ffn_out = self.add([ffn_out, attn])
        ffn_out = self.ln(ffn_out)

        return self.flatten(ffn_out)


# ============================================================
# 2) Proposed Model (Multimodal RHP variant)
# ============================================================

def RHP(
    feature_dimension: int,
    num_heads: int,
    dropout: float = 0.1,
    learning_rate: float = 1e-4,
    key_dim: int = 64,
    dff: int = 2048,
) -> Model:
    """
    Convenience wrapper.
    """
    return build_rhp(
        feature_dimension=feature_dimension,
        num_heads=num_heads,
        dropout=dropout,
        learning_rate=learning_rate,
        key_dim=key_dim,
        dff=dff,
    )


def build_rhp(
    feature_dimension: int,
    num_heads: int,
    dropout: float = 0.1,
    learning_rate: float = 1e-4,
    key_dim: int = 64,
    dff: int = 2048,
) -> Model:

    # -----------------------------
    # Input layer
    # -----------------------------
    bert_input = Input(shape=(768,), name="bert_input")
    vgg16_input = Input(shape=(4096,), name="vgg16_input")
    text_per_input = Input(shape=(4,), name="text_peripheral_input")
    image_per_input = Input(shape=(4,), name="image_peripheral_input")

    # -----------------------------
    # Local / Global Layer
    # -----------------------------
    bert = LayerNormalization(axis=-1, epsilon=1e-6, name="LN_bert")(bert_input)
    bert = Dense(feature_dimension, activation="relu", name="Dense_Layer_bert")(bert)
    bert = Dropout(dropout, name="Dropout_bert")(bert)

    vgg16 = LayerNormalization(axis=-1, epsilon=1e-6, name="LN_vgg16")(vgg16_input)
    vgg16 = Dense(feature_dimension, activation="relu", name="Dense_Layer_vgg16")(vgg16)
    vgg16 = Dropout(dropout, name="Dropout_vgg16")(vgg16)

    # Text Peripheral
    text_per = Dense(4, activation="elu", name="Dense_text_per_1")(text_per_input)
    text_per = Dense(16, activation="elu", name="Dense_text_per_2")(text_per)
    text_per = Dense(64, activation="elu", name="Dense_text_per_3")(text_per)
    text_per = Dense(256, activation="elu", name="Dense_text_per_4")(text_per)
    text_per = Dense(feature_dimension, activation="elu", name="Dense_text_per_proj")(text_per)
    text_per = LayerNormalization(axis=-1, epsilon=1e-6, name="LN_text_per")(text_per)

    # Image Peripheral
    image_per = Dense(4, activation="elu", name="Dense_image_per_1")(image_per_input)
    image_per = Dense(16, activation="elu", name="Dense_image_per_2")(image_per)
    image_per = Dense(64, activation="elu", name="Dense_image_per_3")(image_per)
    image_per = Dense(256, activation="elu", name="Dense_image_per_4")(image_per)
    image_per = Dense(feature_dimension, activation="elu", name="Dense_image_per_proj")(image_per)
    image_per = LayerNormalization(axis=-1, epsilon=1e-6, name="LN_image_per")(image_per)

    # -----------------------------
    # Co-Attention (two blocks)
    # -----------------------------
    co_att_text_block = CoAttentionBlock(
        num_heads=num_heads,
        d_model=feature_dimension,
        key_dim=key_dim,
        dff=dff,
        dropout_rate=dropout,
        name="CoAtt_Text",
    )
    co_att_image_block = CoAttentionBlock(
        num_heads=num_heads,
        d_model=feature_dimension,
        key_dim=key_dim,
        dff=dff,
        dropout_rate=dropout,
        name="CoAtt_Image",
    )

    co_attention_text = co_att_text_block((bert, text_per))
    co_attention_image = co_att_image_block((vgg16, image_per))

    # -----------------------------
    # Feature Fusion (gate)
    # -----------------------------
    text_tanh = Dense(feature_dimension, activation="tanh", name="Text_Tanh")(co_attention_text)
    image_tanh = Dense(feature_dimension, activation="tanh", name="Image_Tanh")(co_attention_image)

    combined = Concatenate(name="Concat_text_image")([co_attention_text, co_attention_image])
    gate = Dense(feature_dimension, activation="sigmoid", name="Gate_Text_Image")(combined)

    gated_text = Multiply(name="Gated_Text")([gate, text_tanh])
    # NOTE: (1 - gate) 연산은 텐서 연산으로 처리
    gate_comp = GateComplement(name="Gate_Complement")(gate)

    gated_image = Multiply(name="Gated_Image")(
        [gate_comp, image_tanh]
    )


    fused = Add(name="Text_Image_Fusion")([gated_text, gated_image])

    # -----------------------------
    # Helpfulness Prediction Layer
    # -----------------------------
    dense_1 = Dense(64, activation="linear", name="Dense_layer_1")(fused)
    dense_2 = Dense(32, activation="linear", name="Dense_layer_2")(dense_1)
    dense_3 = Dense(16, activation="linear", name="Dense_layer_3")(dense_2)
    output_layer = Dense(1, activation="linear", name="Output_layer")(dense_3)

    model = Model(
        inputs=[bert_input, vgg16_input, text_per_input, image_per_input],
        outputs=output_layer,
        name="RHP_Multimodal",
    )

    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss="mean_squared_error",
        metrics=["mean_absolute_error", "mean_squared_error"],
    )

    return model


# ============================================================
# 3) tf.data.Dataset: get_data_loader
# ============================================================

def get_data_loader(args: dict, df, shuffle: bool = True):
    """
    df columns expected:
      - 'bert' (list/np.array shape [768])
      - 'vgg16' (list/np.array shape [4096])
      - 'text_peripheral' (list/np.array shape [4])
      - 'image_peripheral' (list/np.array shape [4])
      - 'log_vote' (float)
    """

    # numpy arrays
    bert_arr = np.stack(df["bert"].to_list()).astype("float32")
    vgg_arr = np.stack(df["vgg16"].to_list()).astype("float32")

    # 이 컬럼들은 종종 (1,4) 형태로 들어있어서 stack으로 맞추는 게 안전
    text_per_arr = np.stack(df["text_peripheral"].to_list()).astype("float32").reshape(-1, 4)
    image_per_arr = np.stack(df["image_peripheral"].to_list()).astype("float32").reshape(-1, 4)

    labels = df["log_vote"].to_numpy().astype("float32")

    x_dict = {
        "bert_input": bert_arr,
        "vgg16_input": vgg_arr,
        "text_peripheral_input": text_per_arr,
        "image_peripheral_input": image_per_arr,
    }

    batch_size = int(args.get("batch_size", 128))
    seed = int(args.get("seed", 42))

    ds = tf.data.Dataset.from_tensor_slices((x_dict, labels))

    if shuffle:
        ds = ds.shuffle(buffer_size=len(df), seed=seed)

    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


# ============================================================
# 4) Trainer & Tester
# ============================================================

def rhp_trainer(
    args: dict,
    model: Model,
    train_loader,
    val_loader,
    best_model_path: str,
):
    epochs = int(args.get("num_epochs", 100))
    patience = int(args.get("patience", 5))

    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=patience,
            mode="auto",
            restore_best_weights=True,
            verbose=1,
        ),
        ModelCheckpoint(
            filepath=best_model_path,
            monitor="val_loss",
            save_best_only=True,
            save_weights_only=False,
            verbose=1,
        ),
    ]

    history = model.fit(
        train_loader,
        validation_data=val_loader,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1,
    )
    return history


def rhp_tester(
    args: dict,
    model: Model,
    test_loader,
):
    preds = model.predict(test_loader).reshape(-1)

    trues_list = []
    for _, y in test_loader:
        trues_list.append(y.numpy())
    trues = np.concatenate(trues_list, axis=0).reshape(-1)

    return preds, trues
