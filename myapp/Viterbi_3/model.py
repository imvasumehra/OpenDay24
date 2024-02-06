import json
import tensorflow as tf
from keras_self_attention import SeqSelfAttention


def getParams():
    json_file_path = "/home3/luharj/AiSteth/Code/Segmentation/model/model/BiLSTM.json"
    weights_file_path = "/home3/luharj/AiSteth/Code/Segmentation/Viterbi_2/model/best_model/Checkpoint-099-0.643-0.723.h5"
    return json_file_path, weights_file_path

def Hybrid_S1S2_BiLSTM_Attention(input_shape, padding_value, name=None):
    inp = tf.keras.Input(shape=input_shape)
    
    BiLSTM = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(80, return_sequences=True,
                                                                dropout=0.2, recurrent_dropout=0.2))(inp)
    
    SelfAttention = SeqSelfAttention(attention_type=SeqSelfAttention.ATTENTION_TYPE_MUL,
                                     kernel_regularizer=tf.keras.regularizers.l2(1e-4),
                                     bias_regularizer=tf.keras.regularizers.l1(1e-4),
                                     attention_regularizer_weight=1e-4,
                                     name='Attention')(BiLSTM)
    
    dense = tf.keras.layers.Dense(160, activation='tanh', kernel_initializer='he_normal')(SelfAttention)
    dense = tf.keras.layers.Dense(80, activation='tanh', kernel_initializer='he_normal')(dense)
    
    final_out = tf.keras.layers.Dense(4, activation='softmax', kernel_initializer='he_normal',
                                      name='Dense')(dense)
    
    model = tf.keras.Model(inputs=[inp], outputs=[final_out])
    
    return model


def getModel():
    json_file_path, weights_file_path = getParams()
    input_shape = (150, 41)
    model = Hybrid_S1S2_BiLSTM_Attention(input_shape, padding_value = 0)
    model.load_weights(weights_file_path)
    # print("Success")
    return model


if __name__ == "__main__":
    model = getModel()
    