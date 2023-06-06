import pandas as pd
from datasets import load_dataset
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import layers, Sequential
import pickle

def preprocess(df):
    X = df.text
    y = df.label

    ### Let's tokenize the vocabulary
    tk = Tokenizer()
    tk.fit_on_texts(X)
    vocab_size = len(tk.word_index)
    print(f'There are {vocab_size} different words in your corpus')
    X_token = tk.texts_to_sequences(X)

    ### Pad the inputs
    X_pad = pad_sequences(X_token, dtype='float32', padding='post', truncating='post', maxlen=100)

    return X_pad, y, vocab_size


dataset = load_dataset("mediabiasgroup/mbib-base", "hate-speech")
df = pd.DataFrame(dataset['train'])


X_pad, y, vocab_size = preprocess(df)
embedding_size = 100

def initialize_model():
    model = Sequential()
    model.add(layers.Embedding(
        input_dim=vocab_size+1, # 16 +1 for the 0 padding
        input_length=100,
        output_dim=8, # 100
        mask_zero=True, # Built-in masking layer :)
    ))
    model.add(layers.LSTM(20, return_sequences=True, activation="tanh"))
    model.add(layers.Dense(1, activation="sigmoid"))
    return model

def compile_model(model):
    model.compile(loss='binary_crossentropy',
                optimizer='rmsprop',
                metrics=['accuracy'],
                )
model = initialize_model()
compile_model(model)

es = EarlyStopping(patience=5, restore_best_weights=True)
history = model.fit(X_pad, y, epochs=100, batch_size=64,
                    verbose=1, validation_split=0.2, callbacks=[es])

model.save('model.h5')
model.save_weights('model_weights.h5')
history_df = pd.DataFrame(history.history)
pickle.dump(history_df, open('history_df.pkl', 'wb'))
