import pandas as pd
from datasets import load_dataset
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import layers, Sequential
import pickle



dataset = load_dataset("mediabiasgroup/mbib-base", "hate-speech")
df = pd.DataFrame(dataset['train'])

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
    X_pad = pad_sequences(X_token, dtype='float32', padding='post')

    return X_pad, y, vocab_size

X_pad, y, vocab_size = preprocess(df)
### Let's build the neural network now
# Size of your embedding space = size of the vector representing each word
embedding_size = 100

model = Sequential()
model.add(layers.Embedding(
    input_dim=vocab_size+1, # 16 +1 for the 0 padding
    output_dim=embedding_size, # 100
    mask_zero=True, # Built-in masking layer :)
))

model.add(layers.LSTM(20))
model.add(layers.Dense(1, activation="sigmoid"))
model.summary()

es = EarlyStopping(patience=5, restore_best_weights=True)
model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

history = model.fit(X_pad, y, epochs=100, batch_size=64, verbose=1, validation_split=0.2, callbacks=[es])

model.save('model.h5')
model.save_weights('model_weights.h5')
history_df = pd.DataFrame(history.history)
pickle.dump(history_df, open('history_df.pkl', 'wb'))
