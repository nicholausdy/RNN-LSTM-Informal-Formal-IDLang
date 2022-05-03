from data_loader import DataLoader
from rnn_lstm import RNNLSTM

import pandas as pd

# data variables
filename = './dataset.csv'
test_proportion = 0.2
val_proportion = 0.2

# model hyperparameters
embed_dim = 24
batch_size = 8
epochs = 100

# test parameters
test_batch_size = 5

# saving model
model_name = "./result/saved_model"
csv_file = "./result/training_log.csv"

def pipeline():
    data = DataLoader(filename, test_proportion, val_proportion)

    vocab_size = data.x_train.shape[1]
    x_train_arr = data.x_train.toarray()
    x_val_arr = data.x_val.toarray()
    x_test_arr = data.x_test.toarray()

    # train and validate model
    rnn_lstm = RNNLSTM(vocab_size, embed_dim)
    rnn_lstm.model.compile(optimizer="adam", loss="binary_crossentropy", metrics="accuracy")
    rnn_lstm.model.summary()

    history = rnn_lstm.model.fit(
        x_train_arr, data.y_train, batch_size, epochs, shuffle=True,
        validation_data = (x_val_arr, data.y_val),
    )

    # test model
    test_result = rnn_lstm.model.evaluate(x_test_arr, data.y_test, test_batch_size)
    print("test loss, test accuracy:", test_result)

    # save model
    rnn_lstm.model.save(model_name)

    # save training log
    history_df = pd.DataFrame(history.history)
    with open(csv_file, mode='w') as f:
        history_df.to_csv(f)

pipeline()

