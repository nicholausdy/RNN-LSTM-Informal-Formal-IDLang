# RNN-LSTM-Informal-Formal-IDLang
Implementation of a Bidirectional RNN-LSTM model to differentiate between formal and informal Indonesian sentences.

## Method
1. Vectorize sentences into a matrix of token counts. One token is one word in a sentence.
2. Utilize a single bidirectional RNN-LSTM cell to encode the input vector.
![Bidirectional RNN-LSTM](https://www.tensorflow.org/text/tutorials/images/bidirectional.png)
4. Classify the encoded sentence, where 1 = formal and 0 = informal

## Requirements
1. Ubuntu 18.04
2. Python 3.8
3. Numpy 1.22.3
4. Tensorflow 2.8.0
5. Pandas 1.4.2
6. Scikit-learn 1.0.2

## How to run
1. Install the necessary packages and ensure that the environment follows the requirements above
2. Simply run the command below
```python
python run.py
```
3. The resulting model and training log are saved inside the "result/" folder at the end of training and validation.

## Dataset
1. Created by the author of this repository himself.
2. The dataset is composed of 58 formal and 67 informal Indonesian sentences for a total of 125 sentences.
3. Dataset will be split into: training dataset (80 sentences), validation dataset (20 sentences), and test dataset (25 sentences)

## References
1. https://arxiv.org/abs/1808.03314 (paper explaining RNN and LSTM)
2. https://www.tensorflow.org/text/tutorials/text_classification_rnn (reference Bidirectional RNN-LSTM implementation)
