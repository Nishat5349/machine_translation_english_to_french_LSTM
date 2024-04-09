# English to French Machine Translation using LSTM

## Introduction:
Machine Translation, the process of automatically translating text from one language to another, has become an essential tool in today's globalized world. In this project, we focus on the task of translating English text into French using advanced machine learning techniques. By harnessing the power of Neural Machine Translation (NMT) models, we aim to bridge linguistic barriers and facilitate seamless communication across languages.

## Dataset Collection:
To train our English-to-French translation model, we collected the dataset from "https://www.manythings.org/anki/". This dataset comprises pairs of English sentences and their corresponding French translations. By leveraging this comprehensive dataset, we ensure that our model learns to accurately capture the nuances and intricacies of both languages, enabling high-quality translations.

## Dependencies
- numpy
- keras
- matplotlib

## Setting Up the Environment
Make sure to have the required dependencies installed. You can install them using pip:
```bash
pip install numpy keras matplotlib
```

## Usage
1. Clone this repository to your local machine.
2. Ensure you have the necessary dataset downloaded or collected.
3. Run the provided Python script or Jupyter Notebook to train the translation model.

## Model Architecture:
The model architecture consists of an Encoder-Decoder LSTM network with an embedding layer.Our machine translation model is built upon state-of-the-art Neural Machine Translation (NMT) architecture. It employs a sequence-to-sequence (Seq2Seq) framework with attention mechanisms, allowing the model to effectively encode input English sentences and generate corresponding French translations. The architecture consists of encoder and decoder components, each comprising multiple layers of recurrent or transformer neural networks. Through extensive training on the collected dataset, our model learns to generate fluent and contextually accurate translations, offering a powerful tool for cross-lingual communication. The model utilizes a Long Short-Term Memory (LSTM) network architecture implemented using Keras.
Below is a brief overview of the model: 

![image/png](https://cdn-uploads.huggingface.co/production/uploads/65ab673cd2adc31ee3273db4/-2FQtc0SQGdVQodGNlZHF.png)

In this project, we utilize GloVe word embeddings for English sentences (inputs) and custom word embeddings for translated French sentences (outputs). GloVe embeddings provide a pre-trained set of word vectors, while custom embeddings are trained specifically for our task.
In the model architecture, the input placeholder for the encoder (input_1) is embedded and passed through the encoder LSTM (lstm_1), generating outputs including the hidden layer and cell state. These states are then passed to the decoder LSTM (lstm_2) along with the output sentences tokenized at the start (input_2). The decoder LSTM processes the input and generates predictions, which are passed through a dense layer to produce the final output.


### Compiling and Training the Model
We train the model using the provided input and output sequences along with their corresponding targets. An early stopping callback is employed to prevent overfitting by monitoring validation loss. The model is trained over 8 epochs with a batch size of 64 and a validation split of 20%.
```python
# Compile the model
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit([encoder_inputs, decoder_inputs], decoder_targets,
                    batch_size=64,
                    epochs=8,
                    validation_split=0.2)
```
A graph illustrating the training and validation loss and accuracy is generated during the training process, providing insights into the model's performance and convergence.

![image/png](https://cdn-uploads.huggingface.co/production/uploads/65ab673cd2adc31ee3273db4/VsqGk16iW0ftfCvDbRKK0.png)

![image/png](https://cdn-uploads.huggingface.co/production/uploads/65ab673cd2adc31ee3273db4/oF2UiyrBiokjQgzMJErGm.png)


## Contributor
NISHAT TASNIM (nishattasnim296318@gmail.com)

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---


