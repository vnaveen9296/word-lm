# Telugu Language Modeling and Predictive Typing
Trying Telugu LM and Predictive Typing using PyTorch....

I have created a mini corpus of Teugu to try out LM based on LSTMs using PyTorch. Some samples are (see `telugu.txt`):
* ఎలా ఉన్నావు
* ఏం చేస్తున్నావు
* నేను మీటింగ్లో ఉన్నాను

**Key Concepts:**
* Dataset preparation (PyTorch Datasets and DataLoaders)
* Creating Model by inheriting from `torch.nn.Module` class
* Using `nn.Embedding` and `nn.LSTM` layers
* Writing a Predict method
* Loading and saving models

