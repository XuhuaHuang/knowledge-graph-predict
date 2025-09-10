# KG-Predict
## Comprehensive Drug Repositioning Framework

This comprehensive framework integrates a detailed knowledge base with advanced artificial intelligence (AI) models to facilitate the identification of potential therapeutic candidates for Parkinson disease.

## Requirements:
Python(version >= 3.6)
pytorch(version>=1.4.0)
ordered_set(version>=3.1)
numpy(version>=1.16.2)
torch_scatter(version>=2.0.4)
scikit_learn(version>=0.21.1)

We highly recommend you use Conda for package management.

If you have an existing conda virtual environment that you would like to use, (eg. `base`), you can install the required dependency with the following:

```bash
conda install pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 -c pytorch -c nvidia
```

If you are creating a new virtual environment, you can do so by running the following:

```bash
conda env create -f environment.yml
conda activate knowledge-graph-predict
```

Verify the setup with the following:

```bash
>>> import torch
>>> torch.__version__
'2.5.1.post303'
>>> torch.cuda.is_available()
True
>>> torch.version.cuda
'12.0'
```

If you are using the provided `launch.json` inside `Visual Studio Code`, use the first available GPU node `0` like the following:

```bash
python main.py -data test_data -gpu 0 -name test_model -epoch 500
```


## Model Training:
1)Create a folder "test_data" under folder "data" and move training data, valid data, and test data to the folder. 

2)Use the following command to train the model, the model will be named as "test_model" and saved in the directory "model_saved".
```python
  python main.py -data test_data -gpu 1 -name test_model -epoch 500
```

## Target-based drug Prediction:
1)Create a test file named "ad_pre.txt" and move the file to the folder "test_data".

2)Run the following command, predicting results will be saved in the file "pre_results.txt".
```python
  python test.py -data test_data -gpu 1 -name test_model -save_result pre_results.txt -test_file ad_pre.txt
```

### Parameter Note:

-data : the directory of training and testing data

-gpu : the GPU to use

-name : the name of the model snapshot (used for storing model parameters)

-epoch : the number of epochs

-save_result : the filename that is used to store test results

-test_file : the name of testing file
