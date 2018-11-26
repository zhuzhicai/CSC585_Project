# CSC585 Project 

## 1.Requirements
- python (version 3, issues reported with version 2)
- wget, unzip (for download.sh)
- python packages: numpy, spacy, torch and matplotlib

## 2.Running the Code
### 2.0 Install the dependencies:
- `wget` comes with most Linux distributions. If you use OS X, you can install wget with Homebrew:
```
brew install wget
``` 
- If you use the dockerfile provided, you **do not need to** install the python packages, otherwise you can install all python packages with `pip3 install package_name==version`

### 2.1 Download the dependencies: SQuAD2.0 dataset, glove vectors and counter-fitting vectors
```
chmod +x download.sh
./download.sh
``` 

### 2.2 Training model
Use the following command:
```
python3 train.py data_file w2v_file
``` 

## Citation
We use the function to convert token to index
