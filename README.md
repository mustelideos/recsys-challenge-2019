# recsys-challenge-2019
[An Attentive RNN Model for Session-based and Context-aware Recommendations](https://drive.google.com/file/d/1R5LGJgbOkSDCB_dbXZg8-elMhVEbhiVD/view)

7th place solution @ [RecSys 2019 Data Challenge](http://www.recsyschallenge.com/2019/).

## Setup instructions

1. Clone this repo:
    ```bash
    git clone https://github.com/hugoguh/saca_recsys
    ```
2. Copy/move/download the competion data files to `./data/`

3. Install [Anaconda](https://www.anaconda.com/download/) (**Python 3 version**).

4. Install the environment:
    ```bash
    cd recsys-challenge-2019/
    conda env create --file environment_recsys-challenge-2019.yml
    ```
5. Activate the environment:
    ```bash
    conda activate recsys-challenge-2019
    ```
6. Run all features and model:
    ```bash
    cd scripts
    sh run_all_future.sh
    ```
    
## File structure
    ```.
    ├── LICENSE
    ├── README.md
    ├── data
    ├── environment_recsys-challenge-2019.yml
    ├── features
    ├── helpers
    │   ├── __init__.py
    │   ├── feature_helpers.py
    │   └── train_val_split_helpers.py
    ├── models
    ├── predictions
    └── scripts
        ├── 001_Preprocess_Train_Test_split.py
        ├── 011_Features_Items.py
        ├── 012_Features_CTR.py
        ├── 013_Features_Dwell.py
        ├── 014_Features_General_01.py
        ├── 015_Features_General_02.py
        ├── 021_Run_Model.py
        ├── run_all_default.sh
        ├── run_all_future.sh
        ├── run_features_default.sh
        └── run_features_future.sh
    ```
