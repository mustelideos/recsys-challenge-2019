python 001_Preprocess_Train_Test_split.py --split_option future --processed_data_dir_name data_processed_vfuture
python 011_Features_Items.py --processed_data_dir_name data_processed_vfuture --features_dir_name nn_vfuture
python 012_Features_CTR.py --processed_data_dir_name data_processed_vfuture --features_dir_name nn_vfuture
python 013_Features_Dwell.py --processed_data_dir_name data_processed_vfuture --features_dir_name nn_vfuture
python 014_Features_General_01.py --split_option future --processed_data_dir_name data_processed_vfuture --features_dir_name nn_vfuture
python 015_Features_General_02.py --split_option future --processed_data_dir_name data_processed_vfuture --features_dir_name nn_vfuture
