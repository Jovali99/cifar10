# Perform empirical studies of LiRA & RMIA on classification models to evaluate Forgotten by Design
To install all required packages run the install.sh script
## Windows

	./install.sh

## Linux

	source install.sh

The script will run 

	pip install .
	
then install a specific version of torch and torchvision

# Running the project pipeline
The following steps will show how to run the entire pipeline for performing empirical studies

1. Setup desired train and audit configs
2. (Optional) Optimize classification model hyperparameters using optuna in the optimize_model notebook
3. Train the target baseline model using train_baseline_target_model notebook
4. Train shadow models using train_shadow_models notebook
5. (Optional) Optimize the Forgotten by Design parameters using ...
6. Train FbD target model using train_fbd_target_model notebook
7. Visualize the audits using visualize_audit notebook
