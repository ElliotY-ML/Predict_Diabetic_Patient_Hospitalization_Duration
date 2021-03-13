# Patient Selection for Diabetes Drug Testing
This repository contains a completed cap-stone project for Udacity's "Applying AI to EHR Data" course, 
part of the AI for Healthcare Nanodegree program.  It has been reviewed by Udacity instructors and met project specifications.

**Project Premise**  
A hypothetical healthcare company is preparing for Phase III clinical trial testing for its novel diabetes drug.  The drug requires administering and patient monitoring over a duration of 5-7 days in the hospital.
Target patients are those who are likely to be in the hospital for this duration of time, so there will be no significant additional costs for drug administration and patient monitoring. 
The goal of this project is utilize Electronic Health Record (EHR) information to build a regression model that can predict the hospitalization time for a patient, and then use this model to select/filter patients for this study.

**Regression Model for Expected Hospitalization Duration**   
A deep learning regression model was built to predict the expected days of hospitalization duration, and then convert this to a binary prediction of whether to include or exclude that patient from the clinical trial.

This project utilizes EHR data by transforming line-level data into an appropriate data representation at the encounter level (per patient visit level), and then apply filtering, preprocessing, and feature engineering of key medical code sets. 
Tensorflow Feature Column API was used to prepare features and Tensorflow Probability Layers were used to create the regression model.  

The completed regression model achieved binary predication accuracy of 0.77, precision of 0.71, recall of 0.61, and F1-score of 0.66. It can be further optimized by maximizing precision, recall, or F1-score with trade-off between precision and recall.  
For full discussion, please read the "Model Evaluation Metrics" section of `src\student_project_EY_completed.ipynb`.  

To understand model biases across key demographic groups, model predictions were analyzed with UChicago's Aequitas toolkit. 

### Dataset
Udacity provided a synthetic dataset(denormalized at the line level augmentation) built off of the UC Irvine Diabetes re-admission dataset.  
The dataset can be found in `/src/data/final_project_dataset.csv`.

**References**  
[Original UCI Dataset (https://archive.ics.uci.edu/ml/datasets/Diabetes+130-US+hospitals+for+years+1999-2008)](https://archive.ics.uci.edu/ml/datasets/Diabetes+130-US+hospitals+for+years+1999-2008)

## Getting Started

1. Set up your Anaconda environment.  
2. Clone `https://github.com/ElliotY-ML/Predict_Diabetic_Patient_Hospitalization_Duration.git` GitHub repo to your local machine.
3. Open `src/student_project_EY_completed.ipynb` with Jupyer Notebook to explore EDA, feature transformations, feature columns, model training, inference, and bias analysis.


### Dependencies
Using Anaconda consists of the following:

1. Install [`miniconda`](http://conda.pydata.org/miniconda.html) on your computer, by selecting the latest Python version for your operating system. If you already have `conda` or `miniconda` installed, you should be able to skip this step and move on to step 2.
2. Create and activate * a new `conda` [environment](http://conda.pydata.org/docs/using/envs.html).

\* Each time you wish to work on any exercises, activate your `conda` environment!

---

## 1. Installation

**Download** the latest version of `miniconda` that matches your system.

|        | Linux | Mac | Windows | 
|--------|-------|-----|---------|
| 64-bit | [64-bit (bash installer)][lin64] | [64-bit (bash installer)][mac64] | [64-bit (exe installer)][win64]
| 32-bit | [32-bit (bash installer)][lin32] |  | [32-bit (exe installer)][win32]

[win64]: https://repo.continuum.io/miniconda/Miniconda3-latest-Windows-x86_64.exe
[win32]: https://repo.continuum.io/miniconda/Miniconda3-latest-Windows-x86.exe
[mac64]: https://repo.continuum.io/miniconda/Miniconda3-latest-MacOSX-x86_64.sh
[lin64]: https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
[lin32]: https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86.sh

**Install** [miniconda](https://docs.conda.io/en/latest/miniconda.html) on your machine. Detailed instructions:

- **Linux:** https://docs.conda.io/en/latest/miniconda.html#linux-installers
- **Mac:** https://docs.conda.io/en/latest/miniconda.html#macosx-installers
- **Windows:** https://docs.conda.io/en/latest/miniconda.html#windows-installers

## 2. Create and Activate the Environment

For Windows users, these following commands need to be executed from the **Anaconda prompt** as opposed to a Windows terminal window. For Mac, a normal terminal window will work. 

#### Git and version control
These instructions also assume you have `git` installed for working with Github from a terminal window, but if you do not, you can download that first with the command:
```
conda install git
```

**Create local environment**

1. Clone the repository, and navigate to the downloaded folder. This may take a minute or two to clone due to the included image data.
```
git clone https://github.com/ElliotY-ML/Predict_Diabetic_Patient_Hospitalization_Duration.git
cd Predict_Diabetic_Patient_Hospitalization_Duration
```

2. Create (and activate) a new environment, named `udacity-ehr-env` with Python 3.8. If prompted to proceed with the install `(Proceed [y]/n)` type y.

	- __Linux__ or __Mac__: 
	```
	conda create -n udacity-ehr-env 
	source activate udacity-ehr-env
	```
	- __Windows__: 
	```
	conda create --name udacity-ehr-env 
	activate udacity-ehr-env
	```
	
	At this point your command line should look something like: `(udacity-ehr-env) <User>:USER_DIR <user>$`. The `(udacity-ehr-env)` indicates that your environment has been activated, and you can proceed with further package installations.



6. Install a few required pip packages, which are specified in the requirements text file. Be sure to run the command from the project root directory since the requirements.txt file is there.
 
```
pip install -r requirements.txt
```


## Repository Instructions  

Udacity's original project instructions can be read in the `Project_Instructions.md` file.

**Project Overview**

1. Project Instructions & Prerequisites
2. Learning Objectives
3. Data Preparation and Exploratory Data Analysis
4. Create Categorical Features with TensorFlow Feature Columns
5. Create Continuous/Numerical Features with TensorFlow Feature Columns
6. Build Deep Learning Regression Model with Sequential API and TensorFlow Probability Layers
7. Evaluating Potential Model Biases with Aequitas Toolkit


Begin by opening `/src/student_project_EY_completed.ipynb` with Jupyter Notebook.  

Inputs:  
-  Udacity Dataset: `src/data/final_project_dataset.csv`  
-  Admission Type ID: `src/data_schema_references/IDs_mapping.csv`
-  NDC Codes to Drugs Lookup Table: `src/data_schema_references/ndc_lookup_table.csv`
-  Dataset Schema: `src/data_schema_references/project_data_schema.csv`
-  NDC Codes to Drugs Lookup Tabble (copy): `src/medication_lookup_tables/final_ndc_lookup_table`

Output:  
-  Trained Deep learning regression model with TensorFlow Probability Layers in notebook
-  Predictions output in `/out/pred_test_df3.csv`

1.  Data preparation begins in section 3.  The project dataset is loaded into a pandas DataFrame.  There are medical code reference tables in `src/data_schema_references` that translate medical and medicine codes into descriptions.  These are also loaded into dataframes.
2.  An exploratory data analysis proceeds to understand the data and demographics.  
3.  The dataset is then reduced from the line level into an aggregated encounter level.  In other words, all indiviual medical codes are aggregated by individual patient visits.
4.  Select categorical and numerical features to use for the model.
5.  Split dataset into a 60%/20%/20% train/validation/test split and ensure that the demographics are reflective of the overall dataset.  The `patient_dataset_splitter` function was completed in `student_utils.py` module.
6.  Use TensorFlow Feature Columns API to create categorical features and embedding columns for each feature.  The `create_tf_categorical_feature_cols` function was completed in `student_utils.py` module.
7.  Use TensorFlow Feature Columns API to create numeric features.  The `create_tf_numeric_feature` function was completed in `student_utils.py` module.
8.  Build and train a deep learning regression model with Keras DenseFeatures and TensorFlow Probability Layers.
9.  Convert regression output to classification for patient selection
9.  Use scikit-learn `classification_report` and `confusion_matrix` functions to compare patient selection performance of trained model against actual hospitalization durations.  
10. Evaluate potential model biases with Aequitas toolkit.  Visualizations show if there are model biases for gender and race demographics.  


## License

This project is licensed under the MIT License - see the [LICENSE.md](./LICENSE.md)
