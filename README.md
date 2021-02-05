# Patient Selection for Diabetes Drug Testing
This repository contains a completed cap-stone project for Udacity's "Applying AI to EHR Data" course, 
part of the AI for Healthcare Nanodegree program.  It has been reviewed by Udacity instructors and met project specifications.

**Context**: A hypothetical healthcare company is preparing for Phase III clinical trial testing for its novel diabetes drug.  The drug requires administering and patient monitoring over a duration of 5-7 days in the hospital.
Target patients are those who are likely to be in the hospital for this duration of time, so there will be no significant additional costs for drug administration and patient monitoring. 
The goal of this project is utilize Electronic Health Record (EHR) information to build a regression model that can predict the hospitalization time for a patient, and then use this model to select/filter patients for this study.

**Expected Hospitalization Time Regression Model:** 
A regression model was built to predict the expected days of hospitalization time, and then convert this to a binary prediction of whether to include or exclude that patient from the clinical trial.

This project demonstrates the importance of transforming EHR data into an appropriate data representation at the encounter level (per patient visit level), then apply filtering, preprocessing, and feature engineering of key medical code sets.  
A Deep Learning regression model was built with Tensorflow's Feature Column API and Tensorflow Probability Layers.  

The completed regression model achieved binary predication accuracy of 0.77, precision of 0.71, recall of 0.61, and F1-score of 0.66.  
It can be further optimized by maximizing precision, recall, or F1-score with trade-off between precision and recall.  
For full discussion, please read the "Model Evaluation Metrics" section of `src\student_project_EY_completed.ipynb`.  

Model biases across key demographic groups were analyzed with UChicago's Aequitas toolkit. 

### Dataset
Udacity provided a synthetic dataset(denormalized at the line level augmentation) built off of the UC Irvine Diabetes re-admission dataset. 
- [Original UCI Dataset (https://archive.ics.uci.edu/ml/datasets/Diabetes+130-US+hospitals+for+years+1999-2008)](https://archive.ics.uci.edu/ml/datasets/Diabetes+130-US+hospitals+for+years+1999-2008)

## Getting Started

1. Set up your Anaconda environment.  
2. Clone `https://github.com/ElliotY-ML/Predict_Diabetic_Patient_Hospital_Stay` GitHub repo to your local machine.
3. Open `src/student_project_EY_completed.ipynb` with Jupyer Notebook to explore EDA, feature transformations, model training, inference, and bias analysis.


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

**Install** [miniconda](http://conda.pydata.org/miniconda.html) on your machine. Detailed instructions:

- **Linux:** http://conda.pydata.org/docs/install/quick.html#linux-miniconda-install
- **Mac:** http://conda.pydata.org/docs/install/quick.html#os-x-miniconda-install
- **Windows:** http://conda.pydata.org/docs/install/quick.html#windows-miniconda-install

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
git clone https://github.com/ElliotY-ML/Predict_Diabetic_Patient_Hospital_Stay
cd Predict_Diabetic_Patient_Hospital_Stay
```

2. Create (and activate) a new environment, named `udacity-ehr-env` with Python 3.7. If prompted to proceed with the install `(Proceed [y]/n)` type y.

	- __Linux__ or __Mac__: 
	```
	conda create -n udacity-ehr-env python=3.7
	source activate udacity-ehr-env
	```
	- __Windows__: 
	```
	conda create --name udacity-ehr-env python=3.7
	activate udacity-ehr-env
	```
	
	At this point your command line should look something like: `(udacity-ehr-env) <User>:USER_DIR <user>$`. The `(udacity-ehr-env)` indicates that your environment has been activated, and you can proceed with further package installations.



6. Install a few required pip packages, which are specified in the requirements text file. Be sure to run the command from the project root directory since the requirements.txt file is there.
 
```
pip install -r requirements.txt
```


## Project Instructions

Follow the instructions in `src/student_project_EY_completed.ipynb`.

**Project Overview**
1. Project Instructions & Prerequisites
2. Learning Objectives
3. Data Preparation
4. Create Categorical Features with TF Feature Columns
5. Create Continuous/Numerical Features with TF Feature Columns
6. Build Deep Learning Regression Model with Sequential API and TF Probability Layers
7. Evaluating Potential Model Biases with Aequitas Toolkit

For more information on this project's instructions prompt, please read Udacity's original project instructions in the Project_Instructions markdown file.


## License

This project is licensed under the MIT License - see the [LICENSE.md](./LICENSE.md)
