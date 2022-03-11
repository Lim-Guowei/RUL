### Remaining Useful Life Prediction for Commercial Modular Aero-Propulsion System Simulation (C-MAPSS) dynamical model dataset

#### System information
| System                | Version                                                                                        |
|:---------------------:|:----------------------------------------------------------------------------------------------:|
| Operating system      | Ubuntu 20.04                                                                                   |
| GPU                   | Nvidia GeForce RTX3080 |
| Anaconda              | v4.10.3 |
| Python                | v3.8.12 |

#### Data source (N-CMAPSS)
- Link: ```https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/#turbofan-2```
  - 17. Turbofan Engine Degradation Simulation Data Set-2
- Description: 
  - The generation of data-driven prognostics models requires the availability of datasets with run-to-failure trajectories. In order to contribute to the development of these methods, the dataset provides a new realistic dataset of run-to-failure trajectories for a small fleet of aircraft engines under realistic flight conditions. The damage propagation modelling used for the generation of this synthetic dataset builds on the modeling strategy from previous work . The dataset was generated with the Commercial Modular Aero-Propulsion System Simulation (C-MAPSS) dynamical model. The data set is been provided by the Prognostics CoE at NASA Ames in collaboration with ETH Zurich and PARC.
- Scope:
  - Only **N-CMAPSS_DS01-005.h5** was used in this analysis
 
 #### Reference
- Performance Benchmarking And Analysis Of Prognostic Methods For CMAPSS Datasets
  - Link: ```https://papers.phmsociety.org/index.php/ijphm/article/view/2236```
  - Author: Emmanuel Ramasso, Abhinav Saxena
- Aircraft Engine Run-to-Failure Dataset under Real Flight Conditions for Prognostics and Diagnostics
  - Link: ```https://www.researchgate.net/publication/348472709_Aircraft_Engine_Run-to-Failure_Dataset_under_Real_Flight_Conditions_for_Prognostics_and_Diagnostics```
  - Author: Manuel Arias Chao, Chetan S Kulkarni, Kai Goebel, Olga Fink
- PHM Society Data Challenge 2021
  - Link: ```https://data.phmsociety.org/wp-content/uploads/sites/9/2021/08/2021_Data_Challenge.pdf```
  - Author: Manuel Arias Chao, Chetan S Kulkarni, Kai Goebel, Olga Fink

#### Run instructions
1. Create conda environment: ```conda env create --file environment.yaml```
2. Create *Dataset* folder in the same level as *.py scripts
3. Download and store dataset *.h5 in *Dataset* folder
4. Read dataset using dataloader.py
5. Perform simple exploratory data analysis using eda.py
6. Train model using model.py (model folder will be created in workspace)
7. Infer model and generate test score using predictor.py (model folder needs to exist in workspace before inference)

#### Results
- Methodology, analysis and results are summarized in **ds01_report.pptx**