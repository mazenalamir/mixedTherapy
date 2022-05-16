# mixedTherapy
This repository contains the python code that is used to produce the results of the paper entitled:

**Learning-based sensitivity analysis and feedback deign for drug delivery of mixed therapy of cancer in the presence of high model uncertainties**
by *Mazen Alamir*

**Abstract**

In this paper, a methodology is proposed that enables to analyze the sensitivity of the outcome of a therapy to unavoidable high dispersion of the patient specific parameters on one hand and to the choice of the parameters that define the drug delivery feedback strategy on the other hand. More precisely, a method is given that enables to extract and rank the most influent parameters that determine the probability of success/failure of a given feedback therapy for a given set of initial conditions over a cloud of realizations of uncertainties. Moreover predictors of the expectations of the amounts of drugs being used can also be derived. This enables to design an efficient stochastic optimization framework that guarantees safe contraction of the tumor while minimizing a weighted sum of the quantities of the different drugs being used. The framework is illustrated and validated using the example of a mixed therapy of cancer involving three combined drugs namely: a chemotherapy drug, an immunology vaccine and an immunotherapy drug. Finally, in this specific case, it is shown that dash-boards can be built in the 2D-space of the most influent state components that summarize the outcomes' probabilities and the associated drug usage as iso-values curves in the reduced state space. 

## list of files 

- `mixed_therapy.py` the python module that contains the main classes and methods 
- `learning_based_analysis_and_design_of_mixed_therapy_of_cancer.ipynb`the jupyter notebook that produces the results using the previous module. 

