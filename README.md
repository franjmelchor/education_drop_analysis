# An empirical evaluation of clustering processes for early detection of university dropout
The elevated rates of dropout within academic institutions have prompted the use of Artificial Intelligence (AI) to tackle this issue. These efforts often rely mainly on administrative and academic data, lacking personal information about students. In a previous study, we explored machine learning models to leverage this data and harness their knowledge-extraction capabilities. However, a critical factor, the availability of labeled data, was not addressed. Obtaining these data may be challenging due to their distribution across different systems or the considerable time required to collect them, especially when new degrees are being implemented. The lack of labeled data is a major obstacle for institutions that do not possess them so that they are unable to take advantage of the full potential of AI for their purposes. Clustering algorithms have conventionally been employed to uncover latent patterns within unlabeled data. These unsupervised algorithms may reduce the need for data labeling; nonetheless, it necessitates rigorous validation of the resulting clusters, particularly when dealing with datasets encompassing numerical and categorical attributes. This paper introduces a comparison of various clustering algorithms to discern the most appropriate technique for uncovering the underlying factors contributing to university student attrition, employing unlabeled data. The novelty lies not only in the algorithmic comparison but also in their integration with diverse data preprocessing methodologies, streamlining the selection of the optimal combination including advanced data transformations for the harmonization of numerical and categorical information. It is illustrated through a real-world case utilizing academic data from a Spanish university, providing empirical validation for the proposed methodology. The insights gained can be extrapolated to analogous experiments where social or economic data is scarce, and most of the available attributes are academic in nature.

## License
This project is licensed under the [MIT License](pte)


## Project structure:
This project is based on the framework defined by [MD4DSPRR Utils](https://github.com/i3uex/apitep_utils). 

This framework defines the structure of a data science project divided into the following phases:
- __ETL__: Extract Transform and Load data tasks
- __Feature Engineering__: process of creating or transforming features to enhance model performance.
- __Analysis and Modelling__: process of analysing data using machine learning models or other techniques.

Furthermore, in our case, using the ETL class, we will implement an additional phase called __Fetch Data__, which will allow us to give a common format to the raw data of the project.

Based on this framework, the structure of this project is as follows:
- __Data__:
    - __Raw__: raw originally data.
    - __Interim__: data with common format.
    - __Processed__: data resulting from the application of the ETL phase
    - __For_analyisis_and_modeling__: data resulting from the application of the Feature Engineering phase.
- __Code__:
  - __data_acquisition_and_understanding__: 
    - __fetch_data__: Data capture and formatting into a common format.
    - __etl__:  Data that may provide noisy information to the models are eliminated, mainly due to being incomplete or not providing correct information.
  - __feature_engineering__: features creation, elimination and fusion for personal access data and for course data.
  - __analysis_and_modeling__: notebooks with different experiments based on clustering algorithms: K-Means, Agglomerative, Gaussian Mixture, K-Modes and COOLCCAT.

It is important to note that the __data in this repository is not real data,__ but rather a set of fictitious data used to reproduce the analysis performed.

## Prerequisites

- **Python 3.10+** (virtual environment recommended)
- **Operating System:** Linux (or Windows Subsystem for Linux)
- **Dependencies:** All packages listed in `requirements.txt`

## Setup

1. Create and activate a virtual environment:
   ```bash
   python3.10 -m venv .venv
   source .venv/bin/activate
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Running the Pipeline

To execute the full project pipeline with DVC, run:

```bash
 dvc repro
```

This will sequentially perform:

1. **fetch data**
2. **ETL**
3. **feature engineering**

## Launching Clustering Models

- **Python-based models:** Open the corresponding Jupyter notebooks and run each cell in order.
- **ROCK & COOLCAT algorithms:** Open and execute the provided `.R` scripts.
