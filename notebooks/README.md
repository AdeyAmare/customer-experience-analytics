# Notebooks

This folder contains the Jupyter Notebooks used for exploratory data analysis (EDA), data visualization, and detailed NLP model implementation and testing.

These notebooks document the intermediate steps, experiments, and visualization results of the project.

1. ```preprocessing_eda.ipynb```

- Purpose: Detailed exploration and cleaning of the data.

- Content: This notebook focuses on the initial steps after data collection. It includes:

    - Loading the raw or minimally processed review data.

    - Performing Exploratory Data Analysis (EDA), such as checking distributions of ratings, review counts per bank, and temporal trends.

    - Visualizing key statistics and data quality checks (e.g., missing values).

    - Step-by-step documentation of the preprocessing logic defined in ```src/data_preprocessing.py```.

2. ```sentiment_analysis.ipynb```

- Purpose: Implementation, experimentation, and comparison of the NLP models.

- Content: This notebook focuses on the core analysis, covering:

    - Implementing and running the VADER and TextBlob sentiment analysis techniques.

    - Setting up and testing the DistilBERT (Hugging Face) model for deep learning sentiment classification.

    - Experimenting with different TF-IDF parameters for keyword extraction.

    - Visualizing the sentiment distribution across banks and ratings.

    - Generating and visualizing the thematic distribution results.

## Usage
To view or run these notebooks:

1. Ensure you have Jupyter installed (pip install jupyter).

2. Open your terminal in the main project directory and run jupyter notebook.

3. Navigate to the notebooks/ folder and click on the file you wish to open.

