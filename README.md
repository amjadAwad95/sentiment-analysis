# Sentiment Analysis Project
This project is designed to perform sentiment analysis on text data using various preprocessing techniques and feature extraction methods. The goal is to classify text into positive or negative sentiment categories based on the extracted features. The project is implemented in a Jupyter Notebook and uses Python libraries such as Pandas, Matplotlib, and regular expressions for data manipulation, visualization, and text processing.

## Introduction
Sentiment analysis is a natural language processing (NLP) technique used to determine the sentiment expressed in a piece of text. This project focuses on analyzing text data to classify it into positive or negative sentiment categories. The project is divided into several components, including data loading, preprocessing, feature extraction, and visualization.

## Features

- **Data Loading**: The project includes a `DataLoader` class that loads text data from a directory structure where each subdirectory corresponds to a specific sentiment label (e.g., "pos", "neg").

- **Preprocessing**: The project provides several preprocessing classes to clean and prepare the text data. These include:
    * `RemoveSpecialCharacters`: Removes special characters from the text.
    
    * `ConvertEmoji`: Converts emojis into their textual representation.
    
    * `TextStemmer`: Applies stemming to the words in the text.
    
    * `ConvertNumberToWords`: Converts numerical values into their word equivalents.
    
    * `NegationHandling`: Handles negations in the text.
    
    * `RemoveStopWords`: Removes stopwords from the text.

- **Feature Extraction**: The project includes various feature extractor classes to extract specific features from the text data. These features include:

    * `NegationCount`: Counts the number of negation words.
    
    * `IngCount`: Counts the number of words ending with "ing".
    
    * `EdCount`: Counts the number of words ending with "ed".
    
    * `PositiveWordCount`: Counts the number of positive words.
    
    * `NegativeWordsCount`: Counts the number of negative words.
    
    * `PositiveEmojiCount`: Counts the number of positive emojis.
    
    * `NegativeEmojiCount`: Counts the number of negative emojis.
    
    * `WordCount`: Counts the total number of words.
    
    * `AverageWordLength`: Calculates the average length of words.
    
    * `UniqueWordRatio`: Calculates the ratio of unique words to the total number of words.
    
    * `ExclamationCount`: Counts the number of exclamation marks.
    
    * `QuestionCount`: Counts the number of question marks.
    
    * `EllipsisCount`: Counts the number of ellipses.
    
    * `CapitalizedRatio`: Calculates the ratio of fully capitalized words.
    
    * `RepeatedLettersCount`: Counts the number of words with repeated letters.
- **Visualization**: The project includes a `BarVisualize` class for creating bar plots to visualize the extracted features.

## Installation
```bash
pip install -r requirements.txt
```

## Usage

1. **Data Loading**: Use the `DataLoader` class to load your text data. The data should be organized in a directory structure where each subdirectory corresponds to a specific sentiment label.
   ``` python
   data_load = DataLoder("data/training_dat", labels=["neg", "pos"], shuffle=True)
   sentiment_data = data_load.load_data()
   ```

2. **Preprocessing**: Apply the preprocessing steps to the text data using the `PreprocessingPipeline` class.
   ```python
   preprocessing_pipeline = PreprocessingPipeline([
    NegationHandling(),
    TextStemmer(),
    ConvertEmoji(),
    RemoveSpecialCharacters(),
    ConvertNumberToWords(),
    RemoveStopWords()
    ])
    sentiment_data["text"] = sentiment_data["text"].apply(preprocessing_pipeline.transform)
    ```

3. **Feature Extraction**: Extract features from the preprocessed text data using the `FeaturesExtractorPipeline` class.
   ```python
   features_extractor_pipeline = FeaturesExtractorPipeline([
    NegationCount(),
    IngCount(),
    EdCount(),
    PositiveWordCount(),
    NegativeWordsCount(),
    PositiveEmojiCount(),
    NegativeEmojiCount(),
    WordCount(),
    AverageWordLength(),
    UniqueWordRatio(),
    ExclamationCount(),
    QuestionCount(),
    EllipsisCount(),
    CapitalizedRatio(),
    RepeatedLettersCount(),
    ])
    expanded = sentiment_data["text"].apply(extract_features).apply(pd.Series)
    sentiment_data = sentiment_data.join(expanded)
   ```

## Feature Representation and Engineering
The project includes a `TFIDFVectorizer` class that transforms text data into numerical feature vectors using the Term Frequency-Inverse Document Frequency (TF-IDF) method. TF-IDF is a statistical measure used to evaluate the importance of a word in a document relative to a collection of documents (corpus). It is widely used in text analysis and natural language processing tasks, such as sentiment analysis, document classification, and information retrieval.

```python
vectorizer=TFIDFVectorizer(sentiment_data["text"],{"max_df":0.8,"min_df":2,"ngram_range":(1, 2),    
                            "max_features":8000})

tfidf_matrix=vectorizer.transform()

tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=vectorizer.vectorizer.get_feature_names_out())

sentiment_data = pd.concat([sentiment_data, tfidf_df], axis=1)
```

## ModelTrain and ModelEvaluate

The project includes a `LogisticRegressionTrain` class for sentiment analysis, which is a widely used machine learning algorithm for binary and multi-class classification tasks. Logistic Regression is particularly effective for text classification problems, such as sentiment analysis, where the goal is to predict whether a given text expresses a positive or negative sentiment.

### ModelEvaluate

```python
accuracy_evaluate=AcuracyEvaluate(X_test,y_test)
precision_evaluate=PrecisionEvaluate(X_test,y_test)
recall_evaluate=RecallEvaluate(X_test,y_test)
f1_evaluate=F1Evaluate(X_test,y_test)
k_fold=KFoldEvaluate(X_train,y_train,scoring=["accuracy", "neg_log_loss"],params={"n_splits" :10,
                        "shuffle" : True, "random_state" : 42})


cross_validation_evaluate = CrossValidationEvaluate(X_train,y_train,scoring="accuracy",
                                                    params={"n_splits" :10,"shuffle" : True, "random_state" : 42})
```

### ModelTrain

```python
logistic=LogisticRegressionTrain(X_train,y_train,{"C":10,"penalty":'l2',"solver":'liblinear',
                                                        "class_weight":'balanced',"random_state":42,"max_iter":1000})
logistic_model=logistic.train()

cross_validation=cross_validation_evaluate.evaluate(logistic_model)
print(f"Cross-Validation Accuracy Scores: {cross_validation}")
print(f"Mean Accuracy Scores: {cross_validation.mean()}")
```

```pyton
cv_scores = k_fold.evaluate(logistic_model)
TrainingValidationPlot(cv_scores).visualize()
```

```python
evaluate_printer=EvaluatePrinter(logistic_model,X_train,y_train,X_test,y_test)
evaluate_printer.print()
```

## Summary of Logistic Regression Model Performance

### Evaluation Metrics

The Logistic Regression model was evaluated using various performance metrics:

- **Accuracy**: 0.8700906344410876  
- **Precision**: 0.8787878787878788  
- **Recall**: 0.8630952380952381  
- **F1-Score**: 0.8708708708708709  

These metrics indicate good performance, with an overall accuracy of about 87%.

### Classification Report

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| 0     | 0.86      | 0.88   | 0.87     | 163     |
| 1     | 0.88      | 0.86   | 0.87     | 168     |

### Interpretation:
- For both classes, the model shows balanced precision and recall, indicating consistent performance across different classes.

### Plot Analysis

### Accuracy Comparison Across Folds

- **Training Accuracy**: Remains consistently high across all folds.
- **Validation Accuracy**: Shows some fluctuations but generally follows a similar trend to training accuracy.

### Log Loss Comparison Across Folds

- **Training Loss**: Stays very low and stable across all folds.
- **Validation Loss**: Exhibits some variations but remains relatively close to the training loss.

### Overfitting Indication

The plots suggest that the model may not be significantly overfitting:
- The training and validation accuracies are relatively close, with some minor fluctuations in validation accuracy.
- Training and validation losses are also close, with some minor variations in validation loss.

## Function Usage Guide

### Function Usage Guide  

To use this function, you need to provide **two parameters**:  

1.  **Path**: The directory where the data is stored. The structure should be as follows: 

2.  **Model**: The model for which you want to display the test data. 

   ```plaintext
   data/
   ├── pos/
   │   ├── file1.txt
   │   ├── file2.txt
   │   ├── ...
   ├── neg/
   │   ├── file1.txt
   │   ├── file2.txt
   │   ├── ...
```

```python
def test_on_your_data(path,model):
    data_load=DataLoder(path,labels=["neg","pos"])
    data=data_load.load_data()

    preprocessing_pipeline=PreprocessingPipeline([
    NegationHandling(),
    TextStemmer(),
    ConvertEmoji(),
    RemoveSpecialCharacters(),
    ConvertNumberToWords(),
    RemoveStopWords()
    ])

    def text_preprocessing(text:str):
        preprocessed_text=preprocessing_pipeline.transform(text)
        return preprocessed_text

    data["text"]=data["text"].apply(text_preprocessing)

    tfidf_matrix=vectorizer.vectorizer.transform(data["text"])

    tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=vectorizer.vectorizer.get_feature_names_out())

    data = pd.concat([data, tfidf_df], axis=1)

    data["sentiment_label"] = np.where(data["sentiment_label"] == "pos", 1, 0)

    data=data.drop(["text"], axis=1)

    y_test=data["sentiment_label"]
    X_test=data.drop("sentiment_label",axis=1)
    
    y_pred=model.predict(X_test)
    
    accuracy = accuracy_score(y_test,y_pred)
    precision = precision_score(y_test,y_pred)
    recall = recall_score(y_test,y_pred)
    f1 = f1_score(y_test,y_pred)
 
    print(f"accuracy = {accuracy}")
    print(f"precision = {precision}")
    print(f"recall = {recall}")
    print(f"f1 = {f1}")
```

