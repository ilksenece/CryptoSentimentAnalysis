import logging
import pandas as pd
from CryptoSentimentAnalysis import data, models

def retrain_crypto_sent_analysis(filename):
    """Data pipeline and predictions.
    Parameters
    ----------
    filename: str
        Path to the Cryptocurrency tweets CSV input data
    """

    logging.info('Starting the model retraining analysis pipeline')

    header_list = ["Tweet", "Label"]
    processed_data = (
        pd.read_csv(filename, usecols=[1,7], names = header_list)
        .pipe(lambda df: df.astype({'Tweet': 'object',
                                    'Label': 'category'}))
        .pipe(data.clean_tweets)
        .pipe(data.convert_labels_to_numerical)
        .pipe(data.select_clean_tweets_numerical_labels)
    )

    logging.info(processed_data.head())

    models.retrain(processed_data, 'distilbert-base-uncased-finetuned-sst-2-english')

    logging.info('The model retraining analysis pipeline has terminated')

    return

def scoring_crypto_sent_analysis(filename):
    """Data pipeline and predictions.
    Parameters
    ----------
    filename: str
        Path to the Cryptocurrency tweets CSV input data
    """

    logging.info('Starting the data scoring pipeline')

    header_list = ["Tweet", "Label"]
    processed_data = (
        pd.read_csv(filename, usecols=[1,7], names = header_list)
        .pipe(lambda df: df.astype({'Tweet': 'object',
                                    'Label': 'category'}))
        .pipe(data.clean_tweets)
        .pipe(data.convert_labels_to_numerical)
        .pipe(data.select_clean_tweets_numerical_labels)
    )

    preds = models.DF_Predict(processed_data)
    models.compute_metrics(preds, processed_data["Numerical_Labels"])

    logging.info('The data scoring pipeline has terminated')

    return