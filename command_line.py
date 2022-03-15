import sys
import logging
import click
from CryptoSentimentAnalysis import pipelines

logging.basicConfig(
    format='[%(asctime)s|%(module)s.py|%(levelname)s]  %(message)s',
    datefmt='%H:%M:%S',
    level=logging.INFO,
    stream=sys.stdout
)

@click.command()
@click.option('--filename',
              type=click.Path(exists=True),
              prompt='Path to the Cryptocurrency tweets CSV file',
              help='Path to the TCryptocurrency tweets CSV file')
def crypto_sent_analysis_retrain(filename):
    pipelines.retrain_crypto_sent_analysis(filename)
def crypto_sent_analysis_score(filename):
    pipelines.scoring_crypto_sent_analysis(filename)
