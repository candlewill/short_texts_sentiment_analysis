__author__ = 'NLP-PC'
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def log_performance(accuracy, f1, precision_binary, recall_binary, len_test):
    # create a file handler
    handler = logging.FileHandler('./logs/Performance_Log.log')
    handler.setLevel(logging.INFO)
    # create a logging format
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    # add the handlers to the logger
    logger.addHandler(handler)
    logger.info('Accuracy: %s, Macro_F score: %s, Precision: %s, Recall: %s; Test data size: %s', accuracy, f1,precision_binary, recall_binary, len_test)
    logger.removeHandler(handler) # remove the Handler after you finish your job

def log_state(msg):
    # create a file handler
    handler = logging.FileHandler('./logs/State_Log.log')
    handler.setLevel(logging.INFO)
    # create a logging format
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    # add the handlers to the logger
    logger.addHandler(handler)
    logger.info(msg)
    logger.removeHandler(handler)