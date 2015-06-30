__author__ = 'NLP-PC'
import pickle
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def dump_picle(data, filename):
    pickle.dump(data, open(filename, "wb"))
    logger.info('Save complete, data is stored in %s', filename)