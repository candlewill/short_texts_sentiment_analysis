__author__ = 'NLP-PC'
import pickle
import logging
import csv

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def dump_picle(data, filename):
    pickle.dump(data, open(filename, "wb"))
    logger.info('Save complete, data is stored in %s', filename)

def csv_save(data, filename):
    with open(filename, 'w', newline='', encoding='ISO-8859-1') as f:
        writer = csv.writer(f, quoting=csv.QUOTE_NONNUMERIC)
        writer.writerows(data)
    logger.info('File Saved')