import logging
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
consoleFormatter = logging.Formatter('%(levelname)s - %(message)s')

fh = logging.FileHandler('log.txt')
fh.setLevel(logging.DEBUG)
fh.setFormatter(formatter)
logger.addHandler(fh)

ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
ch.setFormatter(consoleFormatter)
logger.addHandler(ch)
logging.getLogger('matplotlib.font_manager').disabled = True