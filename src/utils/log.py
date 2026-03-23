import logging
import os
from datetime import datetime

import config.conf as conf
from config.conf import log_dir


def vlog(src: str, msg: str, level: str = "INFO"):
    if not conf.debug and level is "INFO":
        return

    logger = logging.getLogger("vlog")

    if not logger.handlers:
        logger.setLevel(logging.INFO)
        os.makedirs(log_dir, exist_ok=True)

        fmt = logging.Formatter('%(asctime)s | %(levelname)-5s | %(message)s', '%Y-%m-%d %H:%M:%S')

        sh = logging.StreamHandler()
        sh.setFormatter(fmt)
        logger.addHandler(sh)

        log_file = os.path.join(log_dir, f"{datetime.now().strftime('%Y-%m-%d')}.log")
        fh = logging.FileHandler(log_file, encoding='utf-8')
        fh.setFormatter(fmt)
        logger.addHandler(fh)

        logger.propagate = False

    func = getattr(logger, level.lower(), logger.info)
    func(f"{src}: {msg}")
