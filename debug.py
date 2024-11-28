# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
A minimal training script for DiT.
"""
import logging

def create_logger(logging_dir):
    """
    Create a logger that writes to a log file and stdout.
    """
    pid = 0
    # TODO: this gets permission denied when flush into a gs:// bucket log file.
    # logging_dir = os.path.join(logging_dir, str(pid))
    # os.makedirs(logging_dir, mode=0o777, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format=f"[\033[34m%(asctime)s pid={pid}\033[0m] %(message)s",
        datefmt='%Y-%m-%d %H:%M:%S',
        # handlers=[logging.StreamHandler()]
                  # logging.FileHandler(f"{logging_dir}/log.txt")]
    )
    # logger = logging.getLogger(__name__)
    logger = logging.getLogger("Main")
    return logger
    
if __name__ == "__main__":
    logger = create_logger("")
    logger.info("Test")

