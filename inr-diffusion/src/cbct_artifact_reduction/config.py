from typing import Any

import yaml

from cbct_artifact_reduction.utils import ROOT_DIR

with open(f"{ROOT_DIR}/config.yaml", "r") as f:
    config: dict[str, dict[str, Any]] = yaml.safe_load(f)


LAKEFS_HOST: str = config["lakefs"]["host"]
LAKEFS_USERNAME: str = config["lakefs"]["username"]
LAKEFS_PASSWORD: str = config["lakefs"]["password"]
LAKEFS_SSL_CA_CERT: str = config["lakefs"]["ssl_ca_cert"]
LAKEFS_DATA_REPOSITORY: str = config["lakefs"]["data_repository"]
LAKEFS_COMMIT: str = config["lakefs"]["commit"]
LAKEFS_CACHE_PATH: str = config["lakefs"]["cache_path"]
LAKEFS_VERIFY_SSL: bool = config["lakefs"]["verify_ssl"]
CACHE_PATH: str = config["lakefs"]["cache_path"]

f.close()
