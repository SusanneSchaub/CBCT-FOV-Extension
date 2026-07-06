import hashlib
import os
import warnings

import boto3
from botocore.config import Config
import lakefs
from lakefs.client import Client

import cbct_artifact_reduction.config as cfg

REPO = "mmdental-cbct"
#BRANCH = "processed_data"


class CustomLakeFSClient:
    def __init__(self, repo: str) -> None:
        self.repo: str = repo
        self.client = Client(
            host=cfg.LAKEFS_HOST,
            #verify_ssl=cfg.LAKEFS_VERIFY_SSL,
            ssl_ca_cert=cfg.LAKEFS_SSL_CA_CERT,
            username=cfg.LAKEFS_USERNAME,
            password=cfg.LAKEFS_PASSWORD,
        )

        self.branch: str

    def set_branch(self, branch: str) -> None:
        self.branch = branch

    def get_branches(self):
        return lakefs.repository(f"{self.repo}", client=self.client).branches()

    def list_all_files(self):
        return (
            lakefs.repository(f"{self.repo}", client=self.client)
            .branch(f"{self.branch}")
            .objects()
        )

    def list_files_in_folder(self, folder: str):
        return (
            lakefs.repository(f"{self.repo}", client=self.client)
            .branch(f"{self.branch}")
            .objects(prefix=f"{folder}")
        )


class CustomBoto3Client:
    def __init__(self, repo: str) -> None:
        self.repo = repo

        config = Config(
            request_checksum_calculation="when_required",
            response_checksum_validation="when_required",
            # ... your remaining config
        )

        self.client = boto3.client(
            "s3",
            endpoint_url=cfg.LAKEFS_HOST,
            aws_access_key_id=cfg.LAKEFS_USERNAME,
            aws_secret_access_key=cfg.LAKEFS_PASSWORD,
            config = config
            #verify=cfg.LAKEFS_SSL_CA_CERT, #otherwise it doesnt work
        )
        self.branch = cfg.LAKEFS_COMMIT
        self.cache_path = cfg.CACHE_PATH

    def list_files_in_folder(self, folder: str):
        paginator = self.client.get_paginator("list_objects_v2")
        pages = paginator.paginate(Bucket=self.repo, Prefix=f"{self.branch}/{folder}/")

        filenames = []

        for page in pages:
            for obj in page["Contents"]:
                filenames.append(obj["Key"])
                print(obj["Key"])

        return filenames

    def get_file(self, object_name, file_obj=None):
        """Load the file from the S3 storage to the local disk or directly into the ram. If caching is activated, a
        local cache is used"""
        local_path = None

        if file_obj is None:
            # download the file into the local cache if it is not already in the cache
            suffix = "." + ".".join(object_name.split(".")[-2:])
            local_filename = (
                hashlib.md5(object_name.encode("utf-8")).hexdigest() + suffix
            )
            local_path = os.path.join(self.cache_path, local_filename)

            if not os.path.exists(local_path):
                try:
                    print(f"Cache miss. Download object {object_name}")
                    self.client.download_file(self.repo, object_name, local_path)
                except:
                    local_path = None
                    warnings.warn(
                        "An error occurred during downloading the file from lakefs"
                    )
        else:
            # download the object to a file buffer
            self.client.download_fileobj(self.repo, object_name, file_obj)

        return local_path


if __name__ == "__main__":
    # customLakeFSClient = CustomLakeFSClient(REPO)
    # customLakeFSClient.set_branch(BRANCH)
    customBoto3Client = CustomBoto3Client(REPO)
    print(customBoto3Client.list_files_in_folder("frames"))
