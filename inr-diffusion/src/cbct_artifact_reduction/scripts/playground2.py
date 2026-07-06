import cbct_artifact_reduction.config as cfg
import cbct_artifact_reduction.lakefs_own as lakefs_own

client = lakefs_own.CustomBoto3Client(f"{cfg.LAKEFS_DATA_REPOSITORY}")
print("getting files")
client.list_files_in_folder("frames/256x256")
print("done")
