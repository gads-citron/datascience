import boto3
from sagemaker.processing import ScriptProcessor, ProcessingInput, ProcessingOutput
from sagemaker import get_execution_role
import os
import time
import logging
import datetime as dt
from pathlib import Path

BATCH_SIZE = 7
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")


def wait_for_jobs(sm_client, job_names):
    while True:
        statuses = {
            name: sm_client.describe_processing_job(ProcessingJobName=name)["ProcessingJobStatus"]
            for name in job_names
        }
        in_progress = [n for n, s in statuses.items() if s in ("InProgress", "Stopping")]
        if not in_progress:
            break
        logging.info(f"{len(in_progress)}/{len(job_names)} jobs en cours...")
        time.sleep(60)
    failed = [n for n, s in statuses.items() if s == "Failed"]
    if failed:
        logging.warning(f"{len(failed)} jobs en échec : {failed}")

script_dir = os.path.dirname(os.path.abspath(__file__))
preprocessing_script = os.path.join(script_dir, "preprocessing.py")

s3 = boto3.client("s3")
bucket = "gads-citron-sagemaker-custom-stage"
prefix = "data/occupation/power_clk_collect/"
deps_prefix = "code/occupation/deps/"

paginator = s3.get_paginator("list_objects_v2")
dev_prefixes = set()
for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
    for obj in page.get("Contents", []):
        key = obj["Key"]
        if not key.endswith(".parquet"):
            continue
        parts = Path(key[len(prefix):]).parts
        if len(parts) >= 4:  # site / equip / sensor / file
            dev_prefixes.add(prefix + "/".join(parts[:3]) + "/")


# upload dependency files to S3 (latest version uploaded on each run)
for dep in ["Seasonality.py", "decompose.py", "tools.py"]:
    s3.upload_file(os.path.join(script_dir, dep), bucket, deps_prefix + dep)

role = get_execution_role()
processor = ScriptProcessor(
    image_uri="444505633365.dkr.ecr.eu-west-3.amazonaws.com/sagemaker:v1.0.0",
    command=["python"],
    role=role,
    instance_type="ml.m5.2xlarge",
    instance_count=1,
)

sm_client = boto3.client("sagemaker")
all_prefixes = sorted(dev_prefixes)

for batch_start in range(0, len(all_prefixes), BATCH_SIZE):
    batch = all_prefixes[batch_start:batch_start + BATCH_SIZE]
    job_names = []

    batch_timestamp = dt.datetime.now().strftime("%Y%m%d-%H%M")
    for idx, dev_prefix in enumerate(batch):
        device_path = dev_prefix[len(prefix):]  # site=X/device=Y/sensor=Z/
        parts = device_path.strip("/").split("/")
        device_id = parts[1].split("=")[-1]

        job_name = f"dev{device_id}-{batch_timestamp}-{idx}"[:63]
        processor.run(
            code=preprocessing_script,
            inputs=[
                ProcessingInput(
                    source=f"s3://{bucket}/{dev_prefix}",
                    destination="/opt/ml/processing/input"
                ),
                ProcessingInput(
                    source=f"s3://{bucket}/{deps_prefix}",
                    destination="/opt/ml/processing/dependencies"
                ),
            ],
            outputs=[
                ProcessingOutput(
                    source="/opt/ml/processing/output",
                    destination=f"s3://{bucket}/data/occupation/processed/power/{device_path.rstrip('/')}/"
                )
            ],
            job_name=job_name,
            wait=False
        )
        job_names.append(job_name)

    logging.info(f"Batch {batch_start // BATCH_SIZE + 1} lancé ({len(batch)} jobs)")
    wait_for_jobs(sm_client, job_names)


