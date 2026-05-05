import boto3
from sagemaker.processing import ScriptProcessor, ProcessingInput, ProcessingOutput
from sagemaker import get_execution_role
import os
import datetime as dt
from pathlib import Path

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

for dev_prefix in sorted(dev_prefixes):
    timestamp = dt.datetime.now().strftime("%H%M%S")
    device_path = dev_prefix[len(prefix):]  # site=X/equip=Y/
    parts = device_path.strip("/").split("/")  # ["site=x", "device_id=y"]
    device_id = parts[1].split("=")[-1]

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
                destination=f"s3://{bucket}/data/occupation/processed/{device_path.rstrip('/')}/"
            )
        ],
        job_name = f"dev{device_id}-{timestamp}",
        wait=False
    )


