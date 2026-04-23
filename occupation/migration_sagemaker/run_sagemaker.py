import boto3
from sagemaker.processing import ScriptProcessor, ProcessingInput, ProcessingOutput
from sagemaker import get_execution_role
import os
import datetime as dt
from pathlib import Path

script_dir = os.path.dirname(os.path.abspath(__file__))
preprocessing_script = os.path.join(script_dir, "preprocessing.py")

# list files in partitions
s3 = boto3.client("s3")
bucket = "gads-citron-sagemaker-custom-stage"
prefix = "data/power_clk_collect/"
deps_prefix = "code/occupation/deps/"

response = s3.list_objects_v2(Bucket=bucket, Prefix=prefix)

files = [
    obj["Key"]
    for obj in response.get("Contents", [])
    if obj["Key"].endswith(".parquet")
]

# upload dependency files to S3 (latest version uploaded on each run)
for dep in ["Seasonality.py", "decompose.py", "tools.py"]:
    s3.upload_file(os.path.join(script_dir, dep), bucket, deps_prefix + dep)

# run processing job
role = get_execution_role()

processor = ScriptProcessor(
    image_uri="444505633365.dkr.ecr.eu-west-3.amazonaws.com/sagemaker:v1.0.0",
    command=["python"],
    role=role,
    instance_type="ml.m5.2xlarge",
    instance_count=1,
)

for file_key in files:
    timestamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    partition_path = str(Path(file_key[len(prefix):]).parent)
    processor.run(
        code=preprocessing_script,
        inputs=[
            ProcessingInput(
                source=f"s3://{bucket}/{file_key}",
                destination="/opt/ml/input"
            ),
            ProcessingInput(
                source=f"s3://{bucket}/{deps_prefix}",
                destination="/opt/ml/dependencies"
            ),
        ],
        outputs=[
            ProcessingOutput(
                source="/opt/ml/output",
                destination=f"s3://{bucket}/processed/{partition_path}/"
            )
        ],
        job_name=f"preprocess-{file_key.split('/')[-1].replace('.parquet','').replace('_', '-')}-{timestamp}"
    )