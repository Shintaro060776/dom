import sagemaker
from sagemaker.pytorch import PyTorch

sagemaker_session = sagemaker.Session()

role = sagemaker.get_execution_role()

estimator = PyTorch(
    entry_point='train_sagemaker.py',
    source_dir='/home/ec2-user/SageMaker/test',
    role=role,
    framework_version='1.5.0',
    instance_type='ml.m5.large',
    instance_count=1,
    py_version='py3',
    sagemaker_session=sagemaker_session
)

train_data_path = 's3://well-generate20090317'

estimator.fit({'train': train_data_path})
