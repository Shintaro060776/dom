import sagemaker
from sagemaker.sklearn.estimator import SKLearn

sagemaker_session = sagemaker.Session()

role = sagemaker.get_execution_role()

estimator = SKLearn(
    entry_point='train.py',
    role=role,
    framework_version='0.23-1',
    instance_type='ml.m5.large',
    instance_count=1,
    py_version='py3',
    sagemaker_session=sagemaker_session
)

train_data_path = 's3://realtime20090317'

estimator.fit({'train': train_data_path})
