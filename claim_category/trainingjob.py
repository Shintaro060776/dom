import sagemaker
from sagemaker.sklearn.estimator import SKLearn

sagemaker_session = sagemaker.Session()
role = sagemaker.get_execution_role()

data_location = 's3://claim-category20090317'

sklearn_estimator = SKLearn(
    entry_point='train_sagemaker.py',
    role=role,
    instance_type='ml.m5.large',
    framework_version='0.23-1',
    py_version='py3',
    sagemaker_session=sagemaker_session
)

sklearn_estimator.fit({'train': data_location})
