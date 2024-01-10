import sagemaker
from sagemaker.sklearn.model import SKLearnModel

sagemaker_session = sagemaker.Session()
role = sagemaker.get_execution_role()

model_artifact = 's3://xxxxxxxxxxxxxxxx'

sklearn_model = SKLearnModel(
    model_data=model_artifact,
    role=role,
    entry_point='inference.py',
    framework_version='0.23-1',
    py_version='py3',
    sagemaker_session=sagemaker_session
)

predictor = sklearn_model.deploy(
    initial_instance_count=1,
    instance_type='ml.m5.large'
)
