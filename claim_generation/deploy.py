import sagemaker
from sagemaker.pytorch.model import PyTorchModel

sagemaker_session = sagemaker.Session()
role = sagemaker.get_execution_role()

model_artifact = 's3://sagemaker-ap-northeast-1-715573459931/pytorch-training-2024-01-10-10-27-22-330/output/model.tar.gz'

pytorch_model = PyTorchModel(
    model_data=model_artifact,
    role=role,
    entry_point='inference.py',
    framework_version='1.6.0',
    py_version='py3',
    sagemaker_session=sagemaker_session
)

predictor = pytorch_model.deploy(
    initial_instance_count=1,
    instance_type='ml.m4.xlarge'
)
