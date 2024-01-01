from sagemaker.pytorch import PyTorchModel
import sagemaker

sagemaker_session = sagemaker.Session()

role = sagemaker.get_execution_role()

model_data = 's3://xxxxxxxxxxxx/model.tar.gz'

pytorch_model = PyTorchModel(model_data=model_data,
                             role=role,
                             framework_version='1.5.0',
                             py_version='py3',
                             entry_point='inference.py',
                             sagemaker_session=sagemaker_session)

predictor = pytorch_model.deploy(
    instance_type='ml.m5.large', initial_instance_count=1)
