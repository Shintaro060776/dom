import sagemaker
from sagemaker.sklearn.model import SKLearnModel

sagemaker_session = sagemaker.Session()

role = sagemaker.get_execution_role()

model_data = 's3://sagemaker-ap-northeast-1-715573459931/sagemaker-scikit-learn-2023-12-13-14-53-22-835/output/model.tar.gz'

model = SKLearnModel(model_data=model_data,
                     role=role,
                     entry_point='inference.py',
                     framework_version='0.23-1',
                     py_version='py3',
                     sagemaker_session=sagemaker_session)

predictor = model.deploy(instance_type='ml.m5.large', initial_instance_count=1)
