import sagemaker
from sagemaker.sklearn.model import SKLearnModel

sagemaker_session = sagemaker.Session()

role = sagemaker.get_execution_role()

model_data = 's3://sagemaker-ap-northeast-1-715573459931/sagemaker-scikit-learn-2023-12-23-09-23-48-347/output/model.tar.gz'
# 上記のS3のパスは、修正する

model = SKLearnModel(model_data=model_data,
                     role=role,
                     entry_point='inference.py',
                     framework_version='0.23-1',
                     py_version='py3',
                     sagemaker_session=sagemaker_session)

predictor = model.deploy(instance_type='ml.m5.large', initial_instance_count=1)
