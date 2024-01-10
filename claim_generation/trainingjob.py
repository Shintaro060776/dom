import sagemaker
from sagemaker.pytorch import PyTorch

sagemaker_session = sagemaker.Session()
role = sagemaker.get_execution_role()

data_location = 's3://claim-category20090317'

estimator = PyTorch(entry_point='train_sagemaker.py',
                    role=role,
                    framework_version='1.6.0',
                    py_version='py3',
                    sagemaker_session=sagemaker_session,
                    instance_count=1,
                    instance_type='ml.c4.xlarge',
                    hyperparameters={
                        'epochs': 30,
                        'batch-size': 128,
                        'learning-rate': 0.001
                    })

estimator.fit({'train': data_location})
