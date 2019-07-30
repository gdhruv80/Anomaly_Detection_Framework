#Install SecIntPythonQP
cd /home/ec2-user
export STAGE=beta
aws configure set role_arn arn:aws:iam::384154713874:role/SecIntPythonQPAccessRole-$STAGE --profile pythonqp 
aws configure set credential_source Ec2InstanceMetadata --profile pythonqp 
aws sts get-caller-identity --profile pythonqp
git config --global credential.helper '!aws --profile pythonqp codecommit credential-helper $@' 
git config --global credential.UseHttpPath
git clone https://git-codecommit.us-west-2.amazonaws.com/v1/repos/SecIntPythonQP-$STAGE
cd /home/ec2-user/SecIntPythonQP-beta
pip3 install --user -e .
pip3 install --user pyarrow
