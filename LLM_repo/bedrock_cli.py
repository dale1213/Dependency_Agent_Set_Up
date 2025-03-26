import boto3

session = boto3.Session()
credentials = session.get_credentials()
frozen_credentials = credentials.get_frozen_credentials()
print("Access Key:", frozen_credentials.access_key)
print("Secret Key:", frozen_credentials.secret_key)
print("Token:", frozen_credentials.token)
