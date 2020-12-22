import jwt
from time import time


# generate jwt using private key
jwtjson = \
{
  "access_token": "P4m2UKk0vKhwaGVJa5SQpq58ClAH",
  "audience": "microgateway",
  "api_product_list": [
    "c360-bank-ca-dev"
  ],
  "application_name": "manulife-c360-ca-bank-dev",
  "nbf": 1607533559,
  "iss": "http://manulife-development-dev.apigee.net/v1/mg/oauth2/token",
  "scopes": [
    "c360:idmapping:bank:read",
    "c360:idmapping:bank:write"
  ],
  "exp": time() + 300,
  "iat": 1607533619,
  "client_id": "1NH8uh0otpnfi49xPHqKocHyRgH1n8l2",
  "jti": "c70a21a8-9cca-45ac-85ea-baf0bc6b71fc"
}

# create private key: openssl genrsa 2048 > key
# or, generate private key and extract public using: ssh-keygen -y -f key > key.pub (not tried)
private_key = open('key').read()
token = jwt.encode(jwtjson, private_key, algorithm='RS256', headers={'kid': '1'}).decode('utf-8')
print(token)

# verify jwt using public key
# extract public key using: openssl rsa -in key -pubout > key.pub
public_key = open('key.pub').read()
payload = jwt.decode(token, key=public_key, algorithms=['RS256'], verify=True)

print(payload)

