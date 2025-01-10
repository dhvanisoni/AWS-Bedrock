import boto3 
import json
import os
import base64


prompt_data = """ 
Provide me an hd image of beach with a blue sky and sunrise view
"""

prompt_template = [{"text":prompt_data,"weight": 1}]

bedrock = boto3.client(service_name="bedrock-runtime")

payload = {
    "prompt": prompt_template,
    "cfg_scale" : 10,
    "seed" : 0,
    "steps" : 50,
    "height" : 1024,
    "width" : 1024
}
body = json.dumps(payload)

model_id = "stability.stable-diffusion-xl-v1"
response = bedrock.invoke_model(
    modelId = model_id,
    accept = "application/json",
    contentType = "application/json",
    body=body
)

response_body = json.loads(response.get("body").read()) 
print(response_body)
artifact = response_body.get("artifact")[0]
image_encoded = artifact.get("base4").encode("utf-8")
image_byte = base64.b64decode(image_encoded)

# save image to a file in the output directory
output_dir = "output"
os.makedirs(output_dir, exist_ok = True)
file_name = f"{output_dir}/generated-img.png"
with open(file_name, "wb") as f:
    f.write(image_byte)
 



