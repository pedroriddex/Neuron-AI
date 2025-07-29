# Terraform Module – Neuron S3 Buckets & IAM (Sprint-0 T-017)

This module provisions the storage backbone for Neuron MVP:

| Bucket            | Purpose                               |
|-------------------|---------------------------------------|
| `<prefix>-raw`    | Raw source datasets (immutable)       |
| `<prefix>-processed` | Pre-processed Parquet/Arrow datasets |
| `<prefix>-models` | Quantised or fine-tuned model files   |

All buckets have:
* **Versioning** enabled
* **AES-256 SSE** enabled
* Multipart uploads aborted after 7 days (raw bucket)

An IAM user `${prefix}-ci` is created with **least-privilege** access:
`ListBucket`, `GetObject`, `PutObject` on the three buckets only.

## Usage

```bash
cd infra/terraform
terraform init -backend-config="bucket=<state-bucket>" -backend-config="key=neuron.tfstate"
terraform plan -var="project_prefix=neuron-mvp"
# Review output
terraform apply -var="project_prefix=neuron-mvp"
```

Variables:
* `project_prefix` (required) – unique prefix, e.g. `neuron-mvp`
* `aws_region` (default `us-east-1`)

After `apply`, capture the Access Key for the user and add it as GitHub
Secrets `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY` for CI jobs that need
artifact upload.
