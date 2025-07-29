resource "aws_s3_bucket" "raw" {
  bucket = "${var.project_prefix}-raw"

  versioning {
    enabled = true
  }

  server_side_encryption_configuration {
    rule {
      apply_server_side_encryption_by_default {
        sse_algorithm = "AES256"
      }
    }
  }

  lifecycle_rule {
    id      = "expire-multipart-uploads"
    enabled = true

    abort_incomplete_multipart_upload_days = 7
  }
}

resource "aws_s3_bucket" "processed" {
  bucket = "${var.project_prefix}-processed"

  versioning {
    enabled = true
  }

  server_side_encryption_configuration {
    rule {
      apply_server_side_encryption_by_default {
        sse_algorithm = "AES256"
      }
    }
  }
}

resource "aws_s3_bucket" "models" {
  bucket = "${var.project_prefix}-models"

  versioning {
    enabled = true
  }

  server_side_encryption_configuration {
    rule {
      apply_server_side_encryption_by_default {
        sse_algorithm = "AES256"
      }
    }
  }
}

# IAM user for CI with least privilege
resource "aws_iam_user" "ci" {
  name = "${var.project_prefix}-ci"
}

resource "aws_iam_policy" "ci_policy" {
  name        = "${var.project_prefix}-ci-policy"
  description = "Least-privilege access to S3 buckets"

  policy = jsonencode({
    Version : "2012-10-17",
    Statement : [
      {
        Effect : "Allow",
        Action : [
          "s3:PutObject",
          "s3:GetObject",
          "s3:ListBucket"
        ],
        Resource : [
          aws_s3_bucket.raw.arn,
          "${aws_s3_bucket.raw.arn}/*",
          aws_s3_bucket.processed.arn,
          "${aws_s3_bucket.processed.arn}/*",
          aws_s3_bucket.models.arn,
          "${aws_s3_bucket.models.arn}/*"
        ]
      }
    ]
  })
}

resource "aws_iam_user_policy_attachment" "ci_attach" {
  user       = aws_iam_user.ci.name
  policy_arn = aws_iam_policy.ci_policy.arn
}
