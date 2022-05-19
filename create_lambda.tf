variable region {
 default = "eu-west-1"
}
 
provider aws {
 region = var.region
 profile = "twitter"
}
 
data aws_caller_identity current {}
 
locals {
 account_id          = data.aws_caller_identity.current.account_id
 ecr_repository_name = "twitter-positivity-analyzer"
 ecr_image_tag       = "latest"
}
 
resource aws_ecr_repository repo {
 name = local.ecr_repository_name
}
 
resource null_resource ecr_image {
 triggers = {
   python_file = md5(file("${path.module}/web_app/app.py"))
   docker_file = md5(file("${path.module}/web_app/backend.Dockerfile"))
 }
 
 provisioner "local-exec" {
   command = <<EOF
           aws ecr get-login-password --profile twitter --region ${var.region} | docker login --username AWS --password-stdin ${local.account_id}.dkr.ecr.${var.region}.amazonaws.com
           docker build -t ${aws_ecr_repository.repo.repository_url}:${local.ecr_image_tag} -f web_app/backend.Dockerfile .
           docker push ${aws_ecr_repository.repo.repository_url}:${local.ecr_image_tag}
       EOF
 }
}
 
data aws_ecr_image lambda_image {
 depends_on = [
   null_resource.ecr_image
 ]
 repository_name = local.ecr_repository_name
 image_tag       = local.ecr_image_tag
}
 
resource aws_iam_role lambda {
 name = "twitter-analyzer-lambda-role"
 assume_role_policy = <<EOF
{
   "Version": "2012-10-17",
   "Statement": [
       {
           "Action": "sts:AssumeRole",
           "Principal": {
               "Service": "lambda.amazonaws.com"
           },
           "Effect": "Allow"
       }
   ]
}
 EOF
}
 
data aws_iam_policy_document lambda {
   statement {
     actions = [
         "logs:CreateLogGroup",
         "logs:CreateLogStream",
         "logs:PutLogEvents"
     ]
     effect = "Allow"
     resources = [ "*" ]
     sid = "CreateCloudWatchLogs"
   }
 
   statement {
     actions = [
         "codecommit:GitPull",
         "codecommit:GitPush",
         "codecommit:GitBranch",
         "codecommit:ListBranches",
         "codecommit:CreateCommit",
         "codecommit:GetCommit",
         "codecommit:GetCommitHistory",
         "codecommit:GetDifferences",
         "codecommit:GetReferences",
         "codecommit:BatchGetCommits",
         "codecommit:GetTree",
         "codecommit:GetObjectIdentifier",
         "codecommit:GetMergeCommit"
     ]
     effect = "Allow"
     resources = [ "*" ]
     sid = "CodeCommit"
   }
}
 
resource aws_iam_policy lambda {
   name = "twitter-analyzer-lambda-policy"
   path = "/"
   policy = data.aws_iam_policy_document.lambda.json
}
 
resource aws_lambda_function git {
 depends_on = [
   null_resource.ecr_image
 ]
 function_name = "twitter-analyzer-lambda"
 role = aws_iam_role.lambda.arn
 timeout = 300
 image_uri = "${aws_ecr_repository.repo.repository_url}@${data.aws_ecr_image.lambda_image.id}"
 package_type = "Image"
}
 
output "lambda_name" {
 value = aws_lambda_function.git.id
}