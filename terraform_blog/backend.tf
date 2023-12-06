terraform {
  backend "s3" {
    bucket = "blog20090317"
    key    = "terraform.tfstate"
    region = "ap-northeast-1"
  }
}