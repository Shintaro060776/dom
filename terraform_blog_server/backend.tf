terraform {
  backend "s3" {
    bucket = "blogserver20090317"
    key    = "terraform.tfstate"
    region = "ap-northeast-1"
  }
}