terraform {
  backend "s3" {
    bucket = "dom20090317"
    key    = "terraform.tfstate"
    region = "ap-northeast-1"
  }
}
