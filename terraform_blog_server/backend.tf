terraform {
  backend "s3" {
    bucket = "blogserver20090317part2"
    key    = "terraform.tfstate"
    region = "ap-northeast-1"
  }
}