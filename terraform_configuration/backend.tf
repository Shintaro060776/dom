terraform {
    backend "s3" {
        bucket = "next20090317"
        key    = "terraform.tfstate"
        region = "ap-northeast-1"
    }
}