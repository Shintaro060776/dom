resource "aws_iam_role" "eks_cluster_role_blog" {
  name = var.eks_cluster_role_name

  assume_role_policy = jsonencode({
    Version = "2012-10-17",
    Statement = [{
      Effect = "Allow",
      Principal = {
        Service = "eks.amazonaws.com"
      },
      Action = "sts:AssumeRole"
    }]
  })
}

resource "aws_iam_role" "eks_node_role_blog" {
  name = var.eks_node_role_name

  assume_role_policy = jsonencode({
    Version = "2012-10-17",
    Statement = [{
      Effect = "Allow",
      Principal = {
        Service = "ec2.amazonaws.com"
      },
      Action = "sts:AssumeRole"
    }]
  })
}

resource "aws_iam_policy" "eks_describe_instances_blog" {
  name        = "eks-describe-instance_blogs"
  description = "Allows EKS cluster role to describe EC2 instances"

  policy = jsonencode({
    Version = "2012-10-17",
    Statement = [
      {
        Effect   = "Allow",
        Action   = "ec2:DescribeInstances",
        Resource = "*"
      },
    ]
  })
}

resource "aws_iam_role_policy_attachment" "cloudwatch_logs_policy_blog" {
  role       = aws_iam_role.eks_cluster_role_blog.name
  policy_arn = "arn:aws:iam::aws:policy/CloudWatchLogsFullAccess"
}

resource "aws_iam_role_policy_attachment" "eks_describe_instances_attachment_blog" {
  role       = aws_iam_role.eks_cluster_role_blog.name
  policy_arn = aws_iam_policy.eks_describe_instances_blog.arn
}

resource "aws_iam_role_policy_attachment" "AmazonEKSWorkerNodePolicy_blog" {
  role       = aws_iam_role.eks_node_role_blog.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonEKSWorkerNodePolicy"
}

resource "aws_iam_role_policy_attachment" "AmazonEKSCNIPolicy_blog" {
  role       = aws_iam_role.eks_node_role_blog.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonEKS_CNI_Policy"
}

resource "aws_iam_role_policy_attachment" "AmazonEC2ContainerRegistryReadOnly_blog" {
  role       = aws_iam_role.eks_node_role_blog.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonEC2ContainerRegistryReadOnly"
}

resource "aws_security_group" "blog" {
  name        = var.eks_security_group_name
  description = "Security Group for EKS Cluster"
  vpc_id      = data.aws_vpc.next.id

  ingress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

resource "aws_eks_cluster" "blog" {
  name     = var.eks_cluster_name
  role_arn = aws_iam_role.eks_cluster_role_blog.arn

  vpc_config {
    security_group_ids      = [data.aws_security_group.vpc_endpoint.id]
    subnet_ids              = [data.aws_subnet.next.id, data.aws_subnet.next2.id]
    endpoint_private_access = true
    endpoint_public_access  = true
    public_access_cidrs     = ["0.0.0.0/0"]
  }

  enabled_cluster_log_types = ["api", "audit", "authenticator", "controllerManager", "scheduler"]

  depends_on = [
    aws_iam_role_policy_attachment.cloudwatch_logs_policy_blog
  ]
}

resource "aws_eks_node_group" "blog" {
  cluster_name    = aws_eks_cluster.blog.name
  node_group_name = var.eks_node_group_name
  node_role_arn   = aws_iam_role.eks_node_role_blog.arn
  subnet_ids      = [data.aws_subnet.next.id, data.aws_subnet.next2.id]

  scaling_config {
    desired_size = var.eks_node_group_desired_size
    max_size     = var.eks_node_group_max_size
    min_size     = var.eks_node_group_min_size
  }

  depends_on = [
    aws_iam_role_policy_attachment.AmazonEKSWorkerNodePolicy_blog,
    aws_iam_role_policy_attachment.AmazonEKSCNIPolicy_blog,
    aws_iam_role_policy_attachment.AmazonEC2ContainerRegistryReadOnly_blog,
  ]
}