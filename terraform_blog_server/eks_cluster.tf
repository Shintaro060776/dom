resource "aws_iam_role" "eks_cluster_role_blog_server" {
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

resource "aws_iam_role" "eks_node_role_blog_server" {
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

resource "aws_iam_policy" "eks_describe_instances_blog_server" {
  name        = "eks-describe-instance_blog_servers"
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

resource "aws_iam_role_policy_attachment" "cloudwatch_logs_policy_blog_server" {
  role       = aws_iam_role.eks_cluster_role_blog_server.name
  policy_arn = "arn:aws:iam::aws:policy/CloudWatchLogsFullAccess"
}

resource "aws_iam_role_policy_attachment" "eks_describe_instances_attachment_blog_server" {
  role       = aws_iam_role.eks_cluster_role_blog_server.name
  policy_arn = aws_iam_policy.eks_describe_instances.arn
}

resource "aws_iam_role_policy_attachment" "AmazonEKSWorkerNodePolicy_blog_server" {
  role       = aws_iam_role.eks_node_role_blog_server.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonEKSWorkerNodePolicy"
}

resource "aws_iam_role_policy_attachment" "AmazonEKSCNIPolicy_blog_server" {
  role       = aws_iam_role.eks_node_role_blog_server.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonEKS_CNI_Policy"
}

resource "aws_iam_role_policy_attachment" "AmazonEC2ContainerRegistryReadOnly_blog_server" {
  role       = aws_iam_role.eks_node_role_blog_server.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonEC2ContainerRegistryReadOnly"
}

resource "aws_security_group" "blog_server" {
  name        = var.eks_security_group_name
  description = "Security Group for EKS Cluster"
  vpc_id      = aws_vpc.blog_server.id

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

resource "aws_eks_cluster" "blog_server" {
  name     = var.eks_cluster_name
  role_arn = aws_iam_role.eks_cluster_role_blog_server.arn

  vpc_config {
    security_group_ids      = [aws_security_group.blog_server.id]
    subnet_ids              = [aws_subnet.blog_server.id, aws_subnet.blog_server2.id]
    endpoint_private_access = true
    endpoint_public_access  = true
    public_access_cidrs     = ["0.0.0.0/0"]
  }

  enabled_cluster_log_types = ["api", "audit", "authenticator", "controllerManager", "scheduler"]

  depends_on = [
    aws_iam_role_policy_attachment.cloudwatch_logs_policy_blog_server
  ]
}

resource "aws_eks_node_group" "blog_server" {
  cluster_name    = aws_eks_cluster.blog_server.name
  node_group_name = var.eks_node_group_name
  node_role_arn   = aws_iam_role.eks_node_role_blog_server.arn
  subnet_ids      = [aws_subnet.blog_server.id, aws_subnet.blog_server2.id]

  scaling_config {
    desired_size = var.eks_node_group_desired_size
    max_size     = var.eks_node_group_max_size
    min_size     = var.eks_node_group_min_size
  }

  depends_on = [
    aws_iam_role_policy_attachment.AmazonEKSWorkerNodePolicy_blog_server,
    aws_iam_role_policy_attachment.AmazonEKSCNIPolicy_blog_server,
    aws_iam_role_policy_attachment.AmazonEC2ContainerRegistryReadOnly_blog_server,
  ]
}