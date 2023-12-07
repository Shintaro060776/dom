resource "aws_cloudwatch_log_group" "ecs_log_group_blog" {
  name = "/ecs/blog-container"
}

resource "aws_iam_role" "ecs_execution_role_blog" {
  name = "ecsExecutionRole_blog"

  assume_role_policy = jsonencode({
    Version = "2012-10-17",
    Statement = [{
      Action = "sts:AssumeRole",
      Effect = "Allow",
      Principal = {
        Service = "ecs-tasks.amazonaws.com"
      }
    }]
  })
}

resource "aws_iam_role_policy" "ecs_execution_role_policy_blog" {
  name   = "ecsExecutionRolePolicy"
  role   = aws_iam_role.ecs_execution_role_blog.id
  policy = data.aws_iam_policy_document.ecs_execution_policy_blog.json
}

data "aws_iam_policy_document" "ecs_execution_policy_blog" {
  statement {
    actions = [
      "ecr:GetAuthorizationToken",
      "ecr:BatchCheckLayerAvailability",
      "ecr:GetDownloadUrlForLayer",
      "ecr:BatchGetImage",
      "ecr:GetRepositoryPolicy",
      "ecr:DescribeRepositories",
      "ecr:ListImages",
      "ecr:DescribeImages",
      "ecr:GetLifecyclePolicy",
      "ecr:GetLifecyclePolicyPreview",
      "ecr:ListTagsForResource",
      "ecr:DescribeImageScanFindings",
      "logs:CreateLogStream",
      "logs:PutLogEvents",
      "logs:CreateLogGroup",
      "secretsmanager:GetSecretValue",
      "kms:Decrypt",
      "ssm:GetParameters"
    ]

    resources = [
      "*"
    ]
  }
}

resource "aws_lb" "blog_lb" {
  name                       = var.aws_lb_name
  internal                   = false
  load_balancer_type         = "application"
  security_groups            = [aws_security_group.next.id]
  subnets                    = [aws_subnet.next.id, aws_subnet.next2.id]
  enable_deletion_protection = false
}

resource "aws_lb_listener" "front_end_blog" {
  load_balancer_arn = aws_lb.blog_lb.arn
  port              = var.aws_lb_listener_port
  protocol          = var.aws_lb_listener_protocol

  default_action {
    type             = "forward"
    target_group_arn = aws_lb_target_group.blog_tg.arn
  }
}
# 
resource "aws_lb_target_group" "blog_tg" {
  name        = var.aws_lb_target_group_name
  port        = 80
  protocol    = var.aws_lb_target_group_protocol
  vpc_id      = aws_vpc.next.id
  target_type = "ip"

  health_check {
    enabled             = true
    healthy_threshold   = 5
    unhealthy_threshold = 5
    timeout             = 5
    interval            = 10
    path                = "/"
    protocol            = "HTTP"
    matcher             = "200-299"
  }
}

resource "aws_ecs_cluster" "blog_cluster" {
  name = var.ecs_cluster_name
}

resource "aws_ecs_task_definition" "blog_task" {
  family                   = "blog_task"
  network_mode             = "awsvpc"
  requires_compatibilities = ["FARGATE"]
  cpu                      = "256"
  memory                   = "512"
  execution_role_arn       = aws_iam_role.ecs_execution_role_blog.arn

  container_definitions = jsonencode([{
    name      = "blog-container"
    image     = "715573459931.dkr.ecr.ap-northeast-1.amazonaws.com/blog:latest"
    essential = true
    memory    = 512
    cpu       = 256
    portMappings = [{
      containerPort = 80
      hostPort      = 80
    }]
    logConfiguration = {
      logDriver = "awslogs"
      options = {
        "awslogs-group"         = aws_cloudwatch_log_group.ecs_log_group_blog.name
        "awslogs-region"        = "ap-northeast-1"
        "awslogs-stream-prefix" = "ecs"
      }
    }
    secrets = [
      {
        name      = "USERNAME"
        valueFrom = "arn:aws:secretsmanager:ap-northeast-1:715573459931:secret:MySecretName-yAQ7uq:username::"
      },
      {
        name      = "PASSWORD"
        valueFrom = "arn:aws:secretsmanager:ap-northeast-1:715573459931:secret:MySecretName-yAQ7uq:password::"
      }
    ]
  }])
}

resource "aws_ecs_service" "blog_service" {
  name            = var.ecs_service_name
  cluster         = aws_ecs_cluster.blog_cluster.id
  task_definition = aws_ecs_task_definition.blog_task.arn
  launch_type     = "FARGATE"

  load_balancer {
    target_group_arn = aws_lb_target_group.blog_tg.arn
    container_name   = "blog-container"
    container_port   = 80
  }

  network_configuration {
    subnets         = [aws_subnet.next.id, aws_subnet.next2.id]
    security_groups = [aws_security_group.next.id]
  }

  desired_count = 1
}

resource "aws_iam_role_policy_attachment" "ecs_execution_role_policy_blog" {
  role       = aws_iam_role.ecs_execution_role_blog.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy"
}

resource "aws_iam_role_policy_attachment" "ecr_read_only_blog" {
  role       = aws_iam_role.ecs_execution_role_blog.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonEC2ContainerRegistryReadOnly"
}
