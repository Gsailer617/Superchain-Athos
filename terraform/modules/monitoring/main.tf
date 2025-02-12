resource "aws_ecs_task_definition" "monitoring" {
  family                   = "${var.environment}-monitoring"
  network_mode             = "awsvpc"
  requires_compatibilities = ["FARGATE"]
  cpu                      = var.monitoring_cpu
  memory                   = var.monitoring_memory
  execution_role_arn       = var.ecs_execution_role_arn
  task_role_arn           = aws_iam_role.monitoring_task_role.arn

  container_definitions = jsonencode([
    {
      name  = "prometheus"
      image = "prom/prometheus:latest"
      
      essential = true
      
      portMappings = [
        {
          containerPort = 9090
          hostPort      = 9090
          protocol      = "tcp"
        }
      ]

      mountPoints = [
        {
          sourceVolume  = "prometheus-config"
          containerPath = "/etc/prometheus"
          readOnly      = true
        },
        {
          sourceVolume  = "prometheus-data"
          containerPath = "/prometheus"
          readOnly      = false
        }
      ]

      environment = [
        {
          name  = "ENVIRONMENT"
          value = var.environment
        }
      ]

      logConfiguration = {
        logDriver = "awslogs"
        options = {
          awslogs-group         = "/ecs/${var.environment}-monitoring"
          awslogs-region        = var.aws_region
          awslogs-stream-prefix = "prometheus"
        }
      }
    },
    {
      name  = "grafana"
      image = "grafana/grafana:latest"
      
      essential = true
      
      portMappings = [
        {
          containerPort = 3000
          hostPort      = 3000
          protocol      = "tcp"
        }
      ]

      mountPoints = [
        {
          sourceVolume  = "grafana-data"
          containerPath = "/var/lib/grafana"
          readOnly      = false
        }
      ]

      environment = [
        {
          name  = "GF_SECURITY_ADMIN_PASSWORD"
          value = var.grafana_admin_password
        },
        {
          name  = "GF_USERS_ALLOW_SIGN_UP"
          value = "false"
        }
      ]

      logConfiguration = {
        logDriver = "awslogs"
        options = {
          awslogs-group         = "/ecs/${var.environment}-monitoring"
          awslogs-region        = var.aws_region
          awslogs-stream-prefix = "grafana"
        }
      }
    },
    {
      name  = "alertmanager"
      image = "prom/alertmanager:latest"
      
      essential = true
      
      portMappings = [
        {
          containerPort = 9093
          hostPort      = 9093
          protocol      = "tcp"
        }
      ]

      mountPoints = [
        {
          sourceVolume  = "alertmanager-config"
          containerPath = "/etc/alertmanager"
          readOnly      = true
        }
      ]

      environment = [
        {
          name  = "SLACK_API_URL"
          value = var.alertmanager_slack_url
        }
      ]

      logConfiguration = {
        logDriver = "awslogs"
        options = {
          awslogs-group         = "/ecs/${var.environment}-monitoring"
          awslogs-region        = var.aws_region
          awslogs-stream-prefix = "alertmanager"
        }
      }
    }
  ])

  volume {
    name = "prometheus-config"
    efs_volume_configuration {
      file_system_id = aws_efs_file_system.monitoring.id
      root_directory = "/prometheus/config"
    }
  }

  volume {
    name = "prometheus-data"
    efs_volume_configuration {
      file_system_id = aws_efs_file_system.monitoring.id
      root_directory = "/prometheus/data"
    }
  }

  volume {
    name = "grafana-data"
    efs_volume_configuration {
      file_system_id = aws_efs_file_system.monitoring.id
      root_directory = "/grafana"
    }
  }

  volume {
    name = "alertmanager-config"
    efs_volume_configuration {
      file_system_id = aws_efs_file_system.monitoring.id
      root_directory = "/alertmanager"
    }
  }

  tags = {
    Environment = var.environment
  }
}

resource "aws_efs_file_system" "monitoring" {
  creation_token = "${var.environment}-monitoring-efs"
  encrypted      = true

  lifecycle_policy {
    transition_to_ia = "AFTER_30_DAYS"
  }

  tags = {
    Environment = var.environment
  }
}

resource "aws_efs_mount_target" "monitoring" {
  count           = length(var.private_subnet_ids)
  file_system_id  = aws_efs_file_system.monitoring.id
  subnet_id       = var.private_subnet_ids[count.index]
  security_groups = [aws_security_group.efs.id]
}

resource "aws_security_group" "efs" {
  name        = "${var.environment}-monitoring-efs-sg"
  description = "Allow EFS access from ECS tasks"
  vpc_id      = var.vpc_id

  ingress {
    description     = "NFS from ECS tasks"
    from_port       = 2049
    to_port         = 2049
    protocol        = "tcp"
    security_groups = [var.ecs_security_group_id]
  }

  tags = {
    Environment = var.environment
  }
}

resource "aws_iam_role" "monitoring_task_role" {
  name = "${var.environment}-monitoring-task-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "ecs-tasks.amazonaws.com"
        }
      }
    ]
  })
}

resource "aws_iam_role_policy" "monitoring_task_policy" {
  name = "${var.environment}-monitoring-task-policy"
  role = aws_iam_role.monitoring_task_role.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "ecs:ListClusters",
          "ecs:ListServices",
          "ecs:DescribeServices",
          "ec2:DescribeInstances",
          "elasticache:DescribeCacheClusters"
        ]
        Resource = "*"
      }
    ]
  })
}

resource "aws_cloudwatch_log_group" "monitoring" {
  name              = "/ecs/${var.environment}-monitoring"
  retention_in_days = 30

  tags = {
    Environment = var.environment
  }
} 