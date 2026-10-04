#!/bin/bash
# RouteMem AI Gateway — AWS CloudWatch Auto-Stop Alarm for GPU Worker Instance
set -e

INSTANCE_ID=$1
REGION=${2:-"us-east-1"}

if [ -z "$INSTANCE_ID" ]; then
    echo "Usage: ./aws_autostop_alarm.sh <INSTANCE_ID> [REGION]"
    exit 1
fi

echo "=== Creating AWS CloudWatch Auto-Stop Alarm for $INSTANCE_ID ==="

aws cloudwatch put-metric-alarm \
  --region "$REGION" \
  --alarm-name "AutoStop-GPU-Worker-$INSTANCE_ID" \
  --alarm-description "Automatically stops Spot GPU instance when CPU utilization drops below 5% for 30 minutes" \
  --metric-name CPUUtilization \
  --namespace AWS/EC2 \
  --statistic Average \
  --period 300 \
  --evaluation-periods 6 \
  --threshold 5.0 \
  --comparison-operator LessThanOrEqualToThreshold \
  --dimensions Name=InstanceId,Value="$INSTANCE_ID" \
  --alarm-actions "arn:aws:automate:$REGION:ec2:stop"

echo "[✔] CloudWatch Auto-Stop Alarm Created for $INSTANCE_ID!"
