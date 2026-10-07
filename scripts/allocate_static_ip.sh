#!/usr/bin/env bash
# RouteMem AI Gateway — AWS Elastic IP (Static IP) Setup
# Allocates and associates an Elastic IP to instance i-0720d9efd39dc72a8

set -e

INSTANCE_ID="i-0720d9efd39dc72a8"
REGION="us-east-1"

echo "=== AWS Elastic IP Allocation for RouteMem Control Plane ==="

# Check for existing unassociated Elastic IPs
UNASSOCIATED_ALLOC_ID=$(aws ec2 describe-addresses --region $REGION \
  --filters "Name=domain,Values=vpc" \
  --query "Addresses[?AssociationId==null].AllocationId" \
  --output text | awk '{print $1}')

if [ -n "$UNASSOCIATED_ALLOC_ID" ] && [ "$UNASSOCIATED_ALLOC_ID" != "None" ]; then
    echo "[*] Found unassociated Elastic IP Allocation ID: $UNASSOCIATED_ALLOC_ID"
    ALLOC_ID=$UNASSOCIATED_ALLOC_ID
else
    echo "[*] Allocating new VPC Elastic IP in $REGION..."
    ALLOC_OUTPUT=$(aws ec2 allocate-address --domain vpc --region $REGION --output json)
    ALLOC_ID=$(echo "$ALLOC_OUTPUT" | jq -r '.AllocationId')
    PUBLIC_IP=$(echo "$ALLOC_OUTPUT" | jq -r '.PublicIp')
    echo "[+] Allocated Elastic IP: $PUBLIC_IP (Allocation ID: $ALLOC_ID)"
fi

echo "[*] Associating Elastic IP ($ALLOC_ID) with instance $INSTANCE_ID..."
ASSOC_ID=$(aws ec2 associate-address --instance-id $INSTANCE_ID --allocation-id $ALLOC_ID --region $REGION --query "AssociationId" --output text)
echo "[+] Successfully associated Elastic IP! Association ID: $ASSOC_ID"

# Get newly assigned Public IP
PERMANENT_IP=$(aws ec2 describe-instances --instance-ids $INSTANCE_ID --region $REGION --query "Reservations[0].Instances[0].PublicIpAddress" --output text)
echo "=========================================================="
echo "[+] PERMANENT ROUTEMEM GATEWAY IP: http://$PERMANENT_IP:8000"
echo "=========================================================="
