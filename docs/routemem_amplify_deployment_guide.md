# RouteMem Frontend — AWS Amplify Deployment Guide & Plan

> **Overview**: This guide provides step-by-step instructions for deploying the Next.js 15 **RouteMem AI Gateway Dashboard** (`frontend/`) to **AWS Amplify Hosting** with automatic continuous deployment (CI/CD) linked to your GitHub repository (`shreeshailchavan/routemem-gateway`).

---

## 1. Prerequisites & Environment Setup

- **GitHub Repository**: [shreeshailchavan/routemem-gateway](https://github.com/shreeshailchavan/routemem-gateway) (Branch: `main`)
- **Active Backend Gateway IP**: `http://34.229.80.244:8000` (AWS EC2 `t4g.xlarge` in `us-east-1`)
- **Next.js App Subdirectory**: `frontend`

---

## 2. AWS Amplify Build Configuration (`amplify.yml`)

When deploying a Next.js monorepo application where the frontend resides in `frontend/`, AWS Amplify requires the following `amplify.yml` build spec:

```yaml
version: 1
frontend:
  phases:
    preBuild:
      commands:
        - cd frontend
        - npm ci
    build:
      commands:
        - env | grep -E 'GATEWAY_BACKEND_URL|NEXT_PUBLIC_' >> .env.production
        - npm run build
  artifacts:
    baseDirectory: frontend/.next
    files:
      - '**/*'
  cache:
    paths:
      - frontend/node_modules/**/*
      - frontend/.next/cache/**/*
```

---

## 3. Step-by-Step Deployment Options

### Option A: AWS Amplify Console (GitHub UI Deployment — Recommended)

1. **Log in to AWS Management Console**:
   - Navigate to **AWS Amplify** in region `us-east-1`.
2. **Create New App**:
   - Click **Host web app**.
   - Select **GitHub** as the source provider and authorize AWS Amplify.
   - Choose repository: `shreeshailchavan/routemem-gateway` and branch: `main`.
3. **Configure Monorepo Settings**:
   - Enable **Connecting a monorepo**.
   - Set **App root directory** to: `frontend`.
4. **Environment Variables**:
   - Add Environment Variable:
     - `GATEWAY_BACKEND_URL`: `http://34.229.80.244:8000`
5. **Deploy & Launch**:
   - Click **Save and Deploy**. Amplify will automatically trigger pre-build, Next.js compilation, and assign a custom HTTPS domain (e.g., `https://main.d123456abcdef.amplifyapp.com`).

---

### Option B: AWS CLI Deployment (Headless Automation)

If deploying via AWS CLI from your local terminal:

```bash
# 1. Create Amplify App
aws amplify create-app \
  --name "routemem-frontend" \
  --repository "https://github.com/shreeshailchavan/routemem-gateway" \
  --platform "WEB_COMPUTE" \
  --region us-east-1

# 2. Add Environment Variables & Branch
aws amplify create-branch \
  --app-id <YOUR_APP_ID> \
  --branch-name main \
  --enable-auto-build \
  --region us-east-1

# 3. Trigger Initial Build
aws amplify start-job \
  --app-id <YOUR_APP_ID> \
  --branch-name main \
  --job-type SUSTAINED \
  --region us-east-1
```

---

## 4. Verification & Post-Deployment Testing

1. **HTTPS Domain Access**: Visit your generated Amplify domain (`https://main.xxxx.amplifyapp.com`).
2. **Gateway API Proxy**: Submit a test prompt in the **Chat Studio**. Next.js server-side `/api/chat` route will forward the request to `http://34.229.80.244:8000/v1/chat/completions`.
3. **Real-time Pipeline Tracing**: Confirm 8-stage execution steps, TTFT latency meters, and cost savings telemetry render properly.
