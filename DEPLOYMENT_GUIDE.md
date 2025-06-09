# 🚀 Enhanced Deployment Guide

This guide covers all deployment options for the Enhanced Codebase Analytics application, including local development, Docker containers, and Modal serverless deployment.

## 📋 Prerequisites

### Required Software
- **Python 3.11+**: Backend runtime
- **Node.js 18+**: Frontend development
- **Git**: Repository management
- **Docker** (optional): Container deployment
- **Modal CLI** (optional): Serverless deployment

### Installation Commands
```bash
# Python (via pyenv recommended)
curl https://pyenv.run | bash
pyenv install 3.11.0
pyenv global 3.11.0

# Node.js (via nvm recommended)
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.0/install.sh | bash
nvm install 18
nvm use 18

# Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sh get-docker.sh

# Modal CLI
pip install modal
modal token new  # Authenticate with Modal
```

## 🎯 Deployment Options

### 1. 🔧 Local Development Deployment

**Best for**: Development, testing, and local analysis

```bash
# Quick start with automatic dependency installation
./dev-deploy.sh --install-deps

# Backend only (for frontend development)
./dev-deploy.sh --backend-only

# Frontend only (if backend is running elsewhere)
./dev-deploy.sh --frontend-only
```

**Manual Setup**:
```bash
# Terminal 1 - Backend
cd backend
pip install -r requirements.txt
python api.py

# Terminal 2 - Frontend  
cd frontend
npm install
npm run dev
```

**Access Points**:
- Frontend: http://localhost:3000
- Backend API: http://localhost:8000
- API Documentation: http://localhost:8000/docs

### 2. 🐳 Docker Container Deployment

**Best for**: Production environments, consistent deployments

```bash
# Development environment
./deploy.sh --env development

# Production environment with rebuild
./deploy.sh --env production --rebuild

# Show logs during deployment
./deploy.sh --env production --logs
```

**Manual Docker Commands**:
```bash
# Build and start all services
docker-compose up -d --build

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

**Access Points**:
- Frontend: http://localhost (via Nginx)
- Backend API: http://localhost/api
- API Documentation: http://localhost/api/docs

### 3. ⚡ Modal Serverless Deployment

**Best for**: Scalable production, serverless architecture, cost optimization

```bash
# Development mode (temporary endpoint)
./deploy-modal.sh --serve

# Production deployment (persistent endpoint)
./deploy-modal.sh --deploy --env production

# Backend only to Modal
./deploy-modal.sh --deploy --backend-only

# Frontend only (using existing Modal backend)
./deploy-modal.sh --serve --frontend-only
```

**Manual Modal Setup**:
```bash
# Authenticate with Modal
modal token new

# Deploy backend
cd backend
modal deploy modal_api.py

# Start frontend with Modal backend
cd frontend
npm run dev
```

**Access Points**:
- Frontend: http://localhost:3000
- Backend API: https://your-modal-app.modal.run
- API Documentation: https://your-modal-app.modal.run/docs

## 🔧 Configuration Options

### Environment Variables

**Backend Configuration**:
```bash
# .env file in backend/
PYTHONUNBUFFERED=1
LOG_LEVEL=INFO
MAX_REPO_SIZE=1000000000  # 1GB limit
ANALYSIS_TIMEOUT=600      # 10 minutes
```

**Frontend Configuration**:
```bash
# .env.local file in frontend/
NEXT_PUBLIC_API_URL=http://localhost:8000
NEXT_PUBLIC_DEPLOYMENT_MODE=local
NEXT_PUBLIC_ANALYTICS_ENABLED=true
```

### Docker Configuration

**Development Override**:
```yaml
# docker-compose.override.yml
version: '3.8'
services:
  backend:
    volumes:
      - ./backend:/app
    environment:
      - DEBUG=true
  frontend:
    volumes:
      - ./frontend:/app
    environment:
      - NODE_ENV=development
```

**Production Optimization**:
```yaml
# docker-compose.prod.yml
version: '3.8'
services:
  backend:
    deploy:
      replicas: 3
      resources:
        limits:
          memory: 1G
          cpus: '0.5'
  frontend:
    deploy:
      replicas: 2
      resources:
        limits:
          memory: 512M
          cpus: '0.25'
```

### Modal Configuration

**Resource Allocation**:
```python
# In backend/modal_api.py
@app.function(
    image=image,
    timeout=600,      # 10 minutes
    memory=2048,      # 2GB RAM
    cpu=2.0,          # 2 CPU cores
    retries=3         # Retry failed requests
)
```

## 🚀 Deployment Workflows

### Development Workflow
1. **Start Local Development**:
   ```bash
   ./dev-deploy.sh --install-deps
   ```

2. **Make Changes**: Edit code in your IDE

3. **Test Changes**: 
   - Backend: http://localhost:8000/health
   - Frontend: http://localhost:3000

4. **Commit and Push**:
   ```bash
   git add .
   git commit -m "Your changes"
   git push
   ```

### Production Workflow
1. **Test Locally**:
   ```bash
   ./dev-deploy.sh --install-deps
   # Test your changes
   ```

2. **Deploy to Staging** (Docker):
   ```bash
   ./deploy.sh --env development --rebuild
   ```

3. **Deploy to Production** (Modal):
   ```bash
   ./deploy-modal.sh --deploy --env production
   ```

4. **Monitor Deployment**:
   - Check health endpoints
   - Monitor logs
   - Verify functionality

### CI/CD Integration

**GitHub Actions Example**:
```yaml
# .github/workflows/deploy.yml
name: Deploy to Production

on:
  push:
    branches: [main]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Setup Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'
          
      - name: Setup Node.js
        uses: actions/setup-node@v3
        with:
          node-version: '18'
          
      - name: Install Modal
        run: pip install modal
        
      - name: Deploy to Modal
        env:
          MODAL_TOKEN: ${{ secrets.MODAL_TOKEN }}
        run: |
          modal token set $MODAL_TOKEN
          ./deploy-modal.sh --deploy --env production
```

## 🔍 Monitoring and Troubleshooting

### Health Checks

**Backend Health**:
```bash
curl http://localhost:8000/health
# Expected: {"status": "healthy", "version": "2.0.0"}
```

**Frontend Health**:
```bash
curl http://localhost:3000
# Expected: HTML response with dashboard
```

**Docker Health**:
```bash
docker-compose ps
# All services should show "healthy"
```

### Common Issues

**Port Conflicts**:
```bash
# Check what's using ports
lsof -i :3000
lsof -i :8000

# Kill processes if needed
kill -9 $(lsof -ti:3000)
kill -9 $(lsof -ti:8000)
```

**Docker Issues**:
```bash
# Clean up Docker
docker system prune -a
docker-compose down --volumes --remove-orphans

# Rebuild from scratch
docker-compose build --no-cache
```

**Modal Issues**:
```bash
# Check Modal status
modal app list
modal token list

# Re-authenticate if needed
modal token new

# Check logs
modal app logs enhanced-codebase-analytics
```

### Performance Optimization

**Backend Optimization**:
- Increase worker processes for high load
- Implement Redis caching for repeated analyses
- Use database for storing analysis results

**Frontend Optimization**:
- Enable Next.js static generation
- Implement service worker for offline support
- Use CDN for static assets

**Modal Optimization**:
- Adjust memory and CPU allocation
- Implement function warming
- Use Modal volumes for large dependencies

## 📊 Scaling Considerations

### Horizontal Scaling

**Docker Swarm**:
```bash
# Initialize swarm
docker swarm init

# Deploy stack
docker stack deploy -c docker-compose.yml analytics

# Scale services
docker service scale analytics_backend=3
docker service scale analytics_frontend=2
```

**Kubernetes**:
```yaml
# k8s-deployment.yml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: analytics-backend
spec:
  replicas: 3
  selector:
    matchLabels:
      app: analytics-backend
  template:
    metadata:
      labels:
        app: analytics-backend
    spec:
      containers:
      - name: backend
        image: analytics-backend:latest
        ports:
        - containerPort: 8000
```

### Load Balancing

**Nginx Configuration**:
```nginx
upstream backend {
    server backend1:8000;
    server backend2:8000;
    server backend3:8000;
}

server {
    location /api/ {
        proxy_pass http://backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

## 🔒 Security Considerations

### Production Security Checklist
- [ ] Enable HTTPS with SSL certificates
- [ ] Configure proper CORS origins
- [ ] Implement rate limiting
- [ ] Set up authentication if needed
- [ ] Use environment variables for secrets
- [ ] Enable security headers
- [ ] Regular dependency updates
- [ ] Monitor for vulnerabilities

### SSL/TLS Setup
```bash
# Using Let's Encrypt with Certbot
sudo apt install certbot python3-certbot-nginx
sudo certbot --nginx -d your-domain.com
```

## 📈 Monitoring and Analytics

### Application Monitoring
- Health check endpoints
- Performance metrics
- Error tracking
- User analytics

### Infrastructure Monitoring
- Resource usage (CPU, memory, disk)
- Network performance
- Container health
- Service availability

---

## 🎯 Quick Reference

| Deployment Type | Command | Use Case | Access URL |
|----------------|---------|----------|------------|
| Local Dev | `./dev-deploy.sh --install-deps` | Development | http://localhost:3000 |
| Docker Dev | `./deploy.sh --env development` | Testing | http://localhost |
| Docker Prod | `./deploy.sh --env production` | Production | http://localhost |
| Modal Dev | `./deploy-modal.sh --serve` | Serverless Dev | http://localhost:3000 |
| Modal Prod | `./deploy-modal.sh --deploy` | Serverless Prod | https://your-modal-app.modal.run |

**Need help?** Check the troubleshooting section or create an issue on GitHub! 🚀

