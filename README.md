# Codebase Analytics

A powerful codebase analysis tool with graph-sitter integration for deep code intelligence.

## 🚀 Quick Start (Local Development)

### Prerequisites
- Python 3.11+
- Node.js 18+
- Git

### 1. Clone and Setup
```bash
git clone https://github.com/Zeeeepa/codebase-analytics.git
cd codebase-analytics
```

### 2. Backend Setup
```bash
cd backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Frontend Setup
```bash
cd ../frontend
npm install
```

### 4. Start Development Servers

**Option A: Use the convenience script**
```bash
# From project root
./start-dev.sh
```

**Option B: Manual start**
```bash
# Terminal 1 - Backend
cd backend
source venv/bin/activate
python -m uvicorn api:app --reload --host 0.0.0.0 --port 8000

# Terminal 2 - Frontend
cd frontend
npm run dev
```

### 5. Access the Application
- Frontend: http://localhost:3000
- Backend API: http://localhost:8000
- API Docs: http://localhost:8000/docs

## 🔧 Features

### Core Analysis Functions
- **`get_codebase_summary()`** - High-level statistical overview
- **`get_file_summary()`** - Single file dependency analysis
- **`get_class_summary()`** - Deep class analysis with relationships
- **`get_function_summary()`** - Comprehensive function analysis
- **`get_symbol_summary()`** - Universal symbol usage analysis
- **`generate_context()`** - AI context generation from graph-sitter

### Graph-Sitter Integration
- Real-time code parsing and analysis
- Symbol dependency tracking
- Import resolution
- Cross-reference analysis

## 📁 Project Structure
```
codebase-analytics/
├── backend/
│   ├── api.py              # FastAPI application with analysis endpoints
│   ├── requirements.txt    # Python dependencies
│   └── venv/              # Virtual environment
├── frontend/
│   ├── app/               # Next.js app directory
│   ├── components/        # React components
│   ├── package.json       # Node.js dependencies
│   └── node_modules/      # Node dependencies
├── start-dev.sh           # Development startup script
└── README.md              # This file
```

## 🛠 Development

### Backend Development
```bash
cd backend
source venv/bin/activate

# Install new dependencies
pip install package-name
pip freeze > requirements.txt

# Run with auto-reload
python -m uvicorn api:app --reload --host 0.0.0.0 --port 8000
```

### Frontend Development
```bash
cd frontend

# Install new dependencies
npm install package-name

# Run development server
npm run dev

# Build for production
npm run build
```

### Environment Variables
Create `.env.local` in the frontend directory:
```bash
NEXT_PUBLIC_API_URL=http://localhost:8000
```

## 🧪 Testing

### Backend Testing
```bash
cd backend
source venv/bin/activate

# Test API endpoints
curl http://localhost:8000/health
curl http://localhost:8000/docs
```

### Frontend Testing
```bash
cd frontend
npm run build  # Test build process
npm run lint   # Run linting
```

## 📊 API Endpoints

### Core Endpoints
- `GET /` - Root endpoint
- `GET /health` - Health check
- `POST /analyze` - Analyze repository
- `GET /docs` - Interactive API documentation

### Analysis Functions
All analysis functions are integrated into the `/analyze` endpoint and provide:
- Codebase statistics and metrics
- File-level dependency analysis
- Class and function breakdowns
- Symbol usage patterns
- AI-ready context generation

## 🔍 Usage Examples

### Analyze a Repository
```bash
curl -X POST "http://localhost:8000/analyze" \
  -H "Content-Type: application/json" \
  -d '{"repo_url": "https://github.com/user/repo"}'
```

### Get Analysis Results
The analysis includes:
- File counts and structure
- Import dependencies
- Symbol relationships
- Code complexity metrics
- AI context for each component

## 🚀 Production Deployment

For production deployment, you can:
1. Use a process manager like PM2 for the backend
2. Build and serve the frontend with a web server
3. Set up a reverse proxy (nginx/apache)
4. Configure environment variables appropriately

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test locally
5. Submit a pull request

## 📝 License

This project is open source and available under the MIT License.

