# Codebase Analytics

A web application that provides comprehensive analytics for GitHub repositories.  The project combines a Modal-based FastAPI backend with a Next.js frontend to provide efficient and beautiful codebase metrics.

Users submit a GitHub repository through the frontend. The Modal API processes the request using custom implementations of the metric calculations using the `codegen` library. Results are returned to the frontend for display.

## How it Works

### Backend (Modal API)

The backend is built using [Modal](https://modal.com/) and [FastAPI](https://fastapi.tiangolo.com/), providing a serverless API endpoint for code research.

There is a main API endpoint that handles codebase analysis requests. It uses the `codegen` library for these operations.

The following metrics are calculated:
- `Maintainability Index`: Measures the maintainability of the codebase, on a scale of 0-100.
- `Cyclomatic Complexity`: Measures the complexity of the codebase. Higher values indicate more complex code.
- `Halstead Volume`: Quantifies the information content in the code based on operators and operands. Higher values indicate more complex code.
- `Depth of Inheritance`: Measures the depth of inheritance of the codebase.
- `Lines of Code (LOC, SLOC, LLOC)`: Measures the number of lines of code in the codebase. LOC, SLOC, and LLOC represent the total, source, and logical lines of code, respectively. 
- `Comment Density`: Measures the density of comments in the codebase, given as a percentage of the total lines of code. More comments can indicate better documentation.

```python
complexity = calculate_cyclomatic_complexity(func)
operators, operands = get_operators_and_operands(func)
volume, _, _, _, _ = calculate_halstead_volume(operators, operands)
loc = len(func.code_block.source.splitlines())
mi_score = calculate_maintainability_index(volume, complexity, loc)
```

Monthly commit history is also pulled from the GitHub API and displayed in a chart.

The codebase is also graded on a scale of A-F for overall quality and maintainability. The complexity of the codebase is also given a qualitative rank.

### Frontend (Next.js)

The frontend provides an interface for users to submit a GitHub repository and research query. The components come from the [shadcn/ui](https://ui.shadcn.com/) library. This triggers the Modal API to perform the code research and returns the results to the frontend.

## Getting Started

### Simple 2-Command Setup

1. **Install Dependencies:**
   ```bash
   npm install
   ```

2. **Start the Application:**
   ```bash
   npm run dev
   ```
   This will start both the backend (port 9998) and frontend (port 9999) simultaneously.
   The UI will be available at `http://localhost:9999`.

### Manual Setup (Alternative)

If you prefer to start services separately:

1. **Start the Backend API:**
   ```bash
   python3 backend/api.py
   ```
   This starts the server at `http://localhost:9998` with real repository analysis capabilities.

2. **Start the Frontend:**
   ```bash
   cd frontend
   npm install
   npm run dev
   ```
   The frontend will be available at `http://localhost:9999`.

3. **Test the Integration:**
   ```bash
   python3 test_integration.py
   ```

### Legacy: Production Deployment with Modal (Deprecated)

1. **Deploy the Modal API:**
   ```bash
   modal serve backend/api.py
   ```
   `modal serve` runs the API locally for development, creating a temporary endpoint that's active only while the command is running.
   ```bash
   modal deploy backend/api.py
   ```
   `modal deploy` creates a persistent Modal app and deploys the FastAPI app to it, generating a permanent API endpoint.

2. **Configure Frontend for Production:**
   ```bash
   cd frontend
   # Update .env.local with your Modal app URL:
   # NEXT_PUBLIC_API_URL=https://your-modal-app-url.modal.run
   ```

3. **Build and Deploy Frontend:**
   ```bash
   cd frontend
   npm run build
   npm start
   ```

**Note:** Modal deployment is deprecated in favor of the simplified local setup above.

## Recent Improvements

### Frontend Enhancements
- ✅ **Environment Configuration**: Added support for configurable API endpoints via environment variables
- ✅ **Improved Error Handling**: Better error messages and user feedback for failed requests
- ✅ **Enhanced Loading States**: More informative loading screens showing which repository is being analyzed
- ✅ **New Analysis Button**: Easy way to start a new analysis from the results page
- ✅ **Better UX**: Improved responsive design and interactive elements

### Backend Improvements
- ✅ **Local Development API**: Created `test_api.py` for local development and testing
- ✅ **Real Repository Analysis**: Actual git repository cloning and analysis
- ✅ **Comprehensive Metrics**: Real line counting, commit history, and repository metadata
- ✅ **Error Handling**: Robust error handling for various failure scenarios
- ✅ **Health Checks**: Added health endpoint for monitoring

### Testing & Validation
- ✅ **Integration Tests**: Automated testing script to verify frontend-backend communication
- ✅ **Build Validation**: Ensures frontend builds successfully with all changes
- ✅ **API Testing**: Direct API endpoint testing with real repositories

## Learn More

More information about the `codegen` library can be found [here](https://codegen.com/).
