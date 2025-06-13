// API service for codebase analytics
import { 
  RepoRequest, 
  AnalysisResponse, 
  CodebaseStats, 
  FunctionContext, 
  ExplorationData,
  BlastRadiusData,
  FileNode
} from './api-types';

// Get backend URL - uses environment variable or falls back to defaults
const getBackendUrl = () => {
  // Check for environment variable first
  if (typeof process !== 'undefined' && process.env.NEXT_PUBLIC_BACKEND_URL) {
    return process.env.NEXT_PUBLIC_BACKEND_URL;
  }
  
  // In development, try local backend first
  if (typeof window !== 'undefined' && window.location.hostname === 'localhost') {
    // Connect to our deployed backend on port 8000
    return 'http://localhost:8000';
  }
  
  // Fallback to Modal deployment for production
  return 'https://zeeeepa--analytics-app-fastapi-modal-app-dev.modal.run';
};

// Parse repo URL to get owner/repo format
export const parseRepoUrl = (input: string): string => {
  if (input.includes('github.com')) {
    try {
      const url = new URL(input);
      const pathParts = url.pathname.split('/').filter(Boolean);
      if (pathParts.length >= 2) {
        return `${pathParts[0]}/${pathParts[1]}`;
      }
    } catch (e) {
      // If URL parsing fails, just return the input
      console.error('Error parsing URL:', e);
    }
  }
  return input;
};

// API methods
export const apiService = {
  // Analyze repository
  async analyzeRepo(repoUrl: string): Promise<AnalysisResponse> {
    const parsedRepoUrl = parseRepoUrl(repoUrl);
    const response = await fetch(`${getBackendUrl()}/analyze_repo`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Accept': 'application/json'
      },
      body: JSON.stringify({ repo_url: parsedRepoUrl }),
    });

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    return await response.json();
  },

  // Get codebase stats
  async getCodebaseStats(codebaseId: string): Promise<CodebaseStats> {
    const response = await fetch(`${getBackendUrl()}/get_codebase_stats/${codebaseId}`);
    
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    return await response.json();
  },

  // Get function context
  async getFunctionContext(functionId: string): Promise<FunctionContext> {
    const response = await fetch(`${getBackendUrl()}/get_function_context/${functionId}`);
    
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    return await response.json();
  },

  // Get function call chain
  async getFunctionCallChain(functionId: string): Promise<string[]> {
    const response = await fetch(`${getBackendUrl()}/get_function_call_chain/${functionId}`);
    
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    return await response.json();
  },

  // Visual exploration
  async exploreVisual(repoUrl: string, mode: string = 'error_focused'): Promise<ExplorationData> {
    const parsedRepoUrl = parseRepoUrl(repoUrl);
    const response = await fetch(`${getBackendUrl()}/explore_visual`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        repo_url: parsedRepoUrl,
        mode: mode
      })
    });

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    return await response.json();
  },

  // Blast radius analysis
  async analyzeBlastRadius(repoUrl: string, symbolName: string): Promise<BlastRadiusData> {
    const parsedRepoUrl = parseRepoUrl(repoUrl || '.');
    const response = await fetch(`${getBackendUrl()}/analyze_blast_radius`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        repo_url: parsedRepoUrl,
        symbol_name: symbolName
      })
    });

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    return await response.json();
  },

  // Get repository structure
  async getRepoStructure(repoUrl: string): Promise<FileNode> {
    const parsedRepoUrl = parseRepoUrl(repoUrl);
    const response = await fetch(`${getBackendUrl()}/get_repo_structure`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        repo_url: parsedRepoUrl
      })
    });

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    return await response.json();
  }
};

