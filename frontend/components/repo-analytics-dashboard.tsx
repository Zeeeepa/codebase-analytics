"use client"

import { useState } from "react"
import { BarChart3, Code2, FileCode2, GitBranch, Github, Settings, MessageSquare, FileText, Code, RefreshCcw, PaintBucket, Brain, TreePine, RotateCcw } from "lucide-react"
import { Bar, BarChart, ResponsiveContainer, XAxis, YAxis } from "recharts"

import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import RepoStructure from "@/components/RepoStructure"
import SymbolContext from "@/components/SymbolContext"

const mockRepoData = {
  name: "vercel/next.js",
  description: "The React Framework for the Web",
  linesOfCode: 154321,
  cyclomaticComplexity: 15.7,
  depthOfInheritance: 3.2,
  halsteadVolume: 987654,
  maintainabilityIndex: 85,
  commentDensity: 18.5,
  sloc: 132456,
  lloc: 98765,
  numberOfFiles: 1200,
  numberOfFunctions: 4500,
  numberOfClasses: 300,
}

const mockCommitData = [
  { month: "October", commits: 130 },
  { month: "September", commits: 150 },
  { month: "August", commits: 120 },
  { month: "July", commits: 110 },
  { month: "June", commits: 140 },
  { month: "May", commits: 160 },
  { month: "April", commits: 170 },
  { month: "March", commits: 180 },
  { month: "February", commits: 190 },
  { month: "January", commits: 200 },
  { month: "December", commits: 210 },
  { month: "November", commits: 220 },
];

interface InheritanceAnalysis {
  deepest_class_name: string | null;
  deepest_class_depth: number;
  inheritance_chain: string[];
}

interface RecursionAnalysis {
  recursive_functions: string[];
  total_recursive_count: number;
}

interface RepoAnalyticsResponse {
  repo_url: string;
  line_metrics: {
    total: {
      loc: number;
      lloc: number;
      sloc: number;
      comments: number;
      comment_density: number;
    }
  };
  cyclomatic_complexity: { average: number };
  depth_of_inheritance: { average: number };
  halstead_metrics: { 
    total_volume: number;
    average_volume: number;
  };
  maintainability_index: { average: number };
  description: string;
  num_files: number;
  num_functions: number;
  num_classes: number;
  monthly_commits: Record<string, number>;
  inheritance_analysis: InheritanceAnalysis;
  recursion_analysis: RecursionAnalysis;
  repo_structure: any; // Repository structure data
}

interface RepoData {
  name: string;
  description: string;
  linesOfCode: number;
  cyclomaticComplexity: number;
  depthOfInheritance: number;
  halsteadVolume: number;
  maintainabilityIndex: number;
  commentDensity: number;
  sloc: number;
  lloc: number;
  numberOfFiles: number;
  numberOfFunctions: number;
  numberOfClasses: number;
  inheritance_analysis?: InheritanceAnalysis;
  recursion_analysis?: RecursionAnalysis;
}

export default function RepoAnalyticsDashboard() {
  const [repoUrl, setRepoUrl] = useState("")
  const [repoData, setRepoData] = useState(mockRepoData)
  const [hoveredCard, setHoveredCard] = useState<string | null>(null)
  const [commitData, setCommitData] = useState(mockCommitData)
  const [isLoading, setIsLoading] = useState(false)
  const [isLandingPage, setIsLandingPage] = useState(true)
  const [repoStructure, setRepoStructure] = useState<any>(null)
  const [selectedSymbol, setSelectedSymbol] = useState<any>(null)
  const [showSymbolContext, setShowSymbolContext] = useState(false)
  const [selectedFilePath, setSelectedFilePath] = useState<string | null>(null)
  const [expandedFolders, setExpandedFolders] = useState<Set<string>>(new Set())

  const parseRepoUrl = (input: string): string => {
    if (input.includes('github.com')) {
      const url = new URL(input);
      const pathParts = url.pathname.split('/').filter(Boolean);
      if (pathParts.length >= 2) {
        return `${pathParts[0]}/${pathParts[1]}`;
      }
    }
    return input;
  };

  // Get backend URL - uses environment variable or falls back to defaults
  const getBackendUrl = () => {
    // Check for environment variable first
    if (process.env.NEXT_PUBLIC_BACKEND_URL) {
      return process.env.NEXT_PUBLIC_BACKEND_URL;
    }
    
    // In development, try local backend first
    if (typeof window !== 'undefined' && window.location.hostname === 'localhost') {
      // Backend runs on port 8000
      return 'http://localhost:8000';
    }
    
    // Fallback to Modal deployment for production
    return 'https://zeeeepa--analytics-app-fastapi-modal-app-dev.modal.run';
  };

  const handleFetchRepo = async () => {
    console.log("Fetching repo data...");
    
    const parsedRepoUrl = parseRepoUrl(repoUrl);
    console.log(parsedRepoUrl);
    
    setIsLoading(true);
    setIsLandingPage(false);
    
    try {
      const backendUrl = getBackendUrl();
      console.log(`Using backend: ${backendUrl}`);
      
      console.log("Fetching repo data...");
      const response = await fetch(`${backendUrl}/analyze`, {
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

      const data: RepoAnalyticsResponse = await response.json();
      
      setRepoData({
        name: parsedRepoUrl,
        description: data.description,
        linesOfCode: data.line_metrics.total.loc,
        cyclomaticComplexity: data.cyclomatic_complexity.average,
        depthOfInheritance: data.depth_of_inheritance.average,
        halsteadVolume: data.halstead_metrics.total_volume,
        maintainabilityIndex: data.maintainability_index.average,
        commentDensity: data.line_metrics.total.comment_density,
        sloc: data.line_metrics.total.sloc,
        lloc: data.line_metrics.total.lloc,
        numberOfFiles: data.num_files,
        numberOfFunctions: data.num_functions,
        numberOfClasses: data.num_classes,
        inheritance_analysis: data.inheritance_analysis,
        recursion_analysis: data.recursion_analysis,
      });

      // Store repository structure
      setRepoStructure(data.repo_structure);

      const transformedCommitData = Object.entries(data.monthly_commits)
        .map(([date, commits]) => ({
          month: new Date(date).toLocaleString('default', { month: 'long' }),
          commits,
        }))
        .slice(0, 12)
        .reverse();

      setCommitData(transformedCommitData);
    } catch (error) {
      console.error('Error fetching repo data:', error);
      
      // Handle different types of errors
      let errorMessage = 'Error fetching repository data. Please check the URL and try again.';
      
      if (error instanceof Error) {
        if (error.message.includes('status: 408')) {
          errorMessage = 'Repository analysis timed out. The repository may be too large. Please try with a smaller repository.';
        } else if (error.message.includes('status: 404')) {
          errorMessage = 'Repository not found. Please check the URL and make sure the repository exists.';
        } else if (error.message.includes('status: 5')) {
          errorMessage = 'Server error occurred. Please try again later.';
        } else if (error.message.includes('fetch')) {
          errorMessage = 'Cannot connect to backend server. Please make sure the backend is running on port 8000.';
        }
      }
      
      alert(errorMessage);
      setIsLandingPage(true);
    } finally {
      setIsLoading(false);
    }
  };

  const handleMouseEnter = (cardName: string) => {
    setHoveredCard(cardName)
  }

  const handleMouseLeave = () => {
    setHoveredCard(null)
  }

  const handleKeyPress = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === 'Enter') {
      handleFetchRepo(); 
    }
  }

  // Enhanced Repository structure interaction handlers
  const handleFileClick = (path: string) => {
    console.log('File clicked:', path);
    setSelectedFilePath(path);
    
    // Could fetch file details from API if needed
    // For now, we'll just use the path to identify the file in the UI
  };

  const handleFolderClick = (path: string) => {
    console.log('Folder clicked:', path);
    
    // Toggle folder expansion state
    setExpandedFolders(prev => {
      const newSet = new Set(prev);
      if (newSet.has(path)) {
        newSet.delete(path);
      } else {
        newSet.add(path);
      }
      return newSet;
    });
  };

  const handleSymbolClick = async (symbol: any) => {
    console.log('Symbol clicked:', symbol);
    
    try {
      // Try to fetch detailed symbol information if available
      const backendUrl = getBackendUrl();
      const response = await fetch(`${backendUrl}/symbol/${symbol.id}/context`);
      
      if (response.ok) {
        const symbolDetails = await response.json();
        setSelectedSymbol(symbolDetails);
      } else {
        // If API fails, use the basic symbol info we already have
        setSelectedSymbol(symbol);
      }
      
      setShowSymbolContext(true);
    } catch (error) {
      console.error('Error fetching symbol details:', error);
      // Fallback to basic info
      setSelectedSymbol(symbol);
      setShowSymbolContext(true);
    }
  };

  const handleViewCallChain = async (symbolId: string) => {
    console.log('View call chain for symbol:', symbolId);
    
    try {
      const backendUrl = getBackendUrl();
      const response = await fetch(`${backendUrl}/function/${symbolId}/call-chain`);
      
      if (response.ok) {
        const callChainData = await response.json();
        
        // Update the selected symbol with call chain data
        setSelectedSymbol(prev => ({
          ...prev,
          function_info: {
            ...(prev.function_info || {}),
            call_chain: callChainData.call_chain
          }
        }));
        
        // Show the symbol context modal if not already shown
        setShowSymbolContext(true);
      }
    } catch (error) {
      console.error('Error fetching call chain:', error);
    }
  };

  const handleViewContext = async (symbolId: string) => {
    console.log('View context for symbol:', symbolId);
    
    try {
      const backendUrl = getBackendUrl();
      const response = await fetch(`${backendUrl}/symbol/${symbolId}/context`);
      
      if (response.ok) {
        const context = await response.json();
        setSelectedSymbol(context);
        setShowSymbolContext(true);
      }
    } catch (error) {
      console.error('Error fetching symbol context:', error);
      
      // Try to find the symbol in the repo structure as fallback
      if (repoStructure) {
        const findSymbolInNode = (node: any): any => {
          if (node.symbols) {
            const symbol = node.symbols.find((s: any) => s.id === symbolId);
            if (symbol) return symbol;
          }
          
          if (node.children) {
            for (const childKey in node.children) {
              const result = findSymbolInNode(node.children[childKey]);
              if (result) return result;
            }
          }
          
          return null;
        };
        
        const symbol = findSymbolInNode(repoStructure);
        if (symbol) {
          setSelectedSymbol(symbol);
          setShowSymbolContext(true);
        }
      }
    }
  };

function calculateCodebaseGrade(data: RepoData) {
  const { maintainabilityIndex } = data;
  
  if (maintainabilityIndex >= 90) return 'A+';
  if (maintainabilityIndex >= 85) return 'A';
  if (maintainabilityIndex >= 80) return 'A-';
  if (maintainabilityIndex >= 75) return 'B+';
  if (maintainabilityIndex >= 70) return 'B';
  if (maintainabilityIndex >= 65) return 'B-';
  if (maintainabilityIndex >= 60) return 'C+';
  if (maintainabilityIndex >= 55) return 'C';
  if (maintainabilityIndex >= 50) return 'C-';
  if (maintainabilityIndex >= 45) return 'D+';
  if (maintainabilityIndex >= 40) return 'D';
  return 'F';
}

  return (
    <div className="min-h-screen bg-gray-50">
      {isLandingPage ? (
        <div className="flex flex-col items-center justify-center min-h-screen p-4">
          <div className="text-center mb-8">
            <h1 className="text-4xl font-bold flex items-center justify-center gap-3 mb-4">
              <img src="cg.png" alt="CG Logo" className="h-12 w-12" />
              <span>Codebase Analytics</span>
            </h1>
            <p className="text-muted-foreground">Effortlessly calculate GitHub repository metrics in seconds</p>
          </div>
          <div className="flex items-center gap-3 w-full max-w-lg">
            <Input
              type="text"
              placeholder="Enter the GitHub repo link or owner/repo"
              value={repoUrl}
              onChange={(e) => setRepoUrl(e.target.value)}
              onKeyPress={handleKeyPress}
              className="flex-1"
              title="Format: https://github.com/owner/repo or owner/repo"
            />
            <Button 
              onClick={handleFetchRepo} 
              disabled={isLoading}
            >
              {isLoading ? "Loading..." : "Analyze"}
            </Button>
          </div>
          <footer className="absolute bottom-0 w-full text-center text-xs text-muted-foreground py-4">
            built with <a href="https://codegen.com" target="_blank" rel="noopener noreferrer" className="hover:text-primary">Codegen</a>
          </footer>
        </div>
      ) : isLoading ? (
        <div className="flex flex-col items-center justify-center min-h-screen">
          <div className="text-center mb-8">
            <h2 className="text-2xl font-bold mb-4">Analyzing Repository</h2>
            <p className="text-muted-foreground">Please wait while we calculate codebase metrics with Codegen...</p>
          </div>
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary" />
        </div>
      ) : (
        <div className="flex flex-col min-h-screen">
          <header className="sticky top-0 z-10 bg-white border-b">
            <div className="w-full px-8 py-4">
              <div className="flex items-center justify-between">
                <div className="flex-shrink-0">
                  <h1
                    className="text-2xl font-bold flex items-center space-x-3 cursor-pointer"
                    onClick={() => window.location.reload()}
                  >
                    <img src="cg.png" alt="CG Logo" className="h-8 w-8" />
                    <span>Codebase Analytics</span>
                  </h1>
                </div>
                <div className="flex items-center gap-3 ml-auto">
                  <Input
                    type="text"
                    placeholder="Enter the GitHub repo link or owner/repo"
                    value={repoUrl}
                    onChange={(e) => setRepoUrl(e.target.value)}
                    onKeyPress={handleKeyPress}
                    className="w-[320px]"
                    title="Format: https://github.com/owner/repo or owner/repo"
                  />
                  <Button onClick={handleFetchRepo} disabled={isLoading}>
                    {isLoading ? "Loading..." : "Analyze"}
                  </Button>
                </div>
              </div>
            </div>
          </header>
          <main className="flex-1 container mx-auto py-6 px-4 md:px-6">
            <div className="grid mb-5 gap-6 grid-cols-1">
              <Card>
                <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                  <CardTitle className="text-sm font-medium">Repository</CardTitle>
                  <Github className="h-4 w-4 text-muted-foreground" />
                </CardHeader>
                <CardContent>
                  <a href={`https://github.com/${repoData.name}`} target="_blank" rel="noopener noreferrer" className="block">
                    <div className="text-2xl font-bold">{repoData.name}</div>
                    <p className="text-xs text-muted-foreground mt-1">{repoData.description}</p>
                  </a>
                  <div className="flex flex-wrap mt-4 gap-4">
                    <div className="flex items-center">
                      <FileCode2 className="h-4 w-4 text-muted-foreground mr-2" />
                      <span className="text-sm font-medium">{repoData.numberOfFiles.toLocaleString()} Files</span>
                    </div>
                    <div className="flex items-center">
                      <Code className="h-4 w-4 text-muted-foreground mr-2" />
                      <span className="text-sm font-medium">{repoData.numberOfFunctions.toLocaleString()} Functions</span>
                    </div>
                    <div className="flex items-center">
                      <BarChart3 className="h-4 w-4 text-muted-foreground mr-2" />
                      <span className="text-sm font-medium">{repoData.numberOfClasses.toLocaleString()} Classes</span>
                    </div>
                  </div>
                </CardContent>
              </Card>
            </div>
            <div className="grid gap-6 md:grid-cols-4 lg:grid-cols-4 xl:grid-cols-4">
              <Card onMouseEnter={() => handleMouseEnter('Maintainability Index')} onMouseLeave={handleMouseLeave}>
                <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                  <CardTitle className="text-sm font-medium">Maintainability Index</CardTitle>
                  <Settings className="h-4 w-4 text-muted-foreground" />
                </CardHeader>
                <CardContent>
                  <div className="text-2xl font-bold">{repoData.maintainabilityIndex}</div>
                  <p className="text-xs text-muted-foreground mt-1">
                    {hoveredCard === 'Maintainability Index' ? 'This evaluates how easy it is to understand, modify, and maintain a codebase (ranging from 0 to 100).' : 'Code maintainability score (0-100)'}
                  </p>
                </CardContent>
              </Card>
              <Card onMouseEnter={() => handleMouseEnter('Cyclomatic Complexity')} onMouseLeave={handleMouseLeave}>
                <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                  <CardTitle className="text-sm font-medium">Cyclomatic Complexity</CardTitle>
                  <RefreshCcw className="h-4 w-4 text-muted-foreground" />
                </CardHeader>
                <CardContent>
                  <div className="text-2xl font-bold">{repoData.cyclomaticComplexity.toFixed(1)}</div>
                  <p className="text-xs text-muted-foreground mt-1">
                    {hoveredCard === 'Cyclomatic Complexity' ? 'This measures the number of independent paths through a program\'s source code' : 'Average complexity score'}
                  </p>
                </CardContent>
              </Card>
              <Card onMouseEnter={() => handleMouseEnter('Halstead Volume')} onMouseLeave={handleMouseLeave}>
                <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                  <CardTitle className="text-sm font-medium">Halstead Volume</CardTitle>
                  <PaintBucket className="h-4 w-4 text-muted-foreground" />
                </CardHeader>
                <CardContent>
                  <div className="text-2xl font-bold">{repoData.halsteadVolume.toLocaleString()}</div>
                  <p className="text-xs text-muted-foreground mt-1">
                    {hoveredCard === 'Halstead Volume' ? 'This quantifies the amount of information in a program by measuring the size and complexity of its code using operators and operands.' : 'Code volume metric'}
                  </p>
                </CardContent>
              </Card>
              <Card onMouseEnter={() => handleMouseEnter('Depth of Inheritance')} onMouseLeave={handleMouseLeave}>
                <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                  <CardTitle className="text-sm font-medium">Depth of Inheritance</CardTitle>
                  <GitBranch className="h-4 w-4 text-muted-foreground" />
                </CardHeader>
                <CardContent>
                  <div className="text-2xl font-bold">{repoData.depthOfInheritance.toFixed(1)}</div>
                  <p className="text-xs text-muted-foreground mt-1">
                    {hoveredCard === 'Depth of Inheritance' ? 'This is the average measure of the number of classes that a class inherits from.' : 'Average inheritance depth'}
                  </p>
                </CardContent>
              </Card>
              <Card onMouseEnter={() => handleMouseEnter('Lines of Code')} onMouseLeave={handleMouseLeave}>
                <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                  <CardTitle className="text-sm font-medium">Lines of Code</CardTitle>
                  <Code2 className="h-4 w-4 text-muted-foreground" />
                </CardHeader>
                <CardContent>
                  <div className="text-2xl font-bold">{repoData.linesOfCode.toLocaleString()}</div>
                  <p className="text-xs text-muted-foreground mt-1">
                    {hoveredCard === 'Lines of Code' ? 'This is the total number of lines of code within this codebase.' : 'Total lines in the repository'}
                  </p>
                </CardContent>
              </Card>
              <Card onMouseEnter={() => handleMouseEnter('SLOC')} onMouseLeave={handleMouseLeave}>
                <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                  <CardTitle className="text-sm font-medium">SLOC</CardTitle>
                  <FileText className="h-4 w-4 text-muted-foreground" />
                </CardHeader>
                <CardContent>
                  <div className="text-2xl font-bold">{repoData.sloc.toLocaleString()}</div>
                  <p className="text-xs text-muted-foreground mt-1">
                    {hoveredCard === 'SLOC' ? 'This is the number of textual lines of code within the codebase, ignoring whitespace and comments.' : 'Source Lines of Code'}
                  </p>
                </CardContent>
              </Card>
              <Card onMouseEnter={() => handleMouseEnter('LLOC')} onMouseLeave={handleMouseLeave}>
                <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                  <CardTitle className="text-sm font-medium">LLOC</CardTitle>
                  <Brain className="h-4 w-4 text-muted-foreground" />
                </CardHeader>
                <CardContent>
                  <div className="text-2xl font-bold">{repoData.lloc.toLocaleString()}</div>
                  <p className="text-xs text-muted-foreground mt-1">
                    {hoveredCard === 'LLOC' ? 'This is the number of lines of code that contribute to executable statements in the codebase.' : 'Logical Lines of Code'}
                  </p>
                </CardContent>
              </Card>
              <Card onMouseEnter={() => handleMouseEnter('Comment Density')} onMouseLeave={handleMouseLeave}>
                <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                  <CardTitle className="text-sm font-medium">Comment Density</CardTitle>
                  <MessageSquare className="h-4 w-4 text-muted-foreground" />
                </CardHeader>
                <CardContent>
                  <div className="text-2xl font-bold">{repoData.commentDensity.toFixed(1)}%</div>
                  <p className="text-xs text-muted-foreground mt-1">
                    {hoveredCard === 'Comment Density' ? 'This is the percentage of the lines in the codebase that are comments.' : 'Percentage of comments in code'}
                  </p>
                </CardContent>
              </Card>
            </div>
            
            {/* New Analysis Features */}
            <div className="grid gap-6 md:grid-cols-2 mt-6">
              <Card onMouseEnter={() => handleMouseEnter('Deepest Inheritance')} onMouseLeave={handleMouseLeave}>
                <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                  <CardTitle className="text-sm font-medium">Deepest Inheritance</CardTitle>
                  <TreePine className="h-4 w-4 text-muted-foreground" />
                </CardHeader>
                <CardContent>
                  {repoData.inheritance_analysis?.deepest_class_name ? (
                    <>
                      <div className="text-2xl font-bold">{repoData.inheritance_analysis.deepest_class_name}</div>
                      <p className="text-xs text-muted-foreground mt-1">
                        {hoveredCard === 'Deepest Inheritance' 
                          ? `Inheritance chain: ${repoData.inheritance_analysis.inheritance_chain.join(' → ')}` 
                          : `Depth: ${repoData.inheritance_analysis.deepest_class_depth} levels`}
                      </p>
                    </>
                  ) : (
                    <>
                      <div className="text-2xl font-bold text-muted-foreground">None</div>
                      <p className="text-xs text-muted-foreground mt-1">No class inheritance found</p>
                    </>
                  )}
                </CardContent>
              </Card>
              
              <Card onMouseEnter={() => handleMouseEnter('Recursive Functions')} onMouseLeave={handleMouseLeave}>
                <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                  <CardTitle className="text-sm font-medium">Recursive Functions</CardTitle>
                  <RotateCcw className="h-4 w-4 text-muted-foreground" />
                </CardHeader>
                <CardContent>
                  <div className="text-2xl font-bold">{repoData.recursion_analysis?.total_recursive_count || 0}</div>
                  <p className="text-xs text-muted-foreground mt-1">
                    {hoveredCard === 'Recursive Functions' 
                      ? (repoData.recursion_analysis?.recursive_functions?.length > 0 
                          ? `Functions: ${repoData.recursion_analysis.recursive_functions.join(', ')}` 
                          : 'No recursive functions found')
                      : 'Functions that call themselves'}
                  </p>
                </CardContent>
              </Card>
            </div>
            
            {/* Repository Structure */}
            {repoStructure && (
              <Card className="mt-6">
                <CardHeader>
                  <CardTitle>📂 Repository Structure</CardTitle>
                  <CardDescription>Interactive file tree with issue counts and symbol information</CardDescription>
                </CardHeader>
                <CardContent>
                  <RepoStructure
                    data={repoStructure}
                    onFileClick={handleFileClick}
                    onFolderClick={handleFolderClick}
                    onSymbolClick={handleSymbolClick}
                    onViewCallChain={handleViewCallChain}
                    onViewContext={handleViewContext}
                  />
                </CardContent>
              </Card>
            )}
            
            <Card className="mt-6">
              <CardHeader>
                <CardTitle>Monthly Commits</CardTitle>
                <CardDescription>Number of commits, batched by month over the past year</CardDescription>
              </CardHeader>
              <CardContent className="pt-4">
                <ResponsiveContainer width="100%" height={300}>
                <BarChart data={commitData}>
                    <XAxis dataKey="month" stroke="#888888" />
                    <YAxis stroke="#888888" />
                    <Bar dataKey="commits" fill="#2563eb" />
                  </BarChart>
                </ResponsiveContainer>
              </CardContent>
            </Card>
            <div className="grid gap-6 md:grid-cols-2">
              <Card className="mt-6">
                <CardContent className="pt-5 flex justify-between items-center">
                  <div>
                    <CardTitle>Codebase Grade</CardTitle>
                    <CardDescription>Overall grade based on code metrics</CardDescription>
                  </div>
                  <div className="text-4xl font-bold text-right">
                    {calculateCodebaseGrade(repoData)}
                  </div>
                </CardContent>
              </Card>
              <Card className="mt-6">
                <CardContent className="pt-5 flex justify-between items-center">
                  <div>
                    <CardTitle>Codebase Complexity</CardTitle>
                    <CardDescription>Judgment based on size and complexity</CardDescription>
                  </div>
                  <div className="text-2xl font-bold text-right">
                  {repoData.numberOfFiles > 1000 ? "Large" : "Moderate"}
                  </div>
                </CardContent>
              </Card>
            </div>
          </main>
          
          {/* Symbol Context Modal - Enhanced */}
          {showSymbolContext && selectedSymbol && (
            <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
              <div className="bg-background rounded-lg p-6 max-w-4xl max-h-[80vh] overflow-auto w-full">
                <div className="flex justify-between items-center mb-4">
                  <h2 className="text-xl font-bold">Symbol Details</h2>
                  <div className="flex gap-2">
                    {selectedSymbol.type === 'function' && (
                      <Button 
                        variant="outline" 
                        onClick={() => handleViewCallChain(selectedSymbol.id)}
                        className="text-sm"
                      >
                        View Call Chain
                      </Button>
                    )}
                    <Button 
                      variant="outline" 
                      onClick={() => setShowSymbolContext(false)}
                    >
                      Close
                    </Button>
                  </div>
                </div>
                <SymbolContext {...selectedSymbol} />
              </div>
            </div>
          )}
          
          <footer className="w-full text-center text-xs text-muted-foreground py-4">
          built with <a href="https://codegen.com" target="_blank" rel="noopener noreferrer" className="hover:text-primary">Codegen</a>
          </footer>
        </div>
      )}
    </div>
  )
}
