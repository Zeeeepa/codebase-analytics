"use client"

import { useState, useEffect } from 'react';
import { useDashboard } from '@/components/dashboard-context';
import { apiService } from '@/lib/api-service';
import { FileTree, FolderTree } from 'lucide-react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { Progress } from '@/components/ui/progress';
import RepoStructure from '@/components/RepoStructure';
import SymbolContext from '@/components/SymbolContext';

export function StructureView() {
  const { 
    repoUrl, 
    repoStructure, 
    setRepoStructure, 
    isLoading, 
    setIsLoading, 
    error, 
    setError 
  } = useDashboard();
  const [selectedSymbol, setSelectedSymbol] = useState<any>(null);
  const [selectedFile, setSelectedFile] = useState<string>('');

  const fetchRepoStructure = async () => {
    if (!repoUrl.trim()) {
      setError("Please enter a repository URL first");
      return;
    }

    setIsLoading(true);
    setError("");
    
    try {
      const data = await apiService.getRepoStructure(repoUrl);
      setRepoStructure(data);
    } catch (error) {
      console.error('Error fetching repository structure:', error);
      setError('Failed to fetch repository structure. Please check the URL and try again.');
    } finally {
      setIsLoading(false);
    }
  };

  const handleFileClick = (path: string) => {
    setSelectedFile(path);
    setSelectedSymbol(null);
  };

  const handleFolderClick = (path: string) => {
    // Could be used to fetch additional data about the folder
    console.log("Folder clicked:", path);
  };

  const handleSymbolClick = (symbol: any) => {
    setSelectedSymbol(symbol);
  };

  const handleViewCallChain = (symbolId: string) => {
    // Implement call chain visualization
    console.log("View call chain for:", symbolId);
  };

  const handleViewContext = (symbolId: string) => {
    // Implement context visualization
    console.log("View context for:", symbolId);
  };

  if (isLoading) {
    return (
      <Card>
        <CardContent className="pt-6">
          <div className="space-y-2">
            <div className="flex justify-between text-sm">
              <span>Loading repository structure...</span>
              <span>Please wait</span>
            </div>
            <Progress value={undefined} className="w-full" />
          </div>
        </CardContent>
      </Card>
    );
  }

  if (!repoStructure) {
    return (
      <div className="space-y-4">
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <FileTree className="h-5 w-5" />
              Repository Structure
            </CardTitle>
            <CardDescription>
              Explore the structure of your codebase
            </CardDescription>
          </CardHeader>
          <CardContent>
            <Button 
              onClick={fetchRepoStructure} 
              className="flex items-center gap-2"
            >
              <FolderTree className="h-4 w-4" />
              Load Repository Structure
            </Button>
            
            {error && (
              <Alert variant="destructive" className="mt-4">
                <AlertDescription>{error}</AlertDescription>
              </Alert>
            )}
          </CardContent>
        </Card>
      </div>
    );
  }

  return (
    <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
      <div className="md:col-span-1">
        <RepoStructure
          data={repoStructure}
          onFileClick={handleFileClick}
          onFolderClick={handleFolderClick}
          onSymbolClick={handleSymbolClick}
          onViewCallChain={handleViewCallChain}
          onViewContext={handleViewContext}
        />
      </div>
      <div className="md:col-span-2">
        {selectedSymbol ? (
          <Card>
            <CardHeader>
              <CardTitle>Symbol Details</CardTitle>
              <CardDescription>
                Detailed information about the selected symbol
              </CardDescription>
            </CardHeader>
            <CardContent>
              <SymbolContext
                id={selectedSymbol.id}
                name={selectedSymbol.name}
                type={selectedSymbol.type}
                filepath={selectedSymbol.filepath}
                start_line={selectedSymbol.start_line}
                end_line={selectedSymbol.end_line}
                source={null} // Would need to fetch this
                usage_stats={{
                  total_usages: 0,
                  usage_breakdown: {
                    functions: 0,
                    classes: 0,
                    global_vars: 0,
                    interfaces: 0,
                  },
                  imports: {
                    total: 0,
                    breakdown: {
                      functions: 0,
                      classes: 0,
                      global_vars: 0,
                      interfaces: 0,
                      external_modules: 0,
                      files: 0,
                    },
                  },
                }}
              />
            </CardContent>
          </Card>
        ) : selectedFile ? (
          <Card>
            <CardHeader>
              <CardTitle>File Details</CardTitle>
              <CardDescription>{selectedFile}</CardDescription>
            </CardHeader>
            <CardContent>
              <p>File content would be displayed here</p>
            </CardContent>
          </Card>
        ) : (
          <Card>
            <CardHeader>
              <CardTitle>Repository Overview</CardTitle>
              <CardDescription>Select a file or symbol to view details</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-2 gap-4">
                <div className="p-4 border rounded-lg">
                  <h3 className="font-medium">Files</h3>
                  <p className="text-2xl font-bold">{repoStructure.stats.files}</p>
                </div>
                <div className="p-4 border rounded-lg">
                  <h3 className="font-medium">Directories</h3>
                  <p className="text-2xl font-bold">{repoStructure.stats.directories}</p>
                </div>
                <div className="p-4 border rounded-lg">
                  <h3 className="font-medium">Symbols</h3>
                  <p className="text-2xl font-bold">{repoStructure.stats.symbols}</p>
                </div>
                <div className="p-4 border rounded-lg">
                  <h3 className="font-medium">Issues</h3>
                  <p className="text-2xl font-bold">{repoStructure.stats.issues}</p>
                </div>
              </div>
            </CardContent>
          </Card>
        )}
      </div>
    </div>
  );
}

