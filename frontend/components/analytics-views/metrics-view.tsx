"use client"

import { useState } from 'react';
import { useDashboard } from '@/components/dashboard-context';
import { 
  BarChart3, 
  Code2, 
  FileCode2, 
  GitBranch, 
  Settings, 
  MessageSquare, 
  FileText, 
  Code, 
  RefreshCcw, 
  PaintBucket, 
  Brain 
} from 'lucide-react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Bar, BarChart, ResponsiveContainer, XAxis, YAxis } from 'recharts';

export function MetricsView() {
  const { repoData, commitData } = useDashboard();
  const [hoveredCard, setHoveredCard] = useState<string | null>(null);

  const handleMouseEnter = (cardName: string) => {
    setHoveredCard(cardName);
  };

  const handleMouseLeave = () => {
    setHoveredCard(null);
  };

  function calculateCodebaseGrade(data: any) {
    const { maintainabilityIndex } = data;
    
    if (!maintainabilityIndex || isNaN(maintainabilityIndex)) return 'N/A';
    
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

  if (!repoData) {
    return (
      <div className="flex items-center justify-center h-full">
        <p className="text-muted-foreground">Please analyze a repository to view metrics</p>
      </div>
    );
  }

  // Ensure commitData is valid for the chart
  const validCommitData = Array.isArray(commitData) && commitData.length > 0 
    ? commitData 
    : [{ month: 'No Data', commits: 0 }];

  return (
    <div className="space-y-6">
      <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Repository</CardTitle>
            <GitBranch className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{repoData.name || 'Unknown Repository'}</div>
            <p className="text-xs text-muted-foreground mt-1">{repoData.description || 'No description available'}</p>
            <div className="grid grid-cols-3 gap-4 mt-4">
              <div className="flex items-center">
                <FileCode2 className="h-4 w-4 text-muted-foreground mr-2" />
                <span className="text-sm font-medium">{(repoData.numberOfFiles || 0).toLocaleString()} Files</span>
              </div>
              <div className="flex items-center">
                <Code className="h-4 w-4 text-muted-foreground mr-2" />
                <span className="text-sm font-medium">{(repoData.numberOfFunctions || 0).toLocaleString()} Functions</span>
              </div>
              <div className="flex items-center">
                <BarChart3 className="h-4 w-4 text-muted-foreground mr-2" />
                <span className="text-sm font-medium">{(repoData.numberOfClasses || 0).toLocaleString()} Classes</span>
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
            <div className="text-2xl font-bold">{repoData.maintainabilityIndex?.toFixed(1) || 'N/A'}</div>
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
            <div className="text-2xl font-bold">{repoData.cyclomaticComplexity?.toFixed(1) || 'N/A'}</div>
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
            <div className="text-2xl font-bold">{(repoData.halsteadVolume || 0).toLocaleString()}</div>
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
            <div className="text-2xl font-bold">{repoData.depthOfInheritance?.toFixed(1) || 'N/A'}</div>
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
            <div className="text-2xl font-bold">{(repoData.linesOfCode || 0).toLocaleString()}</div>
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
            <div className="text-2xl font-bold">{(repoData.sloc || 0).toLocaleString()}</div>
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
            <div className="text-2xl font-bold">{(repoData.lloc || 0).toLocaleString()}</div>
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
            <div className="text-2xl font-bold">{(repoData.commentDensity || 0).toFixed(1)}%</div>
            <p className="text-xs text-muted-foreground mt-1">
              {hoveredCard === 'Comment Density' ? 'This is the percentage of the lines in the codebase that are comments.' : 'Percentage of comments in code'}
            </p>
          </CardContent>
        </Card>
      </div>
      <Card className="mt-6">
        <CardHeader>
          <CardTitle>Monthly Commits</CardTitle>
          <CardDescription>Number of commits, batched by month over the past year</CardDescription>
        </CardHeader>
        <CardContent className="pt-4">
          <ResponsiveContainer width="100%" height={300}>
          <BarChart data={validCommitData}>
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
            {(repoData.numberOfFiles || 0) > 1000 ? "Large" : (repoData.numberOfFiles || 0) > 100 ? "Moderate" : "Small"}
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}

