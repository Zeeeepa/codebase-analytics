"use client"

import React, { createContext, useContext, useState, ReactNode, useEffect } from 'react';
import { RepoData, ExplorationData, BlastRadiusData, Symbol, FileNode, Issue, VisualNode } from '@/lib/api-types';
import { useSharedAnalysisState } from '@/hooks/useSharedAnalysisState';

interface DashboardContextType {
  // Repository data
  repoUrl: string;
  setRepoUrl: (url: string) => void;
  repoData: RepoData | null;
  setRepoData: (data: RepoData | null) => void;
  
  // Analysis data
  explorationData: ExplorationData | null;
  setExplorationData: (data: ExplorationData | null) => void;
  blastRadiusData: BlastRadiusData | null;
  setBlastRadiusData: (data: BlastRadiusData | null) => void;
  
  // UI state
  activeView: string;
  setActiveView: (view: string) => void;
  isLoading: boolean;
  setIsLoading: (loading: boolean) => void;
  error: string;
  setError: (error: string) => void;
  
  // Selected items
  selectedSymbol: string;
  setSelectedSymbol: (symbol: string) => void;
  selectedFile: string;
  setSelectedFile: (file: string) => void;
  
  // Repository structure
  repoStructure: FileNode | null;
  setRepoStructure: (structure: FileNode | null) => void;
  
  // Analysis mode
  analysisMode: string;
  setAnalysisMode: (mode: string) => void;
  
  // Commit data
  commitData: Array<{month: string, commits: number}>;
  setCommitData: (data: Array<{month: string, commits: number}>) => void;
  
  // Enhanced functionality
  allIssues: Issue[];
  getAllIssues: () => Issue[];
  allNodes: VisualNode[];
  getAllNodes: () => VisualNode[];
  findNodeById: (id: string) => VisualNode | null;
  findIssueById: (id: string) => Issue | null;
  findRelatedIssues: (issueId: string) => Issue[];
  findRelatedNodes: (nodeId: string) => VisualNode[];
  
  // Navigation
  navigateToView: (view: string, params?: Record<string, any>) => void;
  navigateBack: () => boolean;
  
  // Data sharing
  shareDataBetweenViews: (key: string, data: any) => void;
  getSharedData: (key: string) => any;
}

const DashboardContext = createContext<DashboardContextType | undefined>(undefined);

export function DashboardProviderEnhanced({ children }: { children: ReactNode }) {
  // Repository data
  const [repoUrl, setRepoUrl] = useState<string>('');
  const [repoData, setRepoData] = useState<RepoData | null>(null);
  
  // Analysis data
  const [explorationData, setExplorationData] = useState<ExplorationData | null>(null);
  const [blastRadiusData, setBlastRadiusData] = useState<BlastRadiusData | null>(null);
  
  // UI state
  const [activeView, setActiveView] = useState<string>('metrics');
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [error, setError] = useState<string>('');
  
  // Selected items
  const [selectedSymbol, setSelectedSymbol] = useState<string>('');
  const [selectedFile, setSelectedFile] = useState<string>('');
  
  // Repository structure
  const [repoStructure, setRepoStructure] = useState<FileNode | null>(null);
  
  // Analysis mode
  const [analysisMode, setAnalysisMode] = useState<string>('error_focused');
  
  // Commit data
  const [commitData, setCommitData] = useState<Array<{month: string, commits: number}>>([]);
  
  // Navigation history
  const [navigationHistory, setNavigationHistory] = useState<Array<{view: string, params?: Record<string, any>}>>([]);
  const [currentHistoryIndex, setCurrentHistoryIndex] = useState<number>(-1);
  
  // Shared data between views
  const [sharedData, setSharedData] = useState<Record<string, any>>({});
  
  // Get shared state from zustand store
  const {
    selectedIssue,
    selectedNode,
    selectedFilePath,
    selectedSymbolName,
    setSelectedIssue,
    setSelectedNode,
    setSelectedFilePath,
    setSelectedSymbolName
  } = useSharedAnalysisState();
  
  // Sync dashboard context with shared state
  useEffect(() => {
    if (selectedSymbol && selectedSymbol !== selectedSymbolName) {
      setSelectedSymbolName(selectedSymbol);
    }
  }, [selectedSymbol, selectedSymbolName, setSelectedSymbolName]);
  
  useEffect(() => {
    if (selectedFile && selectedFile !== selectedFilePath) {
      setSelectedFilePath(selectedFile);
    }
  }, [selectedFile, selectedFilePath, setSelectedFilePath]);
  
  // Enhanced functionality
  const getAllIssues = (): Issue[] => {
    if (!explorationData) return [];
    
    const allIssues: Issue[] = [];
    
    // Collect issues from error hotspots
    if (explorationData.error_hotspots) {
      explorationData.error_hotspots.forEach(hotspot => {
        if (hotspot.issues && hotspot.issues.length > 0) {
          allIssues.push(...hotspot.issues);
        }
      });
    }
    
    return allIssues;
  };
  
  const getAllNodes = (): VisualNode[] => {
    const nodes: VisualNode[] = [];
    
    // Collect nodes from exploration data
    if (explorationData?.error_hotspots) {
      nodes.push(...explorationData.error_hotspots);
    }
    
    // Collect nodes from blast radius data
    if (blastRadiusData?.affected_nodes) {
      // Avoid duplicates by checking if node already exists
      blastRadiusData.affected_nodes.forEach(node => {
        if (!nodes.some(n => n.id === node.id)) {
          nodes.push(node);
        }
      });
      
      // Add target symbol if it's not already included
      if (blastRadiusData.target_symbol && !nodes.some(n => n.id === blastRadiusData.target_symbol.id)) {
        nodes.push(blastRadiusData.target_symbol);
      }
    }
    
    return nodes;
  };
  
  const findNodeById = (id: string): VisualNode | null => {
    const nodes = getAllNodes();
    return nodes.find(node => node.id === id) || null;
  };
  
  const findIssueById = (id: string): Issue | null => {
    const issues = getAllIssues();
    return issues.find(issue => issue.id === id) || null;
  };
  
  const findRelatedIssues = (issueId: string): Issue[] => {
    const issue = findIssueById(issueId);
    if (!issue) return [];
    
    const allIssues = getAllIssues();
    
    return allIssues.filter(otherIssue => {
      // Skip the current issue
      if (otherIssue.id === issueId) return false;
      
      // Check if in the same file
      const sameFile = otherIssue.location?.file_path === issue.location?.file_path;
      
      // Check if same type
      const sameType = otherIssue.type === issue.type;
      
      // Check if related symbols overlap
      const hasOverlappingSymbols = 
        issue.related_symbols && 
        otherIssue.related_symbols && 
        issue.related_symbols.some(symbol => otherIssue.related_symbols?.includes(symbol));
      
      return sameFile || sameType || hasOverlappingSymbols;
    });
  };
  
  const findRelatedNodes = (nodeId: string): VisualNode[] => {
    const node = findNodeById(nodeId);
    if (!node) return [];
    
    const allNodes = getAllNodes();
    
    return allNodes.filter(otherNode => {
      // Skip the current node
      if (otherNode.id === nodeId) return false;
      
      // Check if in the same file
      const sameFile = otherNode.path === node.path;
      
      // Check if same type
      const sameType = otherNode.type === node.type;
      
      // Check if there are common issues
      const hasCommonIssues = 
        node.issues && 
        otherNode.issues && 
        node.issues.some(issue1 => 
          otherNode.issues.some(issue2 => issue1.type === issue2.type)
        );
      
      return sameFile || sameType || hasCommonIssues;
    });
  };
  
  // Navigation
  const navigateToView = (view: string, params?: Record<string, any>) => {
    // Update active view
    setActiveView(view);
    
    // Add to navigation history
    const newHistoryEntry = { view, params };
    const newHistory = [...navigationHistory.slice(0, currentHistoryIndex + 1), newHistoryEntry];
    
    setNavigationHistory(newHistory);
    setCurrentHistoryIndex(newHistory.length - 1);
    
    // Apply params if provided
    if (params) {
      if (params.symbol) {
        setSelectedSymbol(params.symbol);
      }
      if (params.file) {
        setSelectedFile(params.file);
      }
      // Add more param handling as needed
    }
  };
  
  const navigateBack = (): boolean => {
    if (currentHistoryIndex > 0) {
      const previousEntry = navigationHistory[currentHistoryIndex - 1];
      setActiveView(previousEntry.view);
      setCurrentHistoryIndex(currentHistoryIndex - 1);
      
      // Apply params if provided
      if (previousEntry.params) {
        if (previousEntry.params.symbol) {
          setSelectedSymbol(previousEntry.params.symbol);
        }
        if (previousEntry.params.file) {
          setSelectedFile(previousEntry.params.file);
        }
        // Add more param handling as needed
      }
      
      return true;
    }
    
    return false;
  };
  
  // Data sharing
  const shareDataBetweenViews = (key: string, data: any) => {
    setSharedData(prev => ({
      ...prev,
      [key]: data
    }));
  };
  
  const getSharedData = (key: string) => {
    return sharedData[key];
  };
  
  // Memoized values
  const allIssues = getAllIssues();
  const allNodes = getAllNodes();

  const value = {
    repoUrl,
    setRepoUrl,
    repoData,
    setRepoData,
    explorationData,
    setExplorationData,
    blastRadiusData,
    setBlastRadiusData,
    activeView,
    setActiveView,
    isLoading,
    setIsLoading,
    error,
    setError,
    selectedSymbol,
    setSelectedSymbol,
    selectedFile,
    setSelectedFile,
    repoStructure,
    setRepoStructure,
    analysisMode,
    setAnalysisMode,
    commitData,
    setCommitData,
    
    // Enhanced functionality
    allIssues,
    getAllIssues,
    allNodes,
    getAllNodes,
    findNodeById,
    findIssueById,
    findRelatedIssues,
    findRelatedNodes,
    
    // Navigation
    navigateToView,
    navigateBack,
    
    // Data sharing
    shareDataBetweenViews,
    getSharedData
  };

  return (
    <DashboardContext.Provider value={value}>
      {children}
    </DashboardContext.Provider>
  );
}

export function useDashboardEnhanced() {
  const context = useContext(DashboardContext);
  if (context === undefined) {
    throw new Error('useDashboardEnhanced must be used within a DashboardProviderEnhanced');
  }
  return context;
}

