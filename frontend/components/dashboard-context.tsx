"use client"

import React, { createContext, useContext, useState, ReactNode } from 'react';
import { RepoData, ExplorationData, BlastRadiusData, Symbol, FileNode } from '@/lib/api-types';

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
}

const DashboardContext = createContext<DashboardContextType | undefined>(undefined);

export function DashboardProvider({ children }: { children: ReactNode }) {
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
  };

  return (
    <DashboardContext.Provider value={value}>
      {children}
    </DashboardContext.Provider>
  );
}

export function useDashboard() {
  const context = useContext(DashboardContext);
  if (context === undefined) {
    throw new Error('useDashboard must be used within a DashboardProvider');
  }
  return context;
}

