"use client"

import { create } from 'zustand'
import { Issue, IssueSeverity, IssueCategory, IssueType, VisualNode } from '@/lib/api-types'

interface SharedAnalysisState {
  // Selected items for cross-component communication
  selectedIssue: Issue | null
  selectedNode: VisualNode | null
  selectedFilePath: string | null
  selectedSymbolName: string | null
  
  // Filters that can be applied across different views
  severityFilter: IssueSeverity | 'all'
  categoryFilter: IssueCategory | 'all'
  typeFilter: IssueType | 'all'
  searchQuery: string
  
  // View state that can be shared across components
  sortBy: 'impact' | 'severity' | 'type' | 'name'
  viewMode: 'list' | 'tree' | 'graph' | 'table'
  
  // Navigation history for back/forward navigation
  navigationHistory: string[]
  currentHistoryIndex: number
  
  // Actions
  setSelectedIssue: (issue: Issue | null) => void
  setSelectedNode: (node: VisualNode | null) => void
  setSelectedFilePath: (path: string | null) => void
  setSelectedSymbolName: (name: string | null) => void
  
  setSeverityFilter: (severity: IssueSeverity | 'all') => void
  setCategoryFilter: (category: IssueCategory | 'all') => void
  setTypeFilter: (type: IssueType | 'all') => void
  setSearchQuery: (query: string) => void
  
  setSortBy: (sort: 'impact' | 'severity' | 'type' | 'name') => void
  setViewMode: (mode: 'list' | 'tree' | 'graph' | 'table') => void
  
  navigateTo: (path: string) => void
  navigateBack: () => void
  navigateForward: () => void
  
  // Reset all filters
  resetFilters: () => void
  
  // Reset all state
  resetAll: () => void
}

export const useSharedAnalysisState = create<SharedAnalysisState>((set, get) => ({
  // Initial state
  selectedIssue: null,
  selectedNode: null,
  selectedFilePath: null,
  selectedSymbolName: null,
  
  severityFilter: 'all',
  categoryFilter: 'all',
  typeFilter: 'all',
  searchQuery: '',
  
  sortBy: 'impact',
  viewMode: 'list',
  
  navigationHistory: [],
  currentHistoryIndex: -1,
  
  // Actions
  setSelectedIssue: (issue) => set({ selectedIssue: issue }),
  setSelectedNode: (node) => set({ selectedNode: node }),
  setSelectedFilePath: (path) => set({ selectedFilePath: path }),
  setSelectedSymbolName: (name) => set({ selectedSymbolName: name }),
  
  setSeverityFilter: (severity) => set({ severityFilter: severity }),
  setCategoryFilter: (category) => set({ categoryFilter: category }),
  setTypeFilter: (type) => set({ typeFilter: type }),
  setSearchQuery: (query) => set({ searchQuery: query }),
  
  setSortBy: (sort) => set({ sortBy: sort }),
  setViewMode: (mode) => set({ viewMode: mode }),
  
  navigateTo: (path) => {
    const { navigationHistory, currentHistoryIndex } = get()
    
    // Remove any forward history if we're navigating from a point in history
    const newHistory = navigationHistory.slice(0, currentHistoryIndex + 1)
    newHistory.push(path)
    
    set({
      navigationHistory: newHistory,
      currentHistoryIndex: newHistory.length - 1
    })
  },
  
  navigateBack: () => {
    const { currentHistoryIndex } = get()
    if (currentHistoryIndex > 0) {
      set({ currentHistoryIndex: currentHistoryIndex - 1 })
    }
  },
  
  navigateForward: () => {
    const { navigationHistory, currentHistoryIndex } = get()
    if (currentHistoryIndex < navigationHistory.length - 1) {
      set({ currentHistoryIndex: currentHistoryIndex + 1 })
    }
  },
  
  resetFilters: () => set({
    severityFilter: 'all',
    categoryFilter: 'all',
    typeFilter: 'all',
    searchQuery: '',
    sortBy: 'impact'
  }),
  
  resetAll: () => set({
    selectedIssue: null,
    selectedNode: null,
    selectedFilePath: null,
    selectedSymbolName: null,
    
    severityFilter: 'all',
    categoryFilter: 'all',
    typeFilter: 'all',
    searchQuery: '',
    
    sortBy: 'impact',
    viewMode: 'list',
    
    navigationHistory: [],
    currentHistoryIndex: -1
  })
}))

// Helper hooks for specific use cases
export function useIssueSelection() {
  const { selectedIssue, setSelectedIssue } = useSharedAnalysisState()
  return { selectedIssue, setSelectedIssue }
}

export function useNodeSelection() {
  const { selectedNode, setSelectedNode } = useSharedAnalysisState()
  return { selectedNode, setSelectedNode }
}

export function useFileSelection() {
  const { selectedFilePath, setSelectedFilePath } = useSharedAnalysisState()
  return { selectedFilePath, setSelectedFilePath }
}

export function useSymbolSelection() {
  const { selectedSymbolName, setSelectedSymbolName } = useSharedAnalysisState()
  return { selectedSymbolName, setSelectedSymbolName }
}

export function useFilters() {
  const {
    severityFilter,
    categoryFilter,
    typeFilter,
    searchQuery,
    setSeverityFilter,
    setCategoryFilter,
    setTypeFilter,
    setSearchQuery,
    resetFilters
  } = useSharedAnalysisState()
  
  return {
    severityFilter,
    categoryFilter,
    typeFilter,
    searchQuery,
    setSeverityFilter,
    setCategoryFilter,
    setTypeFilter,
    setSearchQuery,
    resetFilters
  }
}

export function useViewOptions() {
  const { sortBy, viewMode, setSortBy, setViewMode } = useSharedAnalysisState()
  return { sortBy, viewMode, setSortBy, setViewMode }
}

export function useNavigation() {
  const {
    navigationHistory,
    currentHistoryIndex,
    navigateTo,
    navigateBack,
    navigateForward
  } = useSharedAnalysisState()
  
  const currentPath = navigationHistory[currentHistoryIndex] || null
  const canGoBack = currentHistoryIndex > 0
  const canGoForward = currentHistoryIndex < navigationHistory.length - 1
  
  return {
    currentPath,
    canGoBack,
    canGoForward,
    navigateTo,
    navigateBack,
    navigateForward
  }
}

