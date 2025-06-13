"use client"

import { useState, useCallback, useEffect } from 'react'

export interface AnalysisState {
  selectedMetric: string | null
  drillDownData: any
  filters: {
    severity?: 'good' | 'warning' | 'critical'
    category?: string
    dateRange?: { start: Date; end: Date }
  }
  viewMode: 'overview' | 'detailed' | 'comparison'
  zoomLevel: number
  isLoading: boolean
  error: string | null
}

export interface InteractiveAnalysisHook {
  state: AnalysisState
  actions: {
    selectMetric: (metric: string, data?: any) => void
    clearSelection: () => void
    setFilter: (key: keyof AnalysisState['filters'], value: any) => void
    clearFilters: () => void
    setViewMode: (mode: AnalysisState['viewMode']) => void
    setZoomLevel: (level: number) => void
    drillDown: (metric: string, value: any) => void
    goBack: () => void
  }
  computed: {
    hasActiveFilters: boolean
    isMetricSelected: boolean
    canGoBack: boolean
    filteredData: any
  }
}

const initialState: AnalysisState = {
  selectedMetric: null,
  drillDownData: null,
  filters: {},
  viewMode: 'overview',
  zoomLevel: 1,
  isLoading: false,
  error: null
}

export function useInteractiveAnalysis(): InteractiveAnalysisHook {
  const [state, setState] = useState<AnalysisState>(initialState)
  const [history, setHistory] = useState<AnalysisState[]>([])

  // Actions
  const selectMetric = useCallback((metric: string, data?: any) => {
    setState(prev => ({
      ...prev,
      selectedMetric: metric,
      drillDownData: data,
      viewMode: 'detailed'
    }))
  }, [])

  const clearSelection = useCallback(() => {
    setState(prev => ({
      ...prev,
      selectedMetric: null,
      drillDownData: null,
      viewMode: 'overview'
    }))
  }, [])

  const setFilter = useCallback((key: keyof AnalysisState['filters'], value: any) => {
    setState(prev => ({
      ...prev,
      filters: {
        ...prev.filters,
        [key]: value
      }
    }))
  }, [])

  const clearFilters = useCallback(() => {
    setState(prev => ({
      ...prev,
      filters: {}
    }))
  }, [])

  const setViewMode = useCallback((mode: AnalysisState['viewMode']) => {
    setState(prev => ({
      ...prev,
      viewMode: mode
    }))
  }, [])

  const setZoomLevel = useCallback((level: number) => {
    setState(prev => ({
      ...prev,
      zoomLevel: Math.max(0.5, Math.min(3, level))
    }))
  }, [])

  const drillDown = useCallback((metric: string, value: any) => {
    // Save current state to history
    setHistory(prev => [...prev, state])
    
    setState(prev => ({
      ...prev,
      selectedMetric: metric,
      drillDownData: value,
      viewMode: 'detailed'
    }))
  }, [state])

  const goBack = useCallback(() => {
    if (history.length > 0) {
      const previousState = history[history.length - 1]
      setState(previousState)
      setHistory(prev => prev.slice(0, -1))
    } else {
      clearSelection()
    }
  }, [history, clearSelection])

  // Computed values
  const hasActiveFilters = Object.keys(state.filters).length > 0
  const isMetricSelected = state.selectedMetric !== null
  const canGoBack = history.length > 0 || isMetricSelected

  // Filter data based on current filters
  const filteredData = useCallback((data: any[]) => {
    if (!data || !hasActiveFilters) return data

    return data.filter(item => {
      // Apply severity filter
      if (state.filters.severity && item.severity !== state.filters.severity) {
        return false
      }

      // Apply category filter
      if (state.filters.category && item.category !== state.filters.category) {
        return false
      }

      // Apply date range filter
      if (state.filters.dateRange && item.date) {
        const itemDate = new Date(item.date)
        const { start, end } = state.filters.dateRange
        if (itemDate < start || itemDate > end) {
          return false
        }
      }

      return true
    })
  }, [state.filters, hasActiveFilters])

  // Keyboard shortcuts
  useEffect(() => {
    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === 'Escape') {
        if (isMetricSelected) {
          clearSelection()
        } else if (hasActiveFilters) {
          clearFilters()
        }
      } else if (event.key === 'Backspace' && event.metaKey) {
        goBack()
      }
    }

    window.addEventListener('keydown', handleKeyDown)
    return () => window.removeEventListener('keydown', handleKeyDown)
  }, [isMetricSelected, hasActiveFilters, clearSelection, clearFilters, goBack])

  return {
    state,
    actions: {
      selectMetric,
      clearSelection,
      setFilter,
      clearFilters,
      setViewMode,
      setZoomLevel,
      drillDown,
      goBack
    },
    computed: {
      hasActiveFilters,
      isMetricSelected,
      canGoBack,
      filteredData
    }
  }
}

// Utility hook for managing chart interactions
export function useChartInteractions() {
  const [hoveredElement, setHoveredElement] = useState<string | null>(null)
  const [clickedElement, setClickedElement] = useState<string | null>(null)
  const [tooltip, setTooltip] = useState<{
    visible: boolean
    x: number
    y: number
    content: any
  }>({
    visible: false,
    x: 0,
    y: 0,
    content: null
  })

  const handleMouseEnter = useCallback((element: string, event?: React.MouseEvent) => {
    setHoveredElement(element)
    if (event) {
      setTooltip({
        visible: true,
        x: event.clientX,
        y: event.clientY,
        content: element
      })
    }
  }, [])

  const handleMouseLeave = useCallback(() => {
    setHoveredElement(null)
    setTooltip(prev => ({ ...prev, visible: false }))
  }, [])

  const handleClick = useCallback((element: string, data?: any) => {
    setClickedElement(element)
    return { element, data }
  }, [])

  const clearInteractions = useCallback(() => {
    setHoveredElement(null)
    setClickedElement(null)
    setTooltip(prev => ({ ...prev, visible: false }))
  }, [])

  return {
    hoveredElement,
    clickedElement,
    tooltip,
    handleMouseEnter,
    handleMouseLeave,
    handleClick,
    clearInteractions
  }
}

// Hook for managing analysis data fetching and caching
export function useAnalysisData(repoUrl?: string) {
  const [data, setData] = useState<any>(null)
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [cache, setCache] = useState<Map<string, any>>(new Map())

  const fetchAnalysis = useCallback(async (url: string, force = false) => {
    // Check cache first
    if (!force && cache.has(url)) {
      setData(cache.get(url))
      return
    }

    setIsLoading(true)
    setError(null)

    try {
      const response = await fetch('/api/analyze', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ repo_url: url })
      })

      if (!response.ok) {
        throw new Error(`Analysis failed: ${response.statusText}`)
      }

      const result = await response.json()
      
      // Cache the result
      setCache(prev => new Map(prev).set(url, result))
      setData(result)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unknown error')
    } finally {
      setIsLoading(false)
    }
  }, [cache])

  const refreshAnalysis = useCallback(() => {
    if (repoUrl) {
      fetchAnalysis(repoUrl, true)
    }
  }, [repoUrl, fetchAnalysis])

  const clearCache = useCallback(() => {
    setCache(new Map())
  }, [])

  // Auto-fetch when repoUrl changes
  useEffect(() => {
    if (repoUrl) {
      fetchAnalysis(repoUrl)
    }
  }, [repoUrl, fetchAnalysis])

  return {
    data,
    isLoading,
    error,
    fetchAnalysis,
    refreshAnalysis,
    clearCache,
    cacheSize: cache.size
  }
}

