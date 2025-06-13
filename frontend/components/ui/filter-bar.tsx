"use client"

import { IssueType, IssueSeverity, IssueCategory } from '@/lib/api-types'
import { useFilters } from '@/hooks/useSharedAnalysisState'
import { Input } from '@/components/ui/input'
import { Button } from '@/components/ui/button'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { Filter, X, Search } from 'lucide-react'

interface FilterBarProps {
  showTypeFilter?: boolean
  showSeverityFilter?: boolean
  showCategoryFilter?: boolean
  showSearchFilter?: boolean
  showResetButton?: boolean
  className?: string
}

export function FilterBar({
  showTypeFilter = true,
  showSeverityFilter = true,
  showCategoryFilter = true,
  showSearchFilter = true,
  showResetButton = true,
  className = ''
}: FilterBarProps) {
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
  } = useFilters()
  
  const hasActiveFilters = 
    severityFilter !== 'all' || 
    categoryFilter !== 'all' || 
    typeFilter !== 'all' || 
    searchQuery !== ''
  
  return (
    <div className={`flex flex-wrap gap-2 ${className}`}>
      {showSeverityFilter && (
        <Select 
          value={severityFilter} 
          onValueChange={(value) => setSeverityFilter(value as any)}
        >
          <SelectTrigger className="w-[180px]">
            <SelectValue placeholder="Filter by Severity" />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="all">All Severities</SelectItem>
            <SelectItem value={IssueSeverity.CRITICAL}>Critical</SelectItem>
            <SelectItem value={IssueSeverity.HIGH}>High</SelectItem>
            <SelectItem value={IssueSeverity.MEDIUM}>Medium</SelectItem>
            <SelectItem value={IssueSeverity.LOW}>Low</SelectItem>
            <SelectItem value={IssueSeverity.INFO}>Info</SelectItem>
          </SelectContent>
        </Select>
      )}
      
      {showCategoryFilter && (
        <Select 
          value={categoryFilter} 
          onValueChange={(value) => setCategoryFilter(value as any)}
        >
          <SelectTrigger className="w-[180px]">
            <SelectValue placeholder="Filter by Category" />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="all">All Categories</SelectItem>
            <SelectItem value={IssueCategory.FUNCTIONAL}>Functional</SelectItem>
            <SelectItem value={IssueCategory.STRUCTURAL}>Structural</SelectItem>
            <SelectItem value={IssueCategory.QUALITY}>Quality</SelectItem>
            <SelectItem value={IssueCategory.SECURITY}>Security</SelectItem>
            <SelectItem value={IssueCategory.PERFORMANCE}>Performance</SelectItem>
          </SelectContent>
        </Select>
      )}
      
      {showTypeFilter && (
        <Select 
          value={typeFilter} 
          onValueChange={(value) => setTypeFilter(value as any)}
        >
          <SelectTrigger className="w-[180px]">
            <SelectValue placeholder="Filter by Type" />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="all">All Types</SelectItem>
            <SelectItem value={IssueType.UNUSED_IMPORT}>Unused Import</SelectItem>
            <SelectItem value={IssueType.UNUSED_VARIABLE}>Unused Variable</SelectItem>
            <SelectItem value={IssueType.UNUSED_FUNCTION}>Unused Function</SelectItem>
            <SelectItem value={IssueType.UNUSED_PARAMETER}>Unused Parameter</SelectItem>
            <SelectItem value={IssueType.UNDEFINED_VARIABLE}>Undefined Variable</SelectItem>
            <SelectItem value={IssueType.UNDEFINED_FUNCTION}>Undefined Function</SelectItem>
            <SelectItem value={IssueType.PARAMETER_MISMATCH}>Parameter Mismatch</SelectItem>
            <SelectItem value={IssueType.TYPE_ERROR}>Type Error</SelectItem>
            <SelectItem value={IssueType.CIRCULAR_DEPENDENCY}>Circular Dependency</SelectItem>
            <SelectItem value={IssueType.DEAD_CODE}>Dead Code</SelectItem>
            <SelectItem value={IssueType.COMPLEXITY_ISSUE}>Complexity Issue</SelectItem>
            <SelectItem value={IssueType.STYLE_ISSUE}>Style Issue</SelectItem>
            <SelectItem value={IssueType.SECURITY_ISSUE}>Security Issue</SelectItem>
            <SelectItem value={IssueType.PERFORMANCE_ISSUE}>Performance Issue</SelectItem>
          </SelectContent>
        </Select>
      )}
      
      {showSearchFilter && (
        <div className="relative flex-1 min-w-[200px]">
          <Search className="absolute left-2 top-2.5 h-4 w-4 text-muted-foreground" />
          <Input
            placeholder="Search issues..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="pl-8"
          />
          {searchQuery && (
            <Button
              variant="ghost"
              size="icon"
              className="absolute right-0 top-0 h-full"
              onClick={() => setSearchQuery('')}
            >
              <X className="h-4 w-4" />
            </Button>
          )}
        </div>
      )}
      
      {showResetButton && hasActiveFilters && (
        <Button 
          variant="outline" 
          size="icon"
          onClick={resetFilters}
          title="Reset filters"
        >
          <X className="h-4 w-4" />
        </Button>
      )}
    </div>
  )
}

