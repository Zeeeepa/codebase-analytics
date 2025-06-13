"use client"

import { useFilters } from '@/hooks/useAnalysisState';
import { IssueType, IssueSeverity, IssueCategory } from '@/lib/api-types';
import { 
  Filter, 
  X, 
  AlertTriangle, 
  Code2, 
  Bug, 
  AlertCircle,
  Ban,
  Unlink,
  Repeat,
  Gauge,
  Shield,
  FileWarning,
  Sparkles,
  Search
} from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { 
  Select, 
  SelectContent, 
  SelectItem, 
  SelectTrigger, 
  SelectValue 
} from '@/components/ui/select';
import { cn } from '@/lib/utils';

interface FilterBarProps {
  showSeverityFilter?: boolean;
  showCategoryFilter?: boolean;
  showTypeFilter?: boolean;
  showSearchFilter?: boolean;
  showResetButton?: boolean;
  className?: string;
}

export function FilterBar({
  showSeverityFilter = true,
  showCategoryFilter = true,
  showTypeFilter = true,
  showSearchFilter = true,
  showResetButton = true,
  className
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
  } = useFilters();
  
  // Check if any filters are active
  const hasActiveFilters = 
    severityFilter !== 'all' || 
    categoryFilter !== 'all' || 
    typeFilter !== 'all' || 
    searchQuery !== '';
  
  return (
    <div className={cn("flex flex-wrap gap-2", className)}>
      {showSeverityFilter && (
        <Select
          value={severityFilter}
          onValueChange={(value) => setSeverityFilter(value as IssueSeverity | 'all')}
        >
          <SelectTrigger className="w-[140px]">
            <SelectValue placeholder="All Severities" />
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
          onValueChange={(value) => setCategoryFilter(value as IssueCategory | 'all')}
        >
          <SelectTrigger className="w-[140px]">
            <SelectValue placeholder="All Categories" />
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
          onValueChange={(value) => setTypeFilter(value as IssueType | 'all')}
        >
          <SelectTrigger className="w-[140px]">
            <SelectValue placeholder="All Types" />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="all">All Types</SelectItem>
            <SelectItem value={IssueType.UNUSED_IMPORT}>Unused Import</SelectItem>
            <SelectItem value={IssueType.UNUSED_VARIABLE}>Unused Variable</SelectItem>
            <SelectItem value={IssueType.UNDEFINED_VARIABLE}>Undefined Variable</SelectItem>
            <SelectItem value={IssueType.PARAMETER_MISMATCH}>Parameter Mismatch</SelectItem>
            <SelectItem value={IssueType.CIRCULAR_DEPENDENCY}>Circular Dependency</SelectItem>
            <SelectItem value={IssueType.PERFORMANCE_ISSUE}>Performance Issue</SelectItem>
            <SelectItem value={IssueType.SECURITY_ISSUE}>Security Issue</SelectItem>
            <SelectItem value={IssueType.SYNTAX_ERROR}>Syntax Error</SelectItem>
            <SelectItem value={IssueType.STYLE_ISSUE}>Style Issue</SelectItem>
          </SelectContent>
        </Select>
      )}
      
      {showSearchFilter && (
        <div className="relative flex-1 min-w-[200px]">
          <Search className="absolute left-2 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
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
              <span className="sr-only">Clear search</span>
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
          <span className="sr-only">Reset filters</span>
        </Button>
      )}
    </div>
  );
}

