"use client"

import { useViewOptions } from '@/hooks/useSharedAnalysisState'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { Button } from '@/components/ui/button'
import { SortDesc, LayoutGrid, LayoutList, Network, Table } from 'lucide-react'

interface SortBarProps {
  showSortOptions?: boolean
  showViewModeOptions?: boolean
  className?: string
}

export function SortBar({
  showSortOptions = true,
  showViewModeOptions = true,
  className = ''
}: SortBarProps) {
  const { sortBy, viewMode, setSortBy, setViewMode } = useViewOptions()
  
  return (
    <div className={`flex flex-wrap items-center gap-2 ${className}`}>
      {showSortOptions && (
        <Select value={sortBy} onValueChange={(value) => setSortBy(value as any)}>
          <SelectTrigger className="w-[180px]">
            <SortDesc className="h-4 w-4 mr-2" />
            <SelectValue placeholder="Sort by" />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="impact">Sort by Impact</SelectItem>
            <SelectItem value="severity">Sort by Severity</SelectItem>
            <SelectItem value="type">Sort by Type</SelectItem>
            <SelectItem value="name">Sort by Name</SelectItem>
          </SelectContent>
        </Select>
      )}
      
      {showViewModeOptions && (
        <div className="flex items-center border rounded-md">
          <Button
            variant={viewMode === 'list' ? 'default' : 'ghost'}
            size="sm"
            className="rounded-r-none"
            onClick={() => setViewMode('list')}
            title="List view"
          >
            <LayoutList className="h-4 w-4" />
          </Button>
          <Button
            variant={viewMode === 'tree' ? 'default' : 'ghost'}
            size="sm"
            className="rounded-none border-l border-r"
            onClick={() => setViewMode('tree')}
            title="Tree view"
          >
            <LayoutGrid className="h-4 w-4" />
          </Button>
          <Button
            variant={viewMode === 'graph' ? 'default' : 'ghost'}
            size="sm"
            className="rounded-none border-r"
            onClick={() => setViewMode('graph')}
            title="Graph view"
          >
            <Network className="h-4 w-4" />
          </Button>
          <Button
            variant={viewMode === 'table' ? 'default' : 'ghost'}
            size="sm"
            className="rounded-l-none"
            onClick={() => setViewMode('table')}
            title="Table view"
          >
            <Table className="h-4 w-4" />
          </Button>
        </div>
      )}
    </div>
  )
}

