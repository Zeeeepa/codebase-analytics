"use client"

import { useViewOptions } from '@/hooks/useAnalysisState';
import { 
  SortDesc, 
  List, 
  LayoutGrid, 
  Network, 
  Table2, 
  AlertTriangle, 
  BarChart3, 
  FileCode, 
  Code2 
} from 'lucide-react';
import { Button } from '@/components/ui/button';
import { 
  Select, 
  SelectContent, 
  SelectItem, 
  SelectTrigger, 
  SelectValue 
} from '@/components/ui/select';
import { cn } from '@/lib/utils';

interface SortBarProps {
  showSortOptions?: boolean;
  showViewModeOptions?: boolean;
  className?: string;
}

export function SortBar({
  showSortOptions = true,
  showViewModeOptions = true,
  className
}: SortBarProps) {
  const { sortBy, viewMode, setSortBy, setViewMode } = useViewOptions();
  
  return (
    <div className={cn("flex items-center gap-2", className)}>
      {showSortOptions && (
        <Select
          value={sortBy}
          onValueChange={(value) => setSortBy(value as 'impact' | 'severity' | 'type' | 'name')}
        >
          <SelectTrigger className="w-[140px]">
            <SelectValue placeholder="Sort by" />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="impact">
              <div className="flex items-center gap-2">
                <BarChart3 className="h-4 w-4" />
                <span>Sort by Impact</span>
              </div>
            </SelectItem>
            <SelectItem value="severity">
              <div className="flex items-center gap-2">
                <AlertTriangle className="h-4 w-4" />
                <span>Sort by Severity</span>
              </div>
            </SelectItem>
            <SelectItem value="type">
              <div className="flex items-center gap-2">
                <FileCode className="h-4 w-4" />
                <span>Sort by Type</span>
              </div>
            </SelectItem>
            <SelectItem value="name">
              <div className="flex items-center gap-2">
                <Code2 className="h-4 w-4" />
                <span>Sort by Name</span>
              </div>
            </SelectItem>
          </SelectContent>
        </Select>
      )}
      
      {showViewModeOptions && (
        <div className="flex items-center gap-1">
          <Button
            variant={viewMode === 'list' ? 'default' : 'ghost'}
            size="icon"
            onClick={() => setViewMode('list')}
            title="List view"
          >
            <List className="h-4 w-4" />
            <span className="sr-only">List view</span>
          </Button>
          
          <Button
            variant={viewMode === 'tree' ? 'default' : 'ghost'}
            size="icon"
            onClick={() => setViewMode('tree')}
            title="Tree view"
          >
            <LayoutGrid className="h-4 w-4" />
            <span className="sr-only">Tree view</span>
          </Button>
          
          <Button
            variant={viewMode === 'graph' ? 'default' : 'ghost'}
            size="icon"
            onClick={() => setViewMode('graph')}
            title="Graph view"
          >
            <Network className="h-4 w-4" />
            <span className="sr-only">Graph view</span>
          </Button>
          
          <Button
            variant={viewMode === 'table' ? 'default' : 'ghost'}
            size="icon"
            onClick={() => setViewMode('table')}
            title="Table view"
          >
            <Table2 className="h-4 w-4" />
            <span className="sr-only">Table view</span>
          </Button>
        </div>
      )}
    </div>
  );
}

