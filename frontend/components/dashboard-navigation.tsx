"use client"

import { useDashboard } from './dashboard-context';
import { 
  BarChart3, 
  Code2, 
  AlertTriangle, 
  Target, 
  GitBranch, 
  FileTree, 
  Search,
  Settings
} from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Separator } from '@/components/ui/separator';

export function DashboardNavigation() {
  const { activeView, setActiveView, repoData, explorationData, blastRadiusData } = useDashboard();

  const navItems = [
    {
      id: 'metrics',
      label: 'Metrics',
      icon: <BarChart3 className="h-5 w-5" />,
      disabled: !repoData,
    },
    {
      id: 'explorer',
      label: 'Code Explorer',
      icon: <Code2 className="h-5 w-5" />,
      disabled: !explorationData,
    },
    {
      id: 'issues',
      label: 'Issues',
      icon: <AlertTriangle className="h-5 w-5" />,
      disabled: !explorationData,
    },
    {
      id: 'blast-radius',
      label: 'Blast Radius',
      icon: <Target className="h-5 w-5" />,
      disabled: !blastRadiusData,
    },
    {
      id: 'structure',
      label: 'Structure',
      icon: <FileTree className="h-5 w-5" />,
      disabled: !repoData,
    },
    {
      id: 'dependencies',
      label: 'Dependencies',
      icon: <GitBranch className="h-5 w-5" />,
      disabled: !repoData,
    },
  ];

  return (
    <div className="flex flex-col h-full">
      <div className="p-4">
        <h2 className="text-lg font-semibold mb-2">Analytics</h2>
        <div className="space-y-1">
          {navItems.map((item) => (
            <Button
              key={item.id}
              variant={activeView === item.id ? "default" : "ghost"}
              className="w-full justify-start"
              disabled={item.disabled}
              onClick={() => setActiveView(item.id)}
            >
              {item.icon}
              <span className="ml-2">{item.label}</span>
            </Button>
          ))}
        </div>
      </div>
      
      <Separator />
      
      <div className="p-4 mt-auto">
        <Button variant="ghost" className="w-full justify-start">
          <Settings className="h-5 w-5" />
          <span className="ml-2">Settings</span>
        </Button>
      </div>
    </div>
  );
}

