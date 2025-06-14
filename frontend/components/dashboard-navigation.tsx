"use client"

import { useDashboard } from './dashboard-context';
import { 
  BarChart3, 
  Code2, 
  AlertTriangle, 
  Target, 
  GitBranch, 
  FolderTree, 
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
      icon: <FolderTree className="h-5 w-5" />,
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
        <h2 className="text-base font-semibold mb-4 px-2 text-muted-foreground uppercase tracking-wider">
          Analytics
        </h2>
        <div className="space-y-1">
          {navItems.map((item) => (
            <Button
              key={item.id}
              variant={activeView === item.id ? "secondary" : "ghost"}
              className={`w-full justify-start py-2 px-3 text-sm font-medium transition-all ${
                activeView === item.id 
                  ? "bg-secondary text-secondary-foreground" 
                  : "text-muted-foreground hover:text-foreground"
              } ${item.disabled ? "opacity-50 cursor-not-allowed" : ""}`}
              disabled={item.disabled}
              onClick={() => setActiveView(item.id)}
            >
              <span className={`mr-3 ${activeView === item.id ? "text-primary" : ""}`}>
                {item.icon}
              </span>
              <span>{item.label}</span>
            </Button>
          ))}
        </div>
      </div>
      
      <Separator className="my-4 opacity-50" />
      
      <div className="p-4 mt-auto">
        <Button 
          variant="ghost" 
          className="w-full justify-start py-2 px-3 text-sm font-medium text-muted-foreground hover:text-foreground transition-all"
        >
          <span className="mr-3">
            <Settings className="h-5 w-5" />
          </span>
          <span>Settings</span>
        </Button>
      </div>
    </div>
  );
}
