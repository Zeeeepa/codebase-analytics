import React, { useState } from 'react';
import {
  FaFolder,
  FaFolderOpen,
  FaFile,
  FaCode,
  FaExclamationTriangle,
  FaPython,
  FaJs,
  FaJava,
  FaHtml5,
  FaCss3,
  FaDocker,
  FaYarn,
  FaMarkdown,
  FaDatabase,
  FaCog,
  FaFileCode,
  FaFileAlt,
  FaChevronRight,
  FaChevronDown,
  FaArrowRight,
  FaLink,
  FaExclamation,
  FaInfoCircle,
  FaExclamationCircle,
  FaCodeBranch,
  FaRegFileCode,
  FaRegFolder
} from 'react-icons/fa';
import { MdError, MdWarning, MdInfo } from 'react-icons/md';
import { Tooltip } from './ui/tooltip';
import { Badge } from './ui/badge';
import { Card } from './ui/card';

interface Stats {
  files: number;
  directories: number;
  symbols: number;
  issues: number;
}

interface IssueCount {
  critical: number;
  major: number;
  minor: number;
}

interface Issue {
  type: 'critical' | 'major' | 'minor';
  message: string;
  line?: number;
}

interface Symbol {
  id: string;
  name: string;
  type: 'function' | 'class' | 'variable';
  filepath: string;
  start_line: number;
  end_line: number;
  issues?: Issue[];
  parameters?: string[];
  return_type?: string;
  dependencies?: { name: string; type: string; filepath: string }[];
  call_chain?: { name: string; filepath: string }[];
}

interface FileNode {
  name: string;
  type: 'file' | 'directory';
  file_type?: string;
  path: string;
  issues: IssueCount;
  stats: Stats;
  symbols?: Symbol[];
  children?: { [key: string]: FileNode };
}

interface EnhancedRepoStructureProps {
  data: FileNode;
  onFileClick: (path: string) => void;
  onFolderClick: (path: string) => void;
  onSymbolClick: (symbol: Symbol) => void;
  onViewCallChain: (symbolId: string) => void;
  onViewContext: (symbolId: string) => void;
}

const getFileIcon = (fileType: string) => {
  switch (fileType) {
    case 'python':
      return <FaPython className="text-blue-500" />;
    case 'javascript':
      return <FaJs className="text-yellow-500" />;
    case 'java':
      return <FaJava className="text-red-500" />;
    case 'html':
      return <FaHtml5 className="text-orange-500" />;
    case 'css':
      return <FaCss3 className="text-blue-400" />;
    case 'docker':
      return <FaDocker className="text-blue-600" />;
    case 'yaml':
      return <FaYarn className="text-purple-500" />;
    case 'markdown':
      return <FaMarkdown className="text-gray-600" />;
    case 'sql':
      return <FaDatabase className="text-green-500" />;
    case 'config':
      return <FaCog className="text-gray-500" />;
    case 'text':
      return <FaFileAlt className="text-gray-400" />;
    default:
      return <FaFileCode className="text-gray-500" />;
  }
};

const IssueTag: React.FC<{ count: number; type: 'critical' | 'major' | 'minor' }> = ({ count, type }) => {
  if (count === 0) return null;

  const getIcon = () => {
    switch (type) {
      case 'critical':
        return <FaExclamationCircle className="text-red-500" />;
      case 'major':
        return <FaExclamation className="text-yellow-500" />;
      case 'minor':
        return <FaInfoCircle className="text-blue-500" />;
    }
  };

  const getLabel = () => {
    switch (type) {
      case 'critical':
        return '⚠️ Critical';
      case 'major':
        return '👉 Major';
      case 'minor':
        return '🔍 Minor';
    }
  };

  return (
    <Badge variant={type === 'critical' ? 'destructive' : type === 'major' ? 'warning' : 'info'} className="ml-2">
      {getIcon()}
      <span className="ml-1">
        {count}
      </span>
    </Badge>
  );
};

const Stats: React.FC<{ stats: Stats }> = ({ stats }) => {
  if (!stats) return null;

  return (
    <div className="text-xs text-gray-500 ml-2 flex flex-wrap">
      {stats.files > 0 && (
        <Tooltip content={`${stats.files} files in this directory`}>
          <span className="mr-2 flex items-center">
            <FaRegFileCode className="mr-1" /> {stats.files}
          </span>
        </Tooltip>
      )}
      {stats.directories > 0 && (
        <Tooltip content={`${stats.directories} subdirectories`}>
          <span className="mr-2 flex items-center">
            <FaRegFolder className="mr-1" /> {stats.directories}
          </span>
        </Tooltip>
      )}
      {stats.symbols > 0 && (
        <Tooltip content={`${stats.symbols} code symbols (functions, classes, etc.)`}>
          <span className="mr-2 flex items-center">
            <FaCode className="mr-1" /> {stats.symbols}
          </span>
        </Tooltip>
      )}
      {stats.issues > 0 && (
        <Tooltip content={`${stats.issues} issues detected`}>
          <span className="mr-2 flex items-center">
            <FaExclamationTriangle className="mr-1 text-yellow-500" /> {stats.issues}
          </span>
        </Tooltip>
      )}
    </div>
  );
};

const SymbolDetails: React.FC<{ symbol: Symbol }> = ({ symbol }) => {
  return (
    <div className="p-3 bg-gray-50 rounded-md mt-1 mb-2 text-sm">
      <div className="font-medium">{symbol.type === 'function' ? 'Function' : symbol.type === 'class' ? 'Class' : 'Variable'}: {symbol.name}</div>
      
      {symbol.parameters && symbol.parameters.length > 0 && (
        <div className="mt-1">
          <span className="text-gray-600">Parameters:</span>
          <div className="pl-2 font-mono text-xs">
            {symbol.parameters.map((param, i) => (
              <div key={i}>{param}</div>
            ))}
          </div>
        </div>
      )}
      
      {symbol.return_type && (
        <div className="mt-1">
          <span className="text-gray-600">Returns:</span>
          <span className="pl-2 font-mono text-xs">{symbol.return_type}</span>
        </div>
      )}
      
      {symbol.dependencies && symbol.dependencies.length > 0 && (
        <div className="mt-1">
          <span className="text-gray-600">Dependencies:</span>
          <div className="pl-2 text-xs">
            {symbol.dependencies.slice(0, 3).map((dep, i) => (
              <div key={i} className="flex items-center">
                <FaLink className="mr-1 text-gray-400" />
                <span>{dep.name}</span>
                <span className="text-gray-500 ml-1">({dep.type})</span>
              </div>
            ))}
            {symbol.dependencies.length > 3 && (
              <div className="text-gray-500">+ {symbol.dependencies.length - 3} more...</div>
            )}
          </div>
        </div>
      )}
      
      {symbol.issues && symbol.issues.length > 0 && (
        <div className="mt-1">
          <span className="text-gray-600">Issues:</span>
          <div className="pl-2 text-xs">
            {symbol.issues.map((issue, i) => (
              <div 
                key={i} 
                className={`flex items-center ${
                  issue.type === 'critical' 
                    ? 'text-red-500' 
                    : issue.type === 'major' 
                    ? 'text-yellow-500' 
                    : 'text-blue-500'
                }`}
              >
                <FaExclamationTriangle className="mr-1" />
                <span>{issue.message}</span>
                {issue.line && <span className="ml-1 text-gray-500">(line {issue.line})</span>}
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};

const SymbolList: React.FC<{
  symbols: Symbol[];
  onSymbolClick: (symbol: Symbol) => void;
  onViewCallChain: (symbolId: string) => void;
  onViewContext: (symbolId: string) => void;
}> = ({ symbols, onSymbolClick, onViewCallChain, onViewContext }) => {
  const [expandedSymbol, setExpandedSymbol] = useState<string | null>(null);

  const toggleSymbol = (symbolId: string) => {
    setExpandedSymbol(expandedSymbol === symbolId ? null : symbolId);
  };

  return (
    <div className="pl-8 border-l-2 border-gray-200">
      {symbols.map((symbol, index) => (
        <div key={`${symbol.id}-${index}`} className="py-1">
          <div className="flex items-center group">
            <button 
              className="mr-1 text-gray-400 hover:text-gray-600 focus:outline-none"
              onClick={() => toggleSymbol(symbol.id)}
            >
              {expandedSymbol === symbol.id ? <FaChevronDown size={12} /> : <FaChevronRight size={12} />}
            </button>
            <FaCode className={`mr-2 ${symbol.type === 'function' ? 'text-purple-500' : symbol.type === 'class' ? 'text-blue-500' : 'text-green-500'}`} />
            <span
              className="flex-1 cursor-pointer hover:text-blue-500 font-medium"
              onClick={() => toggleSymbol(symbol.id)}
            >
              {symbol.name}
              <span className="text-xs text-gray-500 ml-2 font-normal">
                L{symbol.start_line}-{symbol.end_line}
              </span>
            </span>
            <div className="hidden group-hover:flex space-x-2">
              {symbol.type === 'function' && (
                <button
                  className="px-2 py-1 text-xs bg-blue-500 text-white rounded hover:bg-blue-600"
                  onClick={() => onViewCallChain(symbol.id)}
                >
                  Call Chain
                </button>
              )}
              <button
                className="px-2 py-1 text-xs bg-green-500 text-white rounded hover:bg-green-600"
                onClick={() => onViewContext(symbol.id)}
              >
                Context
              </button>
            </div>
            {symbol.issues && symbol.issues.length > 0 && (
              <div className="flex space-x-1 ml-2">
                <IssueTag 
                  count={symbol.issues.filter(i => i.type === 'critical').length} 
                  type="critical" 
                />
                <IssueTag 
                  count={symbol.issues.filter(i => i.type === 'major').length} 
                  type="major" 
                />
                <IssueTag 
                  count={symbol.issues.filter(i => i.type === 'minor').length} 
                  type="minor" 
                />
              </div>
            )}
          </div>
          
          {expandedSymbol === symbol.id && (
            <SymbolDetails symbol={symbol} />
          )}
        </div>
      ))}
    </div>
  );
};

const FileTreeNode: React.FC<{
  node: FileNode;
  level: number;
  onFileClick: (path: string) => void;
  onFolderClick: (path: string) => void;
  onSymbolClick: (symbol: Symbol) => void;
  onViewCallChain: (symbolId: string) => void;
  onViewContext: (symbolId: string) => void;
}> = ({
  node,
  level,
  onFileClick,
  onFolderClick,
  onSymbolClick,
  onViewCallChain,
  onViewContext,
}) => {
  const [isOpen, setIsOpen] = useState(level === 0);

  const handleClick = () => {
    if (node.type === 'directory') {
      setIsOpen(!isOpen);
      onFolderClick(node.path);
    } else {
      onFileClick(node.path);
      setIsOpen(!isOpen); // Toggle symbol list for files
    }
  };

  const getIcon = () => {
    if (node.type === 'directory') {
      return isOpen ? (
        <FaFolderOpen className="text-yellow-500" />
      ) : (
        <FaFolder className="text-yellow-500" />
      );
    }
    return node.file_type ? getFileIcon(node.file_type) : <FaFile className="text-gray-500" />;
  };

  // Format path for display
  const displayPath = node.path ? node.path.split('/').pop() || node.name : node.name;
  
  // Calculate total issues
  const totalIssues = node.issues ? node.issues.critical + node.issues.major + node.issues.minor : 0;

  return (
    <div className="select-none">
      <div
        className={`flex items-center py-1.5 px-2 hover:bg-gray-100 cursor-pointer rounded ${isOpen ? 'bg-gray-50' : ''}`}
        style={{ paddingLeft: `${level * 16 + 8}px` }}
        onClick={handleClick}
      >
        <span className="mr-2 text-lg">{getIcon()}</span>
        <span className="flex-1 font-medium">
          {displayPath}
          {totalIssues > 0 && (
            <span className="ml-2 text-xs text-gray-500">
              [{totalIssues} {totalIssues === 1 ? 'issue' : 'issues'}]
            </span>
          )}
        </span>
        <div className="flex items-center">
          {node.issues && (
            <div className="flex space-x-1">
              <IssueTag count={node.issues.critical} type="critical" />
              <IssueTag count={node.issues.major} type="major" />
              <IssueTag count={node.issues.minor} type="minor" />
            </div>
          )}
          {node.stats && <Stats stats={node.stats} />}
        </div>
      </div>
      {isOpen && (
        <>
          {node.symbols && (
            <SymbolList
              symbols={node.symbols}
              onSymbolClick={onSymbolClick}
              onViewCallChain={onViewCallChain}
              onViewContext={onViewContext}
            />
          )}
          {node.children &&
            Object.entries(node.children)
              .sort(([a], [b]) => {
                // Sort directories first, then files
                const nodeA = node.children![a];
                const nodeB = node.children![b];
                if (nodeA.type === nodeB.type) {
                  return a.localeCompare(b);
                }
                return nodeA.type === 'directory' ? -1 : 1;
              })
              .map(([key, child]) => (
                <FileTreeNode
                  key={`${child.path}-${key}`}
                  node={child}
                  level={level + 1}
                  onFileClick={onFileClick}
                  onFolderClick={onFolderClick}
                  onSymbolClick={onSymbolClick}
                  onViewCallChain={onViewCallChain}
                  onViewContext={onViewContext}
                />
              ))}
        </>
      )}
    </div>
  );
};

const EnhancedRepoStructure: React.FC<EnhancedRepoStructureProps> = ({
  data,
  onFileClick,
  onFolderClick,
  onSymbolClick,
  onViewCallChain,
  onViewContext,
}) => {
  return (
    <Card className="border rounded-lg shadow-sm bg-white overflow-hidden">
      <div className="p-4 border-b bg-gray-50 flex items-center justify-between">
        <div>
          <h2 className="text-lg font-semibold flex items-center">
            <FaCodeBranch className="mr-2 text-blue-500" /> 
            Interactive Repository Structure
          </h2>
          <p className="text-sm text-gray-500">
            Click on folders to expand, files to view symbols, and symbols to see details
          </p>
        </div>
        <Stats stats={data.stats} />
      </div>
      <div className="p-2 max-h-[600px] overflow-y-auto">
        <FileTreeNode
          node={data}
          level={0}
          onFileClick={onFileClick}
          onFolderClick={onFolderClick}
          onSymbolClick={onSymbolClick}
          onViewCallChain={onViewCallChain}
          onViewContext={onViewContext}
        />
      </div>
    </Card>
  );
};

export default EnhancedRepoStructure;

