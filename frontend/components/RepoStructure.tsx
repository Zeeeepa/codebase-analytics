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
  FaChevronDown
} from 'react-icons/fa';
import { MdError, MdWarning, MdInfo } from 'react-icons/md';

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

interface Symbol {
  id: string;
  name: string;
  type: 'function' | 'class' | 'variable';
  filepath: string;
  start_line: number;
  end_line: number;
  issues?: {
    type: 'critical' | 'major' | 'minor';
    message: string;
  }[];
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

interface RepoStructureProps {
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
        return <MdError className="text-red-500" />;
      case 'major':
        return <MdWarning className="text-yellow-500" />;
      case 'minor':
        return <MdInfo className="text-blue-500" />;
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
    <span className="ml-2 inline-flex items-center text-sm">
      {getIcon()}
      <span className="ml-1">
        {getLabel()}: {count}
      </span>
    </span>
  );
};

const Stats: React.FC<{ stats: Stats }> = ({ stats }) => {
  if (!stats) return null;

  return (
    <div className="text-xs text-gray-500 ml-2">
      {stats.files > 0 && <span className="mr-2">📄 {stats.files} files</span>}
      {stats.directories > 0 && <span className="mr-2">📁 {stats.directories} dirs</span>}
      {stats.symbols > 0 && <span className="mr-2">🔧 {stats.symbols} symbols</span>}
      {stats.issues > 0 && <span className="mr-2">⚠️ {stats.issues} issues</span>}
    </div>
  );
};

const SymbolList: React.FC<{
  symbols: Symbol[];
  onSymbolClick: (symbol: Symbol) => void;
  onViewCallChain: (symbolId: string) => void;
  onViewContext: (symbolId: string) => void;
}> = ({ symbols, onSymbolClick, onViewCallChain, onViewContext }) => {
  return (
    <div className="pl-8 border-l-2 border-gray-200">
      {symbols.map((symbol, index) => (
        <div key={`${symbol.id}-${index}`} className="py-1">
          <div className="flex items-center group">
            <FaCode className={`mr-2 ${symbol.type === 'function' ? 'text-purple-500' : 'text-blue-500'}`} />
            <span
              className="flex-1 cursor-pointer hover:text-blue-500"
              onClick={() => onSymbolClick(symbol)}
            >
              {symbol.name}
              <span className="text-xs text-gray-500 ml-2">
                L{symbol.start_line}-{symbol.end_line}
              </span>
            </span>
            <div className="hidden group-hover:flex space-x-2">
              {symbol.type === 'function' && (
                <button
                  className="px-2 py-1 text-xs bg-blue-500 text-white rounded hover:bg-blue-600"
                  onClick={() => onViewCallChain(symbol.id)}
                >
                  View Call Chain
                </button>
              )}
              <button
                className="px-2 py-1 text-xs bg-green-500 text-white rounded hover:bg-green-600"
                onClick={() => onViewContext(symbol.id)}
              >
                View Context
              </button>
            </div>
          </div>
          {symbol.issues && symbol.issues.length > 0 && (
            <div className="pl-6 mt-1">
              {symbol.issues.map((issue, i) => (
                <div
                  key={i}
                  className={`text-sm ${
                    issue.type === 'critical'
                      ? 'text-red-500'
                      : issue.type === 'major'
                      ? 'text-yellow-500'
                      : 'text-blue-500'
                  } cursor-pointer hover:underline`}
                  onClick={() => onViewContext(symbol.id)}
                >
                  <FaExclamationTriangle className="inline mr-1" />
                  {issue.message}
                </div>
              ))}
            </div>
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
  const [isOpen, setIsOpen] = useState(false);
  const [showSymbols, setShowSymbols] = useState(false);

  const handleClick = () => {
    if (node.type === 'directory') {
      setIsOpen(!isOpen);
      onFolderClick(node.path);
    } else {
      // For files, toggle symbol list but don't affect folder open state
      setShowSymbols(!showSymbols);
      onFileClick(node.path);
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

  const getChevron = () => {
    if (node.type === 'directory' || (node.type === 'file' && node.symbols && node.symbols.length > 0)) {
      return isOpen || showSymbols ? (
        <FaChevronDown className="text-gray-400 mr-1" />
      ) : (
        <FaChevronRight className="text-gray-400 mr-1" />
      );
    }
    return <span className="w-4 mr-1"></span>; // Empty space for alignment
  };

  return (
    <div className="select-none">
      <div
        className="flex items-center py-1 px-2 hover:bg-gray-100 cursor-pointer rounded"
        style={{ paddingLeft: `${level * 20}px` }}
        onClick={handleClick}
      >
        {getChevron()}
        <span className="mr-2">{getIcon()}</span>
        <span className="flex-1">{node.name}</span>
        <div className="flex items-center">
          {node.issues && (
            <div className="flex space-x-2">
              <IssueTag count={node.issues.critical} type="critical" />
              <IssueTag count={node.issues.major} type="major" />
              <IssueTag count={node.issues.minor} type="minor" />
            </div>
          )}
          {node.stats && <Stats stats={node.stats} />}
        </div>
      </div>
      {/* Show symbols for files when expanded */}
      {node.type === 'file' && showSymbols && node.symbols && (
        <SymbolList
          symbols={node.symbols}
          onSymbolClick={onSymbolClick}
          onViewCallChain={onViewCallChain}
          onViewContext={onViewContext}
        />
      )}
      {/* Show children for directories when expanded */}
      {node.type === 'directory' && isOpen && node.children &&
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
    </div>
  );
};

export const RepoStructure: React.FC<RepoStructureProps> = ({
  data,
  onFileClick,
  onFolderClick,
  onSymbolClick,
  onViewCallChain,
  onViewContext,
}) => {
  return (
    <div className="border rounded-lg shadow-sm bg-white">
      <div className="p-4 border-b bg-gray-50">
        <h2 className="text-lg font-semibold">📂 Repository Structure</h2>
        <p className="text-sm text-gray-500">Click on folders to expand, files to view symbols, and symbols to view details</p>
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
    </div>
  );
};

export default RepoStructure;
