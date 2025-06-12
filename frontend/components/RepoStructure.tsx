import React, { useState } from 'react';
import { FaFolder, FaFolderOpen, FaFile, FaCode, FaExclamationTriangle } from 'react-icons/fa';
import { MdError, MdWarning, MdInfo } from 'react-icons/md';

interface IssueCount {
  critical: number;
  major: number;
  minor: number;
}

interface Symbol {
  id: string;
  name: string;
  type: 'function' | 'class' | 'variable';
  issues?: {
    type: 'critical' | 'major' | 'minor';
    message: string;
  }[];
}

interface FileNode {
  name: string;
  type: 'file' | 'directory';
  path: string;
  issues?: IssueCount;
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
            <FaCode className="mr-2 text-gray-500" />
            <span
              className="flex-1 cursor-pointer hover:text-blue-500"
              onClick={() => onSymbolClick(symbol)}
            >
              {symbol.name}
            </span>
            <div className="hidden group-hover:flex space-x-2">
              <button
                className="px-2 py-1 text-xs bg-blue-500 text-white rounded hover:bg-blue-600"
                onClick={() => onViewCallChain(symbol.id)}
              >
                View Call Chain
              </button>
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
                  }`}
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
    return <FaFile className="text-gray-500" />;
  };

  return (
    <div className="select-none">
      <div
        className="flex items-center py-1 px-2 hover:bg-gray-100 cursor-pointer"
        style={{ paddingLeft: `${level * 20}px` }}
        onClick={handleClick}
      >
        <span className="mr-2">{getIcon()}</span>
        <span className="flex-1">{node.name}</span>
        {node.issues && (
          <div className="flex space-x-2">
            <IssueTag count={node.issues.critical} type="critical" />
            <IssueTag count={node.issues.major} type="major" />
            <IssueTag count={node.issues.minor} type="minor" />
          </div>
        )}
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
            Object.entries(node.children).map(([key, child]) => (
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
      </div>
      <div className="p-2">
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

