import React, { useState } from 'react';
import { FaFolder, FaFolderOpen, FaFile } from 'react-icons/fa';
import { MdError, MdWarning, MdInfo } from 'react-icons/md';

interface IssueCount {
  critical: number;
  major: number;
  minor: number;
}

interface FileNode {
  name: string;
  type: 'file' | 'directory';
  path: string;
  issues?: IssueCount;
  children?: FileNode[];
}

interface RepoStructureProps {
  data: FileNode;
  onFileClick: (path: string) => void;
  onFolderClick: (path: string) => void;
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

const FileTreeNode: React.FC<{
  node: FileNode;
  level: number;
  onFileClick: (path: string) => void;
  onFolderClick: (path: string) => void;
}> = ({ node, level, onFileClick, onFolderClick }) => {
  const [isOpen, setIsOpen] = useState(false);

  const handleClick = () => {
    if (node.type === 'directory') {
      setIsOpen(!isOpen);
      onFolderClick(node.path);
    } else {
      onFileClick(node.path);
    }
  };

  const getIcon = () => {
    if (node.type === 'directory') {
      return isOpen ? <FaFolderOpen className="text-yellow-500" /> : <FaFolder className="text-yellow-500" />;
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
      {isOpen && node.children && (
        <div>
          {node.children.map((child, index) => (
            <FileTreeNode
              key={`${child.path}-${index}`}
              node={child}
              level={level + 1}
              onFileClick={onFileClick}
              onFolderClick={onFolderClick}
            />
          ))}
        </div>
      )}
    </div>
  );
};

export const RepoStructure: React.FC<RepoStructureProps> = ({ data, onFileClick, onFolderClick }) => {
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
        />
      </div>
    </div>
  );
};

export default RepoStructure;

