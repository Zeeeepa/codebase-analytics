import React from 'react';
import {
  FaCode,
  FaCube,
  FaArrowRight,
  FaExclamationTriangle,
  FaFileImport,
  FaLink,
  FaChartLine,
  FaCodeBranch,
  FaLayerGroup,
  FaExclamationCircle,
  FaExclamation,
  FaInfoCircle
} from 'react-icons/fa';

interface Dependency {
  name: string;
  type: string;
  filepath: string | null;
}

interface Usage {
  filepath: string;
  line: number;
}

interface Issue {
  type: 'critical' | 'major' | 'minor';
  message: string;
  line?: number;
}

interface SymbolContextProps {
  id: string;
  name: string;
  type: string;
  filepath: string;
  start_line: number;
  end_line: number;
  source?: string;
  dependencies?: Dependency[];
  usages?: Usage[];
  issues?: Issue[];
  metrics?: {
    cyclomatic_complexity?: number;
    halstead_volume?: number;
    maintainability_index?: number;
    rank?: string;
  };
  function_info?: {
    parameters?: Array<{
      name: string;
      type?: string;
      default_value?: string;
    }>;
    return_type?: string;
    call_chain?: string[];
  };
  class_info?: {
    superclasses?: string[];
    methods?: string[];
    attributes?: string[];
    inheritance_depth?: number;
  };
}

const IssueDisplay: React.FC<{ issues: Issue[] }> = ({ issues }) => {
  if (!issues || issues.length === 0) return null;
  
  const getIssueIcon = (type: string) => {
    switch (type) {
      case 'critical':
        return <FaExclamationCircle className="text-red-500 mr-2" />;
      case 'major':
        return <FaExclamation className="text-yellow-500 mr-2" />;
      case 'minor':
        return <FaInfoCircle className="text-blue-500 mr-2" />;
      default:
        return <FaExclamationTriangle className="text-gray-500 mr-2" />;
    }
  };
  
  return (
    <div className="mt-4">
      <h3 className="text-lg font-semibold flex items-center">
        <FaExclamationTriangle className="mr-2" />
        Issues ({issues.length})
      </h3>
      <div className="mt-2 p-3 bg-gray-50 rounded">
        {issues.map((issue, index) => (
          <div 
            key={index} 
            className={`mb-2 p-2 rounded ${
              issue.type === 'critical' ? 'bg-red-50' : 
              issue.type === 'major' ? 'bg-yellow-50' : 
              'bg-blue-50'
            }`}
          >
            <div className="flex items-start">
              {getIssueIcon(issue.type)}
              <div>
                <p className="font-medium">{issue.message}</p>
                {issue.line && <p className="text-sm text-gray-600">Line: {issue.line}</p>}
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};

const DependenciesDisplay: React.FC<{ dependencies: Dependency[] }> = ({ dependencies }) => {
  if (!dependencies || dependencies.length === 0) return null;
  
  return (
    <div className="mt-4">
      <h3 className="text-lg font-semibold flex items-center">
        <FaFileImport className="mr-2" />
        Dependencies ({dependencies.length})
      </h3>
      <div className="mt-2 p-3 bg-gray-50 rounded">
        <div className="grid grid-cols-1 md:grid-cols-2 gap-2">
          {dependencies.map((dep, index) => (
            <div key={index} className="p-2 bg-white rounded shadow-sm">
              <p className="font-medium">{dep.name}</p>
              <p className="text-sm text-gray-600">Type: {dep.type}</p>
              {dep.filepath && <p className="text-sm text-gray-600">Path: {dep.filepath}</p>}
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};

const UsagesDisplay: React.FC<{ usages: Usage[] }> = ({ usages }) => {
  if (!usages || usages.length === 0) return null;
  
  return (
    <div className="mt-4">
      <h3 className="text-lg font-semibold flex items-center">
        <FaLink className="mr-2" />
        Usages ({usages.length})
      </h3>
      <div className="mt-2 p-3 bg-gray-50 rounded max-h-60 overflow-y-auto">
        <table className="w-full">
          <thead>
            <tr className="text-left">
              <th className="pb-2">File</th>
              <th className="pb-2">Line</th>
            </tr>
          </thead>
          <tbody>
            {usages.map((usage, index) => (
              <tr key={index} className="border-t border-gray-200">
                <td className="py-2 pr-4">{usage.filepath}</td>
                <td className="py-2">{usage.line}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
};

const MetricsDisplay: React.FC<{ metrics: SymbolContextProps['metrics'] }> = ({ metrics }) => {
  if (!metrics) return null;
  
  const getRankColor = (rank?: string) => {
    if (!rank) return 'text-gray-500';
    
    switch (rank.charAt(0)) {
      case 'A': return 'text-green-500';
      case 'B': return 'text-blue-500';
      case 'C': return 'text-yellow-500';
      default: return 'text-red-500';
    }
  };
  
  return (
    <div className="mt-4">
      <h3 className="text-lg font-semibold flex items-center">
        <FaChartLine className="mr-2" />
        Code Metrics
      </h3>
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mt-2">
        {metrics.cyclomatic_complexity !== undefined && (
          <div className="p-3 bg-gray-50 rounded">
            <h4 className="font-medium">Cyclomatic Complexity</h4>
            <p className="text-2xl">{metrics.cyclomatic_complexity}</p>
          </div>
        )}
        {metrics.halstead_volume !== undefined && (
          <div className="p-3 bg-gray-50 rounded">
            <h4 className="font-medium">Halstead Volume</h4>
            <p className="text-2xl">{metrics.halstead_volume.toFixed(2)}</p>
          </div>
        )}
        {metrics.maintainability_index !== undefined && (
          <div className="p-3 bg-gray-50 rounded">
            <h4 className="font-medium">Maintainability Index</h4>
            <p className="text-2xl">
              {metrics.maintainability_index.toFixed(1)}
              {metrics.rank && (
                <span className={`ml-2 text-sm ${getRankColor(metrics.rank)}`}>
                  (Rank {metrics.rank})
                </span>
              )}
            </p>
          </div>
        )}
      </div>
    </div>
  );
};

const FunctionInfoDisplay: React.FC<{ info: SymbolContextProps['function_info'] }> = ({ info }) => {
  if (!info) return null;
  
  return (
    <div className="mt-4">
      <h3 className="text-lg font-semibold flex items-center">
        <FaCode className="mr-2" />
        Function Details
      </h3>
      
      {info.parameters && info.parameters.length > 0 && (
        <div className="mt-2 p-3 bg-gray-50 rounded">
          <h4 className="font-medium">Parameters ({info.parameters.length})</h4>
          <div className="mt-1">
            {info.parameters.map((param, index) => (
              <div key={index} className="mb-1">
                <code className="bg-gray-100 px-1 py-0.5 rounded">
                  {param.name}
                  {param.type && <span className="text-gray-500">: {param.type}</span>}
                  {param.default_value && <span className="text-gray-500"> = {param.default_value}</span>}
                </code>
              </div>
            ))}
          </div>
        </div>
      )}
      
      {info.return_type && (
        <div className="mt-2 p-3 bg-gray-50 rounded">
          <h4 className="font-medium">Return Type</h4>
          <code className="bg-gray-100 px-1 py-0.5 rounded">{info.return_type}</code>
        </div>
      )}
      
      {info.call_chain && info.call_chain.length > 0 && (
        <div className="mt-2 p-3 bg-gray-50 rounded">
          <h4 className="font-medium">Call Chain</h4>
          <div className="flex flex-wrap items-center mt-1">
            {info.call_chain.map((func, index) => (
              <React.Fragment key={index}>
                {index > 0 && <FaArrowRight className="mx-2 text-gray-400" />}
                <code className="bg-gray-100 px-1 py-0.5 rounded">{func}</code>
              </React.Fragment>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};

const ClassInfoDisplay: React.FC<{ info: SymbolContextProps['class_info'] }> = ({ info }) => {
  if (!info) return null;
  
  return (
    <div className="mt-4">
      <h3 className="text-lg font-semibold flex items-center">
        <FaCube className="mr-2" />
        Class Details
      </h3>
      
      {info.superclasses && info.superclasses.length > 0 && (
        <div className="mt-2 p-3 bg-gray-50 rounded">
          <h4 className="font-medium">Inheritance</h4>
          <div className="mt-1">
            <p>Depth: {info.inheritance_depth || info.superclasses.length}</p>
            <div className="flex flex-wrap items-center mt-1">
              {info.superclasses.map((cls, index) => (
                <React.Fragment key={index}>
                  {index > 0 && <FaArrowRight className="mx-2 text-gray-400" />}
                  <code className="bg-gray-100 px-1 py-0.5 rounded">{cls}</code>
                </React.Fragment>
              ))}
            </div>
          </div>
        </div>
      )}
      
      {info.methods && info.methods.length > 0 && (
        <div className="mt-2 p-3 bg-gray-50 rounded">
          <h4 className="font-medium">Methods ({info.methods.length})</h4>
          <div className="mt-1 grid grid-cols-2 gap-2">
            {info.methods.map((method, index) => (
              <code key={index} className="bg-gray-100 px-1 py-0.5 rounded">{method}</code>
            ))}
          </div>
        </div>
      )}
      
      {info.attributes && info.attributes.length > 0 && (
        <div className="mt-2 p-3 bg-gray-50 rounded">
          <h4 className="font-medium">Attributes ({info.attributes.length})</h4>
          <div className="mt-1 grid grid-cols-2 gap-2">
            {info.attributes.map((attr, index) => (
              <code key={index} className="bg-gray-100 px-1 py-0.5 rounded">{attr}</code>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};

export const SymbolContext: React.FC<SymbolContextProps> = ({
  id,
  name,
  type,
  filepath,
  start_line,
  end_line,
  source,
  dependencies,
  usages,
  issues,
  metrics,
  function_info,
  class_info,
}) => {
  return (
    <div className="p-4">
      <div className="flex items-start justify-between">
        <div>
          <h2 className="text-2xl font-bold flex items-center">
            {type === 'function' ? <FaCode className="mr-2" /> : 
             type === 'class' ? <FaCube className="mr-2" /> : 
             <FaLayerGroup className="mr-2" />}
            {name}
          </h2>
          <p className="text-gray-600">
            {type} @ {filepath}:L{start_line}-{end_line}
          </p>
        </div>
      </div>

      {source && (
        <div className="mt-4">
          <h3 className="text-lg font-semibold flex items-center">
            <FaCodeBranch className="mr-2" />
            Source Code
          </h3>
          <pre className="mt-2 p-4 bg-gray-50 rounded overflow-x-auto text-sm">
            <code>{source}</code>
          </pre>
        </div>
      )}
      
      {issues && issues.length > 0 && <IssueDisplay issues={issues} />}
      
      {metrics && <MetricsDisplay metrics={metrics} />}
      
      {type === 'function' && function_info && <FunctionInfoDisplay info={function_info} />}
      
      {type === 'class' && class_info && <ClassInfoDisplay info={class_info} />}
      
      {dependencies && dependencies.length > 0 && <DependenciesDisplay dependencies={dependencies} />}
      
      {usages && usages.length > 0 && <UsagesDisplay usages={usages} />}
    </div>
  );
};

export default SymbolContext;

