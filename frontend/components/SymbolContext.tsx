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
  FaLayerGroup
} from 'react-icons/fa';

interface Metrics {
  cyclomatic_complexity: {
    value: number;
    rank: string;
  };
  halstead_metrics: {
    volume: number;
    unique_operators: number;
    unique_operands: number;
    total_operators: number;
    total_operands: number;
  };
  maintainability_index: {
    value: number;
    rank: string;
  };
  lines_of_code: {
    total: number;
    code: number;
    comments: number;
  };
}

interface UsageStats {
  total_usages: number;
  usage_breakdown: {
    functions: number;
    classes: number;
    global_vars: number;
    interfaces: number;
  };
  imports: {
    total: number;
    breakdown: {
      functions: number;
      classes: number;
      global_vars: number;
      interfaces: number;
      external_modules: number;
      files: number;
    };
  };
}

interface FunctionInfo {
  return_statements: number;
  parameters: Array<{
    name: string;
    type: string | null;
    default_value: string | null;
  }>;
  function_calls: Array<{
    name: string;
    args: string[];
    line: number;
  }>;
  call_sites: Array<{
    caller: string | null;
    line: number;
    file: string | null;
  }>;
  decorators: string[];
  dependencies: Array<{
    name: string;
    type: string;
    filepath: string | null;
  }>;
}

interface ClassInfo {
  parent_classes: string[];
  methods: Array<{
    name: string;
    parameters: number;
    line: number;
  }>;
  attributes: Array<{
    name: string;
    type: string | null;
    line: number;
  }>;
  decorators: string[];
  dependencies: Array<{
    name: string;
    type: string;
    filepath: string | null;
  }>;
  inheritance_depth: number;
  inheritance_chain: Array<{
    name: string;
    filepath: string | null;
  }>;
}

interface SymbolContextProps {
  id: string;
  name: string;
  type: string;
  filepath: string;
  start_line: number;
  end_line: number;
  source: string | null;
  usage_stats: UsageStats;
  metrics?: Metrics;
  function_info?: FunctionInfo;
  class_info?: ClassInfo;
}

const MetricsDisplay: React.FC<{ metrics: Metrics }> = ({ metrics }) => (
  <div className="mt-4">
    <h3 className="text-lg font-semibold flex items-center">
      <FaChartLine className="mr-2" />
      Metrics
    </h3>
    <div className="grid grid-cols-2 gap-4 mt-2">
      <div className="p-3 bg-gray-50 rounded">
        <h4 className="font-medium">Cyclomatic Complexity</h4>
        <p className="text-2xl">
          {metrics.cyclomatic_complexity.value}{' '}
          <span className={`text-sm ${
            metrics.cyclomatic_complexity.rank === 'A' ? 'text-green-500' :
            metrics.cyclomatic_complexity.rank === 'B' ? 'text-blue-500' :
            metrics.cyclomatic_complexity.rank === 'C' ? 'text-yellow-500' :
            'text-red-500'
          }`}>
            (Rank {metrics.cyclomatic_complexity.rank})
          </span>
        </p>
      </div>
      <div className="p-3 bg-gray-50 rounded">
        <h4 className="font-medium">Maintainability Index</h4>
        <p className="text-2xl">
          {metrics.maintainability_index.value}{' '}
          <span className={`text-sm ${
            metrics.maintainability_index.rank === 'A' ? 'text-green-500' :
            metrics.maintainability_index.rank === 'B' ? 'text-blue-500' :
            metrics.maintainability_index.rank === 'C' ? 'text-yellow-500' :
            'text-red-500'
          }`}>
            (Rank {metrics.maintainability_index.rank})
          </span>
        </p>
      </div>
      <div className="p-3 bg-gray-50 rounded">
        <h4 className="font-medium">Halstead Metrics</h4>
        <div className="text-sm">
          <p>Volume: {metrics.halstead_metrics.volume.toFixed(2)}</p>
          <p>Unique Operators: {metrics.halstead_metrics.unique_operators}</p>
          <p>Unique Operands: {metrics.halstead_metrics.unique_operands}</p>
          <p>Total Operators: {metrics.halstead_metrics.total_operators}</p>
          <p>Total Operands: {metrics.halstead_metrics.total_operands}</p>
        </div>
      </div>
      <div className="p-3 bg-gray-50 rounded">
        <h4 className="font-medium">Lines of Code</h4>
        <div className="text-sm">
          <p>Total: {metrics.lines_of_code.total}</p>
          <p>Code: {metrics.lines_of_code.code}</p>
          <p>Comments: {metrics.lines_of_code.comments}</p>
        </div>
      </div>
    </div>
  </div>
);

const UsageStatsDisplay: React.FC<{ stats: UsageStats }> = ({ stats }) => (
  <div className="mt-4">
    <h3 className="text-lg font-semibold flex items-center">
      <FaLink className="mr-2" />
      Usage Statistics
    </h3>
    <div className="grid grid-cols-2 gap-4 mt-2">
      <div className="p-3 bg-gray-50 rounded">
        <h4 className="font-medium">Direct Usage ({stats.total_usages})</h4>
        <div className="text-sm">
          <p>Functions: {stats.usage_breakdown.functions}</p>
          <p>Classes: {stats.usage_breakdown.classes}</p>
          <p>Global Variables: {stats.usage_breakdown.global_vars}</p>
          <p>Interfaces: {stats.usage_breakdown.interfaces}</p>
        </div>
      </div>
      <div className="p-3 bg-gray-50 rounded">
        <h4 className="font-medium">Import Usage ({stats.imports.total})</h4>
        <div className="text-sm">
          <p>Functions: {stats.imports.breakdown.functions}</p>
          <p>Classes: {stats.imports.breakdown.classes}</p>
          <p>Global Variables: {stats.imports.breakdown.global_vars}</p>
          <p>Interfaces: {stats.imports.breakdown.interfaces}</p>
          <p>External Modules: {stats.imports.breakdown.external_modules}</p>
          <p>Files: {stats.imports.breakdown.files}</p>
        </div>
      </div>
    </div>
  </div>
);

const FunctionInfoDisplay: React.FC<{ info: FunctionInfo }> = ({ info }) => (
  <div className="mt-4">
    <h3 className="text-lg font-semibold flex items-center">
      <FaCode className="mr-2" />
      Function Details
    </h3>
    <div className="grid grid-cols-2 gap-4 mt-2">
      <div className="p-3 bg-gray-50 rounded">
        <h4 className="font-medium">Parameters ({info.parameters.length})</h4>
        <div className="text-sm">
          {info.parameters.map((param, i) => (
            <p key={i}>
              {param.name}
              {param.type && <span className="text-gray-500">: {param.type}</span>}
              {param.default_value && (
                <span className="text-gray-500"> = {param.default_value}</span>
              )}
            </p>
          ))}
        </div>
      </div>
      <div className="p-3 bg-gray-50 rounded">
        <h4 className="font-medium">Function Calls ({info.function_calls.length})</h4>
        <div className="text-sm max-h-40 overflow-y-auto">
          {info.function_calls.map((call, i) => (
            <p key={i}>
              {call.name}
              <span className="text-gray-500"> @ L{call.line}</span>
            </p>
          ))}
        </div>
      </div>
      <div className="p-3 bg-gray-50 rounded">
        <h4 className="font-medium">Call Sites ({info.call_sites.length})</h4>
        <div className="text-sm max-h-40 overflow-y-auto">
          {info.call_sites.map((site, i) => (
            <p key={i}>
              {site.caller || 'Unknown'}
              <span className="text-gray-500"> @ {site.file}:L{site.line}</span>
            </p>
          ))}
        </div>
      </div>
      <div className="p-3 bg-gray-50 rounded">
        <h4 className="font-medium">Dependencies ({info.dependencies.length})</h4>
        <div className="text-sm max-h-40 overflow-y-auto">
          {info.dependencies.map((dep, i) => (
            <p key={i}>
              {dep.name}
              <span className="text-gray-500"> ({dep.type})</span>
            </p>
          ))}
        </div>
      </div>
    </div>
    {info.decorators.length > 0 && (
      <div className="mt-2 p-3 bg-gray-50 rounded">
        <h4 className="font-medium">Decorators</h4>
        <div className="text-sm">
          {info.decorators.map((dec, i) => (
            <p key={i}>{dec}</p>
          ))}
        </div>
      </div>
    )}
  </div>
);

const ClassInfoDisplay: React.FC<{ info: ClassInfo }> = ({ info }) => (
  <div className="mt-4">
    <h3 className="text-lg font-semibold flex items-center">
      <FaCube className="mr-2" />
      Class Details
    </h3>
    <div className="grid grid-cols-2 gap-4 mt-2">
      <div className="p-3 bg-gray-50 rounded">
        <h4 className="font-medium">Methods ({info.methods.length})</h4>
        <div className="text-sm max-h-40 overflow-y-auto">
          {info.methods.map((method, i) => (
            <p key={i}>
              {method.name}
              <span className="text-gray-500">
                ({method.parameters} params) @ L{method.line}
              </span>
            </p>
          ))}
        </div>
      </div>
      <div className="p-3 bg-gray-50 rounded">
        <h4 className="font-medium">Attributes ({info.attributes.length})</h4>
        <div className="text-sm max-h-40 overflow-y-auto">
          {info.attributes.map((attr, i) => (
            <p key={i}>
              {attr.name}
              {attr.type && <span className="text-gray-500">: {attr.type}</span>}
              <span className="text-gray-500"> @ L{attr.line}</span>
            </p>
          ))}
        </div>
      </div>
      <div className="p-3 bg-gray-50 rounded">
        <h4 className="font-medium">Inheritance</h4>
        <div className="text-sm">
          <p>Depth: {info.inheritance_depth}</p>
          <div className="mt-1">
            {info.inheritance_chain.map((cls, i) => (
              <div key={i} className="flex items-center">
                {i > 0 && <FaArrowRight className="mx-1 text-gray-400" />}
                <span>{cls.name}</span>
              </div>
            ))}
          </div>
        </div>
      </div>
      <div className="p-3 bg-gray-50 rounded">
        <h4 className="font-medium">Dependencies ({info.dependencies.length})</h4>
        <div className="text-sm max-h-40 overflow-y-auto">
          {info.dependencies.map((dep, i) => (
            <p key={i}>
              {dep.name}
              <span className="text-gray-500"> ({dep.type})</span>
            </p>
          ))}
        </div>
      </div>
    </div>
    {info.decorators.length > 0 && (
      <div className="mt-2 p-3 bg-gray-50 rounded">
        <h4 className="font-medium">Decorators</h4>
        <div className="text-sm">
          {info.decorators.map((dec, i) => (
            <p key={i}>{dec}</p>
          ))}
        </div>
      </div>
    )}
  </div>
);

export const SymbolContext: React.FC<SymbolContextProps> = ({
  name,
  type,
  filepath,
  start_line,
  end_line,
  source,
  usage_stats,
  metrics,
  function_info,
  class_info,
}) => {
  return (
    <div className="p-4">
      <div className="flex items-start justify-between">
        <div>
          <h2 className="text-2xl font-bold flex items-center">
            <FaCode className="mr-2" />
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
            <FaLayerGroup className="mr-2" />
            Source
          </h3>
          <pre className="mt-2 p-4 bg-gray-50 rounded overflow-x-auto">
            <code>{source}</code>
          </pre>
        </div>
      )}

      <UsageStatsDisplay stats={usage_stats} />
      
      {metrics && <MetricsDisplay metrics={metrics} />}
      
      {function_info && <FunctionInfoDisplay info={function_info} />}
      
      {class_info && <ClassInfoDisplay info={class_info} />}
    </div>
  );
};

export default SymbolContext;

