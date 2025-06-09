"use client"

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Settings, AlertCircle } from "lucide-react"
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/components/ui/collapsible"
import { useState } from "react"

interface FunctionSummaryProps {
  functionSummaries: Array<{
    function_name: string;
    file_path: string;
    summary: string;
    error?: string;
  }>;
}

export function FunctionSummary({ functionSummaries }: FunctionSummaryProps) {
  const [openFunctions, setOpenFunctions] = useState<Set<string>>(new Set())

  const toggleFunction = (functionName: string) => {
    const newOpenFunctions = new Set(openFunctions)
    if (newOpenFunctions.has(functionName)) {
      newOpenFunctions.delete(functionName)
    } else {
      newOpenFunctions.add(functionName)
    }
    setOpenFunctions(newOpenFunctions)
  }

  if (!functionSummaries || functionSummaries.length === 0) {
    return (
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Settings className="h-5 w-5" />
            Function Analysis
          </CardTitle>
          <CardDescription>No function summaries available</CardDescription>
        </CardHeader>
      </Card>
    )
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Settings className="h-5 w-5" />
          Function Analysis
        </CardTitle>
        <CardDescription>
          Comprehensive function analysis including call patterns
        </CardDescription>
      </CardHeader>
      <CardContent>
        <div className="space-y-2 max-h-96 overflow-y-auto">
          {functionSummaries.map((func, index) => (
            <Collapsible key={index}>
              <CollapsibleTrigger
                className="flex items-center justify-between w-full p-3 text-left bg-green-50 hover:bg-green-100 rounded-lg border cursor-pointer"
                onClick={() => toggleFunction(func.function_name)}
              >
                <div className="flex items-center gap-2">
                  {func.error ? (
                    <AlertCircle className="h-4 w-4 text-red-500" />
                  ) : (
                    <Settings className="h-4 w-4 text-green-600" />
                  )}
                  <div>
                    <span className="font-mono text-sm font-semibold">{func.function_name}</span>
                    <div className="text-xs text-gray-500">{func.file_path}</div>
                  </div>
                </div>
                <span className="text-xs text-gray-500">
                  {openFunctions.has(func.function_name) ? '−' : '+'}
                </span>
              </CollapsibleTrigger>
              <CollapsibleContent className="mt-2">
                <div className="pl-6 pr-3 pb-3">
                  {func.error ? (
                    <p className="text-red-600 text-sm">{func.error}</p>
                  ) : (
                    <pre className="whitespace-pre-wrap text-xs bg-white p-3 rounded border">
                      {func.summary}
                    </pre>
                  )}
                </div>
              </CollapsibleContent>
            </Collapsible>
          ))}
        </div>
      </CardContent>
    </Card>
  )
}

