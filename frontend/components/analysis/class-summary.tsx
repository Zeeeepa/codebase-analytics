"use client"

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Code, AlertCircle } from "lucide-react"
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/components/ui/collapsible"
import { useState } from "react"

interface ClassSummaryProps {
  classSummaries: Array<{
    class_name: string;
    file_path: string;
    summary: string;
    error?: string;
  }>;
}

export function ClassSummary({ classSummaries }: ClassSummaryProps) {
  const [openClasses, setOpenClasses] = useState<Set<string>>(new Set())

  const toggleClass = (className: string) => {
    const newOpenClasses = new Set(openClasses)
    if (newOpenClasses.has(className)) {
      newOpenClasses.delete(className)
    } else {
      newOpenClasses.add(className)
    }
    setOpenClasses(newOpenClasses)
  }

  if (!classSummaries || classSummaries.length === 0) {
    return (
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Code className="h-5 w-5" />
            Class Analysis
          </CardTitle>
          <CardDescription>No class summaries available</CardDescription>
        </CardHeader>
      </Card>
    )
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Code className="h-5 w-5" />
          Class Analysis
        </CardTitle>
        <CardDescription>
          Deep analysis of class definitions and relationships
        </CardDescription>
      </CardHeader>
      <CardContent>
        <div className="space-y-2 max-h-96 overflow-y-auto">
          {classSummaries.map((cls, index) => (
            <Collapsible key={index}>
              <CollapsibleTrigger
                className="flex items-center justify-between w-full p-3 text-left bg-blue-50 hover:bg-blue-100 rounded-lg border cursor-pointer"
                onClick={() => toggleClass(cls.class_name)}
              >
                <div className="flex items-center gap-2">
                  {cls.error ? (
                    <AlertCircle className="h-4 w-4 text-red-500" />
                  ) : (
                    <Code className="h-4 w-4 text-blue-600" />
                  )}
                  <div>
                    <span className="font-mono text-sm font-semibold">{cls.class_name}</span>
                    <div className="text-xs text-gray-500">{cls.file_path}</div>
                  </div>
                </div>
                <span className="text-xs text-gray-500">
                  {openClasses.has(cls.class_name) ? '−' : '+'}
                </span>
              </CollapsibleTrigger>
              <CollapsibleContent className="mt-2">
                <div className="pl-6 pr-3 pb-3">
                  {cls.error ? (
                    <p className="text-red-600 text-sm">{cls.error}</p>
                  ) : (
                    <pre className="whitespace-pre-wrap text-xs bg-white p-3 rounded border">
                      {cls.summary}
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

