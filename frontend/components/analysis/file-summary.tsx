"use client"

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { FileCode2, AlertCircle } from "lucide-react"
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/components/ui/collapsible"
import { useState } from "react"

interface FileSummaryProps {
  fileSummaries: Array<{
    file_path: string;
    summary: string;
    error?: string;
  }>;
}

export function FileSummary({ fileSummaries }: FileSummaryProps) {
  const [openFiles, setOpenFiles] = useState<Set<string>>(new Set())

  const toggleFile = (filePath: string) => {
    const newOpenFiles = new Set(openFiles)
    if (newOpenFiles.has(filePath)) {
      newOpenFiles.delete(filePath)
    } else {
      newOpenFiles.add(filePath)
    }
    setOpenFiles(newOpenFiles)
  }

  if (!fileSummaries || fileSummaries.length === 0) {
    return (
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <FileCode2 className="h-5 w-5" />
            File Analysis
          </CardTitle>
          <CardDescription>No file summaries available</CardDescription>
        </CardHeader>
      </Card>
    )
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <FileCode2 className="h-5 w-5" />
          File Analysis
        </CardTitle>
        <CardDescription>
          Per-file dependency analysis and structure overview
        </CardDescription>
      </CardHeader>
      <CardContent>
        <div className="space-y-2 max-h-96 overflow-y-auto">
          {fileSummaries.map((file, index) => (
            <Collapsible key={index}>
              <CollapsibleTrigger
                className="flex items-center justify-between w-full p-3 text-left bg-gray-50 hover:bg-gray-100 rounded-lg border cursor-pointer"
                onClick={() => toggleFile(file.file_path)}
              >
                <div className="flex items-center gap-2">
                  {file.error ? (
                    <AlertCircle className="h-4 w-4 text-red-500" />
                  ) : (
                    <FileCode2 className="h-4 w-4" />
                  )}
                  <span className="font-mono text-sm">{file.file_path}</span>
                </div>
                <span className="text-xs text-gray-500">
                  {openFiles.has(file.file_path) ? '−' : '+'}
                </span>
              </CollapsibleTrigger>
              <CollapsibleContent className="mt-2">
                <div className="pl-6 pr-3 pb-3">
                  {file.error ? (
                    <p className="text-red-600 text-sm">{file.error}</p>
                  ) : (
                    <pre className="whitespace-pre-wrap text-xs bg-white p-3 rounded border">
                      {file.summary}
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

