"use client"

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Code2, FileText } from "lucide-react"

interface CodebaseSummaryProps {
  summary: string;
  error?: string;
}

export function CodebaseSummary({ summary, error }: CodebaseSummaryProps) {
  if (error) {
    return (
      <Card className="border-red-200 bg-red-50">
        <CardHeader>
          <CardTitle className="flex items-center gap-2 text-red-700">
            <Code2 className="h-5 w-5" />
            Codebase Summary - Error
          </CardTitle>
        </CardHeader>
        <CardContent>
          <p className="text-red-600">{error}</p>
        </CardContent>
      </Card>
    )
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Code2 className="h-5 w-5" />
          Codebase Overview
        </CardTitle>
        <CardDescription>
          High-level statistical overview of the entire codebase
        </CardDescription>
      </CardHeader>
      <CardContent>
        <div className="space-y-2">
          <pre className="whitespace-pre-wrap text-sm bg-gray-50 p-4 rounded-lg border">
            {summary}
          </pre>
        </div>
      </CardContent>
    </Card>
  )
}

