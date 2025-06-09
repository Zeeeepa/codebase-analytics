"use client"

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Brain, AlertCircle } from "lucide-react"
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/components/ui/collapsible"
import { useState } from "react"

interface SymbolSummaryProps {
  symbolSummaries: Array<{
    symbol_name: string;
    symbol_type: string;
    summary: string;
    error?: string;
  }>;
}

export function SymbolSummary({ symbolSummaries }: SymbolSummaryProps) {
  const [openSymbols, setOpenSymbols] = useState<Set<string>>(new Set())

  const toggleSymbol = (symbolName: string) => {
    const newOpenSymbols = new Set(openSymbols)
    if (newOpenSymbols.has(symbolName)) {
      newOpenSymbols.delete(symbolName)
    } else {
      newOpenSymbols.add(symbolName)
    }
    setOpenSymbols(newOpenSymbols)
  }

  const getSymbolTypeColor = (type: string) => {
    switch (type.toLowerCase()) {
      case 'function': return 'text-green-600 bg-green-50'
      case 'class': return 'text-blue-600 bg-blue-50'
      case 'variable': return 'text-purple-600 bg-purple-50'
      case 'interface': return 'text-orange-600 bg-orange-50'
      default: return 'text-gray-600 bg-gray-50'
    }
  }

  if (!symbolSummaries || symbolSummaries.length === 0) {
    return (
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Brain className="h-5 w-5" />
            Symbol Analysis
          </CardTitle>
          <CardDescription>No symbol summaries available</CardDescription>
        </CardHeader>
      </Card>
    )
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Brain className="h-5 w-5" />
          Symbol Analysis
        </CardTitle>
        <CardDescription>
          Universal symbol usage analysis across the codebase
        </CardDescription>
      </CardHeader>
      <CardContent>
        <div className="space-y-2 max-h-96 overflow-y-auto">
          {symbolSummaries.map((symbol, index) => (
            <Collapsible key={index}>
              <CollapsibleTrigger
                className="flex items-center justify-between w-full p-3 text-left bg-purple-50 hover:bg-purple-100 rounded-lg border cursor-pointer"
                onClick={() => toggleSymbol(symbol.symbol_name)}
              >
                <div className="flex items-center gap-2">
                  {symbol.error ? (
                    <AlertCircle className="h-4 w-4 text-red-500" />
                  ) : (
                    <Brain className="h-4 w-4 text-purple-600" />
                  )}
                  <div>
                    <div className="flex items-center gap-2">
                      <span className="font-mono text-sm font-semibold">{symbol.symbol_name}</span>
                      <span className={`text-xs px-2 py-1 rounded ${getSymbolTypeColor(symbol.symbol_type)}`}>
                        {symbol.symbol_type}
                      </span>
                    </div>
                  </div>
                </div>
                <span className="text-xs text-gray-500">
                  {openSymbols.has(symbol.symbol_name) ? '−' : '+'}
                </span>
              </CollapsibleTrigger>
              <CollapsibleContent className="mt-2">
                <div className="pl-6 pr-3 pb-3">
                  {symbol.error ? (
                    <p className="text-red-600 text-sm">{symbol.error}</p>
                  ) : (
                    <pre className="whitespace-pre-wrap text-xs bg-white p-3 rounded border">
                      {symbol.summary}
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

