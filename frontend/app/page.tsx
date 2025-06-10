import type { Metadata } from "next"
import EnhancedAnalyticsDashboard from "@/components/enhanced-analytics-dashboard"

export const metadata: Metadata = {
  title: "Enhanced Codebase Analytics",
  description: "Comprehensive code quality analysis with rich visualizations and error detection",
}

export default function Page() {
  return <EnhancedAnalyticsDashboard />
}

