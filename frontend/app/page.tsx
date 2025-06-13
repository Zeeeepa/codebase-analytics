import type { Metadata } from "next"
import CodebaseDashboard from "@/components/codebase-dashboard"

export const metadata: Metadata = {
  title: "Codebase Analytics",
  description: "Comprehensive codebase analysis and visualization platform",
}

export default function Page() {
  return <CodebaseDashboard />
}

