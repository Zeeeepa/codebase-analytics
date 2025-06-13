import type { Metadata } from "next"
import InteractiveCodebaseExplorer from "@/components/interactive-codebase-explorer"

export const metadata: Metadata = {
  title: "Interactive Codebase Analytics",
  description: "Interactive visual analysis for codebase functional errors and parameter issues",
}

export default function Page() {
  return <InteractiveCodebaseExplorer />
}
