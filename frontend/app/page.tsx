import type { Metadata } from "next"
import Main from "./main"

export const metadata: Metadata = {
  title: "Codebase Analytics",
  description: "Comprehensive codebase analysis and visualization platform",
}

export default function Page() {
  return <Main />
}

