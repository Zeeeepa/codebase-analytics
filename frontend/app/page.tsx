'use client'

import { SessionProvider } from 'next-auth/react'
import EnhancedAnalyticsDashboard from '@/components/enhanced-analytics-dashboard'

export default function Home() {
  return (
    <SessionProvider>
      <main>
        <EnhancedAnalyticsDashboard />
      </main>
    </SessionProvider>
  )
}

