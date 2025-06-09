import './globals.css'
import type { Metadata } from 'next'
import type React from 'react'

export const metadata: Metadata = {
  title: 'Codebase Analytics Dashboard',
  description: 'Advanced repository analysis with GitHub integration',
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body className="min-h-screen bg-background font-sans antialiased">
        {children}
      </body>
    </html>
  )
}

