import "@/styles/globals.css"
import type { Metadata } from "next"
import type React from "react" // Import React

import { ThemeProvider } from "@/components/theme-provider"
import { Inter } from 'next/font/google'

// Load Inter font
const inter = Inter({
  subsets: ['latin'],
  display: 'swap',
  variable: '--font-inter',
})

export const metadata: Metadata = {
  title: "Codebase Analytics Dashboard",
  description: "Analytics dashboard for public GitHub repositories",
  icons: {
    icon: '/favicon.ico',
  },
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en" suppressHydrationWarning className={inter.variable}>
      <body className="min-h-screen antialiased">
        <ThemeProvider attribute="class" defaultTheme="dark" enableSystem disableTransitionOnChange>
          {children}
        </ThemeProvider>
      </body>
    </html>
  )
}
