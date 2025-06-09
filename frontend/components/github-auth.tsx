"use client"

import { useState, useEffect } from 'react'
import { useSession, signIn, signOut } from 'next-auth/react'
import { Button } from '@/components/ui/button'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { GitHubRepo } from '@/lib/github'
import { Github, LogOut, Star, GitFork, Eye } from 'lucide-react'

interface GitHubAuthProps {
  onRepoSelect: (repo: GitHubRepo) => void
  selectedRepo?: GitHubRepo | null
}

export function GitHubAuth({ onRepoSelect, selectedRepo }: GitHubAuthProps) {
  const { data: session, status } = useSession()
  const [repos, setRepos] = useState<GitHubRepo[]>([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    if (session?.user) {
      fetchRepos()
    }
  }, [session])

  const fetchRepos = async () => {
    setLoading(true)
    setError(null)
    try {
      const response = await fetch('/api/github/repos')
      if (!response.ok) {
        throw new Error('Failed to fetch repositories')
      }
      const repoData = await response.json()
      setRepos(repoData)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch repositories')
    } finally {
      setLoading(false)
    }
  }

  const handleSignIn = () => {
    signIn('github')
  }

  const handleSignOut = () => {
    signOut()
    setRepos([])
    setError(null)
  }

  const handleRepoSelect = (repoFullName: string) => {
    const repo = repos.find(r => r.full_name === repoFullName)
    if (repo) {
      onRepoSelect(repo)
    }
  }

  if (status === 'loading') {
    return (
      <Card className="w-full">
        <CardContent className="p-6">
          <div className="flex items-center justify-center">
            <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary"></div>
          </div>
        </CardContent>
      </Card>
    )
  }

  if (!session) {
    return (
      <Card className="w-full">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Github className="h-5 w-5" />
            Connect GitHub Account
          </CardTitle>
          <CardDescription>
            Connect your GitHub account to analyze your repositories
          </CardDescription>
        </CardHeader>
        <CardContent>
          <Button onClick={handleSignIn} className="w-full" size="lg">
            <Github className="mr-2 h-4 w-4" />
            Sign in with GitHub
          </Button>
        </CardContent>
      </Card>
    )
  }

  return (
    <Card className="w-full">
      <CardHeader>
        <div className="flex items-center justify-between">
          <div>
            <CardTitle className="flex items-center gap-2">
              <Github className="h-5 w-5" />
              GitHub Repositories
            </CardTitle>
            <CardDescription>
              Signed in as {session.user?.name || session.user?.email}
            </CardDescription>
          </div>
          <Button variant="outline" size="sm" onClick={handleSignOut}>
            <LogOut className="mr-2 h-4 w-4" />
            Sign Out
          </Button>
        </div>
      </CardHeader>
      <CardContent className="space-y-4">
        {error && (
          <div className="p-3 text-sm text-red-600 bg-red-50 dark:bg-red-900/20 dark:text-red-400 rounded-md">
            {error}
          </div>
        )}
        
        {loading ? (
          <div className="flex items-center justify-center py-8">
            <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary"></div>
          </div>
        ) : (
          <div className="space-y-4">
            <div>
              <label className="text-sm font-medium mb-2 block">
                Select Repository to Analyze
              </label>
              <Select 
                value={selectedRepo?.full_name || ''} 
                onValueChange={handleRepoSelect}
              >
                <SelectTrigger className="w-full">
                  <SelectValue placeholder="Choose a repository..." />
                </SelectTrigger>
                <SelectContent>
                  {repos.map((repo) => (
                    <SelectItem key={repo.id} value={repo.full_name}>
                      <div className="flex items-center justify-between w-full">
                        <div className="flex items-center gap-2">
                          <span className="font-medium">{repo.name}</span>
                          {repo.private && (
                            <Badge variant="secondary" className="text-xs">
                              Private
                            </Badge>
                          )}
                        </div>
                        <div className="flex items-center gap-2 text-xs text-muted-foreground ml-4">
                          {repo.language && (
                            <Badge variant="outline" className="text-xs">
                              {repo.language}
                            </Badge>
                          )}
                          <div className="flex items-center gap-1">
                            <Star className="h-3 w-3" />
                            {repo.stargazers_count}
                          </div>
                          <div className="flex items-center gap-1">
                            <GitFork className="h-3 w-3" />
                            {repo.forks_count}
                          </div>
                        </div>
                      </div>
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>

            {selectedRepo && (
              <Card className="bg-muted/50">
                <CardContent className="p-4">
                  <div className="space-y-2">
                    <div className="flex items-center justify-between">
                      <h4 className="font-medium">{selectedRepo.name}</h4>
                      <div className="flex items-center gap-2">
                        {selectedRepo.private && (
                          <Badge variant="secondary">Private</Badge>
                        )}
                        {selectedRepo.language && (
                          <Badge variant="outline">{selectedRepo.language}</Badge>
                        )}
                      </div>
                    </div>
                    {selectedRepo.description && (
                      <p className="text-sm text-muted-foreground">
                        {selectedRepo.description}
                      </p>
                    )}
                    <div className="flex items-center gap-4 text-xs text-muted-foreground">
                      <div className="flex items-center gap-1">
                        <Star className="h-3 w-3" />
                        {selectedRepo.stargazers_count} stars
                      </div>
                      <div className="flex items-center gap-1">
                        <GitFork className="h-3 w-3" />
                        {selectedRepo.forks_count} forks
                      </div>
                      <div className="flex items-center gap-1">
                        <Eye className="h-3 w-3" />
                        <a 
                          href={selectedRepo.html_url} 
                          target="_blank" 
                          rel="noopener noreferrer"
                          className="hover:underline"
                        >
                          View on GitHub
                        </a>
                      </div>
                    </div>
                  </div>
                </CardContent>
              </Card>
            )}

            <div className="text-xs text-muted-foreground">
              Found {repos.length} repositories
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  )
}
